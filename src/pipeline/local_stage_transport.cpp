#include "pipeline/local_stage_transport.hpp"

#include <iostream>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "core/system_memory.hpp"

namespace vlm {

LocalStageTransport::LocalStageTransport(const ModelRegistry& registry, PipelineConfig config)
    : registry_(registry), config_(std::move(config))
{
}

bool LocalStageTransport::distributed() const noexcept
{
    return false;
}

bool LocalStageTransport::loadModel(std::string_view model_id, std::string* error_out)
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        const std::string msg = "Unknown model: " + std::string(model_id);
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    if (loaded_model_id_ == *resolved && vision_.loaded() && llm_.loaded()) {
        return true;
    }

    const ModelSpec* spec = registry_.find(*resolved);
    if (!spec) {
        if (error_out) {
            *error_out = "Model spec not found: " + *resolved;
        }
        return false;
    }

    const int workers = planVisionWorkers(*resolved, error_out);
    if (workers <= 0) {
        return false;
    }

    if (config_.verbose) {
        std::cerr << "Loading model: " << *resolved << " with " << workers << " vision worker(s)\n";
    }

    vision_.unload();
    llm_.unload();
    loaded_model_id_.clear();
#if defined(__GLIBC__)
    malloc_trim(0);
#endif

    if (!vision_.load(spec->vision_model_path, config_.verbose, workers)) {
        const std::string msg = "Failed to load vision model for " + spec->id;
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    const int top_k = spec->top_k.value_or(config_.top_k);
    const float top_p = spec->top_p.value_or(config_.top_p);
    const float temperature = spec->temperature.value_or(config_.temperature);
    const float presence = spec->presence_penalty.value_or(config_.presence_penalty);
    llm_.setSamplingDefaults(top_k, top_p, temperature, presence);
    llm_.setThinkingSamplingDefaults(config_.thinking_temperature, config_.thinking_top_p,
                                     config_.thinking_presence_penalty);
    if (!llm_.load(spec->llm_model_path, config_.default_max_tokens, config_.default_context,
                   config_.default_lang, config_.enable_thinking, config_.verbose)) {
        const std::string msg = "Failed to load LLM for " + spec->id;
        vision_.unload();
        if (error_out) {
            *error_out = msg;
        }
        return false;
    }

    loaded_model_id_ = spec->id;
    return true;
}

int LocalStageTransport::planVisionWorkers(std::string_view model_id,
                                           std::string* error_out) const
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        if (error_out) {
            *error_out = "Unknown model: " + std::string(model_id);
        }
        return 0;
    }
    if (loaded_model_id_ == *resolved && vision_.loaded() && llm_.loaded()) {
        return vision_.workerCount();
    }
    const ModelSpec* spec = registry_.find(*resolved);
    if (!spec) {
        if (error_out) {
            *error_out = "Model spec not found: " + *resolved;
        }
        return 0;
    }

    std::uint64_t credit = 0;
    if (!loaded_model_id_.empty() && loaded_model_id_ != *resolved && vision_.loaded() &&
        llm_.loaded()) {
        if (const ModelSpec* cur = registry_.find(loaded_model_id_)) {
            credit = estimateModelRamBytes(cur->llm_model_path, cur->vision_model_path,
                                           config_.default_context, vision_.workerCount());
        }
    }

    return pickVisionWorkerCount(spec->llm_model_path, spec->vision_model_path,
                                 config_.default_context, VisionEncoder::kWorkerCount, credit,
                                 error_out);
}

bool LocalStageTransport::isModelLoaded(std::string_view model_id) const
{
    const auto resolved = registry_.resolveId(model_id);
    return resolved && loaded_model_id_ == *resolved && vision_.loaded() && llm_.loaded();
}

const std::string& LocalStageTransport::loadedModelId() const
{
    return loaded_model_id_;
}

int LocalStageTransport::resolveFrameBudget(const PipelineConfig& config,
                                            int frame_budget_override) const
{
    if (frame_budget_override > 0) {
        return frame_budget_override;
    }
    if (config.frame_budget > 0) {
        return config.frame_budget;
    }
    if (vision_.loaded()) {
        return vision_.computeFrameBudget(config.default_context, config.default_max_tokens,
                                          config.prompt_reserve_tokens);
    }
    constexpr int kDefaultImageTokens = 196;
    const int available =
        config.default_context - config.default_max_tokens - config.prompt_reserve_tokens;
    return std::max(1, available / kDefaultImageTokens);
}

VisionEncodeResult LocalStageTransport::encodeStreaming(VisionEncodeQueue& queue, int total_hint,
                                                          VisionProgressCallback progress)
{
    VisionEncodeResult result;
    const auto t0 = std::chrono::steady_clock::now();
    const int encoded = vision_.encodeStreaming(queue, total_hint, progress);
    result.encode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (encoded <= 0) {
        result.error = "Vision encoding failed";
        return result;
    }
    result.embeddings = vision_.embeddings();
    result.frame_times = vision_.frameTimes();
    result.model_info = vision_.modelInfo();
    result.n_image = encoded;
    return result;
}

LlmGenerateResult LocalStageTransport::generate(const std::string& prompt,
                                                const VisionEncodeResult& vision,
                                                int max_new_tokens, float temperature,
                                                bool enable_thinking, std::string_view lang)
{
    LlmGenerateResult result;
    llm_.setOutputLang(lang);
    llm_.setEnableThinking(enable_thinking);
    const auto t0 = std::chrono::steady_clock::now();
    llm_.clearKvCache();
    result.text = llm_.generateMultimodal(prompt, vision.embeddings, vision.model_info,
                                          static_cast<std::size_t>(vision.n_image), max_new_tokens,
                                          temperature);
    result.generate_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    result.max_new_tokens = llm_.lastMaxNewTokens();
    result.generate_tokens = llm_.lastGenerateTokens();
    result.truncated = llm_.lastTruncatedByMaxTokens();
    llm_.clearKvCache();
    return result;
}

EncodeThenGenerateResult LocalStageTransport::encodeThenGenerate(
    const std::vector<PendingVisionFrame>& frames, const EncodeThenGenerateParams& params,
    VisionProgressCallback progress)
{
    EncodeThenGenerateResult result;
    std::string load_error;
    if (!loadModel(params.model_id, &load_error)) {
        result.error = load_error.empty() ? "Failed to load model" : load_error;
        return result;
    }

    vision_.clear();
    int encoded = 0;
    result.vision.frame_times.clear();
    for (const auto& pending : frames) {
        if (!vision_.appendFrame(pending.frame)) {
            result.error = "Vision appendFrame failed";
            return result;
        }
        result.vision.frame_times.push_back(pending.time_sec);
        ++encoded;
        if (progress) {
            progress(encoded, static_cast<int>(frames.size()));
        }
    }

    result.vision.embeddings = vision_.embeddings();
    result.vision.frame_times = vision_.frameTimes();
    result.vision.model_info = vision_.modelInfo();
    result.vision.n_image = encoded;

    auto gen = generate(params.prompt, result.vision, params.max_new_tokens, params.temperature,
                        params.enable_thinking, params.lang);
    if (!gen.ok()) {
        result.error = gen.error.empty() ? "LLM generation failed" : gen.error;
        return result;
    }

    result.text = std::move(gen.text);
    result.max_new_tokens = gen.max_new_tokens;
    result.generate_tokens = gen.generate_tokens;
    result.truncated = gen.truncated;
    result.generate_ms = gen.generate_ms;
    result.n_image = encoded;
    return result;
}

void LocalStageTransport::clearVision()
{
    vision_.clear();
}

void LocalStageTransport::releaseModels()
{
    vision_.unload();
    llm_.unload();
    loaded_model_id_.clear();
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

}  // namespace vlm
