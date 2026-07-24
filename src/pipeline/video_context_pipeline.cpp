#include "pipeline/video_context_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>

#include "core/frame_extractor.hpp"
#include "core/llm_runtime.hpp"
#include "core/text_util.hpp"
#include "core/vision_encoder.hpp"

namespace vlm {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

VideoContextPipeline::VideoContextPipeline(ModelRegistry registry, PipelineConfig config)
    : registry_(std::move(registry)),
      config_(std::move(config)),
      extractor_(config_.ffmpeg_bin_path),
      transcriber_(std::make_unique<StubAudioTranscriber>())
{
}

bool VideoContextPipeline::initialize(std::optional<std::string_view> preload_model_id)
{
    if (registry_.models().empty()) {
        std::cerr << "Model registry is empty\n";
        return false;
    }

    if (preload_model_id && !preload_model_id->empty()) {
        if (!ensureModel(*preload_model_id)) {
            std::cerr << "Failed to preload model: " << *preload_model_id << '\n';
            return false;
        }
        if (config_.verbose) {
            std::cerr << "Preloaded model: " << loaded_model_id_ << '\n';
        }
    }

    ready_ = true;
    return true;
}

bool VideoContextPipeline::ensureModel(std::string_view model_id)
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        std::cerr << "Unknown model: " << model_id << '\n';
        return false;
    }

    if (loaded_model_id_ == *resolved && vision_.loaded() && llm_.loaded()) {
        return true;
    }

    if (config_.verbose) {
        std::cerr << "Loading model: " << *resolved;
        if (!loaded_model_id_.empty()) {
            std::cerr << " (replacing " << loaded_model_id_ << ')';
        }
        std::cerr << '\n';
    }

    vision_.unload();
    llm_.unload();

    const ModelSpec* spec = registry_.find(*resolved);
    if (!spec) {
        return false;
    }

    if (!vision_.load(spec->vision_model_path, config_.verbose)) {
        std::cerr << "Failed to load vision model for " << spec->id << '\n';
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
        std::cerr << "Failed to load LLM for " << spec->id << '\n';
        vision_.unload();
        return false;
    }

    loaded_model_id_ = spec->id;
    std::cerr << "Model loaded: " << loaded_model_id_ << '\n';
    return true;
}

void VideoContextPipeline::releaseModels()
{
    if (!vision_.loaded() && !llm_.loaded() && loaded_model_id_.empty()) {
        return;
    }
    const std::string prev = loaded_model_id_;
    vision_.unload();
    llm_.unload();
    loaded_model_id_.clear();
    if (!prev.empty()) {
        std::cerr << "Model unloaded: " << prev << std::endl;
    }
}

int VideoContextPipeline::resolveFrameBudget(const AnalyzeRequest& request) const
{
    const int cap = request.frame_budget > 0 ? request.frame_budget : config_.frame_budget;
    if (cap > 0) {
        return cap;
    }
    if (vision_.loaded()) {
        return vision_.computeFrameBudget(config_.default_context, config_.default_max_tokens,
                                          config_.prompt_reserve_tokens);
    }
    // Vision not loaded yet — estimate from typical Qwen3.5-VL tile count (~196 tokens/frame).
    constexpr int kDefaultImageTokens = 196;
    const int available =
        config_.default_context - config_.default_max_tokens - config_.prompt_reserve_tokens;
    return std::max(1, available / kDefaultImageTokens);
}

int VideoContextPipeline::effectiveMaxFrames(const AnalyzeRequest& request) const
{
    const int budget = resolveFrameBudget(request);
    int max_frames = request.max_frames;
    if (config_.absolute_max_frames > 0) {
        max_frames = std::min(max_frames, config_.absolute_max_frames);
    }
    return std::min(max_frames, budget);
}

AnalyzeResult VideoContextPipeline::analyze(const AnalyzeRequest& request)
{
    AnalyzeResult result;
    const auto t_total = Clock::now();
    auto finish = [&]() -> AnalyzeResult& {
        result.metrics["total_ms"] = elapsedMs(t_total);
        return result;
    };

    if (!ready_) {
        result.error = "Pipeline not initialized";
        return finish();
    }

    const std::string model_id =
        request.model.empty() ? registry_.defaultModelId() : request.model;

    llm_.setOutputLang(request.lang);
    const bool thinking =
        request.enable_thinking.value_or(config_.enable_thinking);
    llm_.setEnableThinking(thinking);
    std::cerr << "analyze: enable_thinking=" << (thinking ? "true" : "false")
              << " lang=" << request.lang << " prompt_mode=" << request.prompt_mode
              << " max_tokens=" << (request.max_tokens > 0 ? request.max_tokens
                                                           : config_.default_max_tokens)
              << (request.max_tokens > 0 ? " (request)" : " (config)")
              << " temperature="
              << (request.temperature ? *request.temperature : config_.temperature)
              << (request.temperature ? " (request)" : " (config)") << '\n';

    const int budget = resolveFrameBudget(request);
    const int effective = effectiveMaxFrames(request);
    result.frames_requested = request.max_frames;
    result.frame_budget = budget;
    result.frames_capped_by_context = effective < request.max_frames;

    struct PrepState {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<cv::Mat> pending_frames;
        bool extract_done = false;
        int extract_count = 0;
        VideoInfo video_info{};
        std::atomic<bool> model_ready{false};
        std::atomic<bool> model_load_finished{false};
        bool model_ok = false;
        double model_load_ms = 0;
        double extract_ms = 0;
        double transcript_ms = 0;
        TranscriptResult transcript;
    } prep;

    if (config_.verbose) {
        std::cerr << "parallel prep: model load + frame extract + transcript\n";
    }

    auto transcript_fut = std::async(std::launch::async, [&] {
        const auto t0 = Clock::now();
        prep.transcript =
            transcriber_->transcribe(request.video_path, request.transcript_override);
        prep.transcript_ms = elapsedMs(t0);
    });

    auto model_fut = std::async(std::launch::async, [&] {
        const auto t0 = Clock::now();
        const bool ok = ensureModel(model_id);
        prep.model_load_ms = elapsedMs(t0);
        prep.model_ok = ok;
        prep.model_ready = ok;
        prep.model_load_finished = true;
        prep.cv.notify_all();
        return ok;
    });

    FrameProgressCallback extract_progress;
    if (config_.verbose) {
        extract_progress = [](int cur, int total) {
            std::cerr << "\rFrame extract: " << cur << '/' << total;
            if (cur == total) {
                std::cerr << '\n';
            }
        };
    }

    auto extract_fut = std::async(std::launch::async, [&] {
        const auto t0 = Clock::now();
        VideoInfo info;
        const int got = extractor_.extractFramesStreaming(
            request.video_path.string(), effective,
            [&](cv::Mat frame, int /*index*/, int /*total*/) {
                std::lock_guard lock(prep.mu);
                prep.pending_frames.push_back(std::move(frame));
                ++prep.extract_count;
                prep.cv.notify_one();
            },
            extract_progress, &info);
        {
            std::lock_guard lock(prep.mu);
            prep.video_info = info;
            prep.extract_done = true;
            prep.cv.notify_all();
        }
        prep.extract_ms = elapsedMs(t0);
        return got;
    });

    vision_.clear();
    int encoded = 0;
    const auto t_encode = Clock::now();
    VisionProgressCallback vision_progress;
    if (config_.verbose) {
        vision_progress = [&](int cur, int total) {
            std::cerr << "\rVision encode: " << cur << '/' << total;
            if (cur == total) {
                std::cerr << '\n';
            }
        };
    }

    while (true) {
        std::unique_lock lock(prep.mu);
        prep.cv.wait(lock, [&] {
            if (prep.model_ready.load() && !prep.pending_frames.empty()) {
                return true;
            }
            if (prep.extract_done && prep.pending_frames.empty()) {
                return true;
            }
            if (prep.model_load_finished.load() && !prep.model_ok) {
                return true;
            }
            return false;
        });

        while (prep.model_ready.load() && !prep.pending_frames.empty()) {
            cv::Mat frame = std::move(prep.pending_frames.front());
            prep.pending_frames.pop_front();
            lock.unlock();
            if (vision_.appendFrame(frame)) {
                ++encoded;
                if (vision_progress) {
                    const int total = prep.extract_done ? prep.extract_count : effective;
                    vision_progress(encoded, std::max(total, encoded));
                }
            }
            lock.lock();
        }

        if (prep.extract_done && prep.pending_frames.empty()) {
            break;
        }
        if (prep.model_load_finished.load() && !prep.model_ok) {
            break;
        }
    }

    const int extracted = extract_fut.get();
    const bool model_ok = model_fut.get();
    transcript_fut.get();

    result.metrics["model_load_ms"] = prep.model_load_ms;
    result.metrics["frame_extract_ms"] = prep.extract_ms;
    result.metrics["transcript_ms"] = prep.transcript_ms;
    result.metrics["vision_encode_ms"] = elapsedMs(t_encode);
    result.transcript = std::move(prep.transcript);
    result.model = loaded_model_id_;
    result.duration_sec = prep.video_info.duration_sec;

    if (!model_ok) {
        result.error = "Failed to load model: " + model_id;
        return finish();
    }

    if (config_.verbose) {
        const VideoInfo& info = prep.video_info;
        const int planned = planFrameCount(info.duration_sec, info.fps, effective);
        const double total_frames = info.duration_sec * info.fps;
        const double step =
            planned > 0 ? total_frames / static_cast<double>(planned) : 0.0;
        std::cerr << "frame sampling: duration=" << info.duration_sec << "s fps=" << info.fps
                  << " requested=" << effective << " planned=" << planned
                  << " got=" << extracted << " encoded=" << encoded << '\n';
        if (planned > 0) {
            std::cerr << "  times_sec:";
            for (int i = 0; i < planned; ++i) {
                const int frame_idx = static_cast<int>(i * step);
                const double t = frame_idx / info.fps;
                std::cerr << ' ' << t;
            }
            std::cerr << '\n';
        }
    }

    if (encoded == 0) {
        result.error =
            extracted == 0 ? "No frames extracted from video" : "Vision encoding failed";
        return finish();
    }

    result.frames_used = encoded;
    const std::string prompt =
        LlmRuntime::buildUserVisionPrompt(request.lang, thinking, request.prompt_mode);
    const float temperature = request.temperature.value_or(-1.0f);
    {
        const auto t0 = Clock::now();
        llm_.clearKvCache();
        result.description = stripThinkingTags(
            llm_.generateMultimodal(prompt, vision_, request.max_tokens, temperature));
        result.metrics["llm_generate_ms"] = elapsedMs(t0);
        result.metrics["max_new_tokens"] = llm_.lastMaxNewTokens();
        result.metrics["generate_tokens"] = llm_.lastGenerateTokens();
        if (llm_.lastTruncatedByMaxTokens()) {
            result.metrics["truncated"] = 1;
        }
    }
    return finish();
}

}  // namespace vlm
