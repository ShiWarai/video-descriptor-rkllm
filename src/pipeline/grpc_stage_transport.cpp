#include "pipeline/grpc_stage_transport.hpp"

#include <iostream>

#include "grpc/tensor_codec.hpp"
#include "vlm/v1/worker.pb.h"

namespace vlm {

GrpcStageTransport::GrpcStageTransport(const ModelRegistry& registry, PipelineConfig config)
    : registry_(registry), config_(std::move(config))
{
    auto vision_targets =
        grpc_util::expandTargets(grpc_util::parseTargets("VISION_TARGETS", "vision-0:50051"));
    auto llm_targets =
        grpc_util::expandTargets(grpc_util::parseTargets("LLM_TARGETS", "llm-0:50052"));
    if (vision_targets.empty() || llm_targets.empty()) {
        throw std::runtime_error("VISION_TARGETS and LLM_TARGETS must be set for distributed runtime");
    }
    vision_pool_ = std::make_unique<VisionClientPool>(std::move(vision_targets));
    llm_pool_ = std::make_unique<LlmClientPool>(std::move(llm_targets));
    vision_pool_->connect();
    llm_pool_->connect();
}

GrpcStageTransport::~GrpcStageTransport()
{
    if (vision_pool_) {
        vision_pool_->close();
    }
    if (llm_pool_) {
        llm_pool_->close();
    }
}

bool GrpcStageTransport::distributed() const noexcept
{
    return true;
}

bool GrpcStageTransport::loadModel(std::string_view model_id, std::string* error_out)
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        if (error_out) {
            *error_out = "Unknown model: " + std::string(model_id);
        }
        return false;
    }
    loaded_model_id_ = *resolved;
    return true;
}

int GrpcStageTransport::planVisionWorkers(std::string_view model_id,
                                          std::string* error_out) const
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        if (error_out) {
            *error_out = "Unknown model: " + std::string(model_id);
        }
        return 0;
    }
    return VisionEncoder::kWorkerCount;
}

bool GrpcStageTransport::isModelLoaded(std::string_view model_id) const
{
    const auto resolved = registry_.resolveId(model_id);
    return resolved && loaded_model_id_ == *resolved;
}

const std::string& GrpcStageTransport::loadedModelId() const
{
    return loaded_model_id_;
}

int GrpcStageTransport::resolveFrameBudget(const PipelineConfig& config,
                                           int frame_budget_override) const
{
    if (frame_budget_override > 0) {
        return frame_budget_override;
    }
    if (config.frame_budget > 0) {
        return config.frame_budget;
    }
    constexpr int kDefaultImageTokens = 196;
    const int available =
        config.default_context - config.default_max_tokens - config.prompt_reserve_tokens;
    return std::max(1, available / kDefaultImageTokens);
}

VisionEncodeResult GrpcStageTransport::encodeStreaming(VisionEncodeQueue& queue, int total_hint,
                                                       VisionProgressCallback progress)
{
    (void)queue;
    (void)total_hint;
    (void)progress;
    VisionEncodeResult result;
    result.error = "encodeStreaming is not supported in distributed mode";
    return result;
}

LlmGenerateResult GrpcStageTransport::generate(const std::string& prompt,
                                               const VisionEncodeResult& vision,
                                               int max_new_tokens, float temperature,
                                               bool enable_thinking, std::string_view lang)
{
    (void)prompt;
    (void)vision;
    (void)max_new_tokens;
    (void)temperature;
    (void)enable_thinking;
    (void)lang;
    LlmGenerateResult result;
    result.error = "generate is not supported in distributed mode";
    return result;
}

EncodeThenGenerateResult GrpcStageTransport::encodeThenGenerate(
    const std::vector<PendingVisionFrame>& frames, const EncodeThenGenerateParams& params,
    VisionProgressCallback progress)
{
    EncodeThenGenerateResult result;
    if (!loadModel(params.model_id, &result.error)) {
        return result;
    }

    const std::string llm_target = llm_pool_->acquireTarget();
    vlm::v1::EncodeThenGenerateRequest request;
    request.set_job_id(params.job_id);
    request.set_model_id(params.model_id);
    *request.mutable_frames() = grpc_util::framesToBatch(frames);
    request.set_llm_target(llm_target);
    request.set_prompt(params.prompt);
    request.set_max_new_tokens(params.max_new_tokens);
    request.set_temperature(params.temperature);
    request.set_enable_thinking(params.enable_thinking);
    request.set_lang(params.lang);

    const auto response = vision_pool_->encodeThenGenerate(request);
    llm_pool_->releaseTarget(llm_target);

    if (!response.error().empty()) {
        result.error = response.error();
        return result;
    }

    result.text = response.text();
    result.max_new_tokens = response.max_new_tokens();
    result.generate_tokens = response.generate_tokens();
    result.truncated = response.truncated();
    result.encode_ms = response.encode_ms();
    result.generate_ms = response.generate_ms();
    result.n_image = response.n_image();

    if (progress && result.n_image > 0) {
        progress(result.n_image, result.n_image);
    }
    return result;
}

void GrpcStageTransport::clearVision() {}

void GrpcStageTransport::releaseModels()
{
    loaded_model_id_.clear();
}

}  // namespace vlm
