#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/vision_encoder.hpp"
#include "pipeline/stage_types.hpp"
#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace vlm {

class StageTransport {
public:
    virtual ~StageTransport() = default;

    [[nodiscard]] virtual bool distributed() const noexcept = 0;

    [[nodiscard]] virtual bool loadModel(std::string_view model_id, std::string* error_out) = 0;
    [[nodiscard]] virtual int planVisionWorkers(std::string_view model_id,
                                                std::string* error_out) const = 0;
    [[nodiscard]] virtual bool isModelLoaded(std::string_view model_id) const = 0;
    [[nodiscard]] virtual const std::string& loadedModelId() const = 0;
    [[nodiscard]] virtual int resolveFrameBudget(const PipelineConfig& config,
                                                 int frame_budget_override) const = 0;

    [[nodiscard]] virtual VisionEncodeResult encodeStreaming(
        VisionEncodeQueue& queue, int total_hint, VisionProgressCallback progress) = 0;

    [[nodiscard]] virtual LlmGenerateResult generate(const std::string& prompt,
                                                     const VisionEncodeResult& vision,
                                                     int max_new_tokens, float temperature,
                                                     bool enable_thinking,
                                                     std::string_view lang) = 0;

    [[nodiscard]] virtual EncodeThenGenerateResult encodeThenGenerate(
        const std::vector<PendingVisionFrame>& frames, const EncodeThenGenerateParams& params,
        VisionProgressCallback progress = nullptr) = 0;

    virtual void clearVision() = 0;
    virtual void releaseModels() = 0;
};

[[nodiscard]] std::unique_ptr<StageTransport> makeStageTransport(
    const ModelRegistry& registry, const PipelineConfig& config, bool distributed);

}  // namespace vlm
