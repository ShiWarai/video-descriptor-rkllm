#pragma once

#include <chrono>
#include <string>

#include "core/llm_runtime.hpp"
#include "core/vision_encoder.hpp"
#include "pipeline/stage_transport.hpp"
#include "runtime/model_registry.hpp"

namespace vlm {

class LocalStageTransport final : public StageTransport {
public:
    LocalStageTransport(const ModelRegistry& registry, PipelineConfig config);

    [[nodiscard]] bool distributed() const noexcept override;
    [[nodiscard]] bool loadModel(std::string_view model_id, std::string* error_out) override;
    [[nodiscard]] int planVisionWorkers(std::string_view model_id,
                                        std::string* error_out) const override;
    [[nodiscard]] bool isModelLoaded(std::string_view model_id) const override;
    [[nodiscard]] const std::string& loadedModelId() const override;
    [[nodiscard]] int resolveFrameBudget(const PipelineConfig& config,
                                         int frame_budget_override) const override;
    [[nodiscard]] VisionEncodeResult encodeStreaming(VisionEncodeQueue& queue, int total_hint,
                                                     VisionProgressCallback progress) override;
    [[nodiscard]] LlmGenerateResult generate(const std::string& prompt,
                                             const VisionEncodeResult& vision, int max_new_tokens,
                                             float temperature, bool enable_thinking,
                                             std::string_view lang) override;
    [[nodiscard]] EncodeThenGenerateResult encodeThenGenerate(
        const std::vector<PendingVisionFrame>& frames, const EncodeThenGenerateParams& params,
        VisionProgressCallback progress) override;
    void clearVision() override;
    void releaseModels() override;

private:
    const ModelRegistry& registry_;
    PipelineConfig config_;
    VisionEncoder vision_;
    LlmRuntime llm_;
    std::string loaded_model_id_;
};

}  // namespace vlm
