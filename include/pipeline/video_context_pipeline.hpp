#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/frame_extractor.hpp"
#include "core/llm_runtime.hpp"
#include "core/vision_encoder.hpp"
#include "pipeline/audio_transcriber.hpp"
#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace vlm {

class VideoContextPipeline {
public:
    VideoContextPipeline(ModelRegistry registry, PipelineConfig config);

    [[nodiscard]] bool initialize(std::optional<std::string_view> preload_model_id = std::nullopt);
    [[nodiscard]] bool isReady() const noexcept { return ready_; }

    [[nodiscard]] AnalyzeResult analyze(const AnalyzeRequest& request);
    [[nodiscard]] const ModelRegistry& registry() const noexcept { return registry_; }
    [[nodiscard]] const std::string& loadedModelId() const noexcept { return loaded_model_id_; }
    [[nodiscard]] const PipelineConfig& config() const noexcept { return config_; }

private:
    ModelRegistry registry_;
    PipelineConfig config_;
    VisionEncoder vision_;
    LlmRuntime llm_;
    FrameExtractor extractor_;
    std::unique_ptr<AudioTranscriber> transcriber_;
    std::string loaded_model_id_;
    bool ready_ = false;

    [[nodiscard]] bool ensureModel(std::string_view model_id);
    void releaseModels();
    [[nodiscard]] int resolveFrameBudget(const AnalyzeRequest& request) const;
    [[nodiscard]] int effectiveMaxFrames(const AnalyzeRequest& request) const;
};

}  // namespace vlm
