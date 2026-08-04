#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "core/frame_extractor.hpp"
#include "pipeline/audio_transcriber.hpp"
#include "pipeline/stage_transport.hpp"
#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace vlm {

class VideoContextPipeline {
public:
    VideoContextPipeline(ModelRegistry registry, PipelineConfig config,
                         std::unique_ptr<StageTransport> transport = nullptr);

    [[nodiscard]] bool initialize(std::optional<std::string_view> preload_model_id = std::nullopt);
    [[nodiscard]] bool isReady() const noexcept { return ready_; }
    [[nodiscard]] bool distributed() const noexcept { return transport_->distributed(); }

    [[nodiscard]] AnalyzeResult analyze(const AnalyzeRequest& request);
    [[nodiscard]] const ModelRegistry& registry() const noexcept { return registry_; }
    [[nodiscard]] const std::string& loadedModelId() const noexcept {
        return transport_->loadedModelId();
    }
    [[nodiscard]] const PipelineConfig& config() const noexcept { return config_; }

private:
    ModelRegistry registry_;
    PipelineConfig config_;
    std::unique_ptr<StageTransport> transport_;
    FrameExtractor extractor_;
    std::unique_ptr<AudioTranscriber> transcriber_;
    bool ready_ = false;

    [[nodiscard]] bool ensureModel(std::string_view model_id, std::string* error_out = nullptr);
    [[nodiscard]] int planVisionWorkers(std::string_view model_id,
                                        std::string* error_out = nullptr) const;
    void releaseModels();
    [[nodiscard]] int resolveFrameBudget(const AnalyzeRequest& request) const;
    [[nodiscard]] int effectiveMaxFrames(const AnalyzeRequest& request) const;
};

}  // namespace vlm
