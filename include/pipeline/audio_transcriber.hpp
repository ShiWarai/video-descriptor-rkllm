#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "types.hpp"

namespace vlm {

class AudioTranscriber {
public:
    virtual ~AudioTranscriber() = default;
    virtual TranscriptResult transcribe(const std::filesystem::path& video_path,
                                        const std::optional<std::string>& override_text) = 0;
};

class StubAudioTranscriber final : public AudioTranscriber {
public:
    TranscriptResult transcribe(const std::filesystem::path& video_path,
                                const std::optional<std::string>& override_text) override;
};

class HttpWhisperTranscriber final : public AudioTranscriber {
public:
    HttpWhisperTranscriber(std::string base_url, std::string ffmpeg_bin_path,
                           std::string workdir);

    TranscriptResult transcribe(const std::filesystem::path& video_path,
                                const std::optional<std::string>& override_text) override;

private:
    std::string base_url_;
    std::string ffmpeg_bin_path_;
    std::string workdir_;
};

[[nodiscard]] std::unique_ptr<AudioTranscriber> makeAudioTranscriber(const PipelineConfig& config);

}  // namespace vlm
