#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

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
    explicit HttpWhisperTranscriber(std::string base_url);

    TranscriptResult transcribe(const std::filesystem::path& video_path,
                                const std::optional<std::string>& override_text) override;

private:
    std::string base_url_;
};

[[nodiscard]] std::unique_ptr<AudioTranscriber> makeAudioTranscriber(const PipelineConfig& config);

/** Parse whisper-rknn /transcribe JSON (text + optional segments). */
[[nodiscard]] bool parseWhisperTranscribeResponse(std::string_view body, TranscriptResult& result);

}  // namespace vlm
