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
                                        const std::optional<std::string>& override_text,
                                        std::string_view language = {}) = 0;
};

class StubAudioTranscriber final : public AudioTranscriber {
public:
    TranscriptResult transcribe(const std::filesystem::path& video_path,
                                const std::optional<std::string>& override_text,
                                std::string_view language = {}) override;
};

class HttpWhisperTranscriber final : public AudioTranscriber {
public:
    HttpWhisperTranscriber(std::string base_url, std::string api_key = {});

    TranscriptResult transcribe(const std::filesystem::path& video_path,
                                const std::optional<std::string>& override_text,
                                std::string_view language = {}) override;

private:
    std::string base_url_;
    std::string api_key_;
};

[[nodiscard]] std::unique_ptr<AudioTranscriber> makeAudioTranscriber(const PipelineConfig& config);

/** Parse OpenAI verbose_json transcription response (text + optional segments + language). */
[[nodiscard]] bool parseWhisperTranscribeResponse(std::string_view body, TranscriptResult& result);

}  // namespace vlm
