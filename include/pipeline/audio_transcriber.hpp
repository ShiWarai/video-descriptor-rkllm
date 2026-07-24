#pragma once

#include <filesystem>
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

}  // namespace vlm
