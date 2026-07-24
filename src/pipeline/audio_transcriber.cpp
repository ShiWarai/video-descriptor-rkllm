#include "pipeline/audio_transcriber.hpp"

namespace vlm {

TranscriptResult StubAudioTranscriber::transcribe(const std::filesystem::path& /*video_path*/,
                                                  const std::optional<std::string>& override_text)
{
    if (override_text && !override_text->empty()) {
        return TranscriptResult{.text = *override_text, .status = "provided"};
    }
    return TranscriptResult{.text = "", .status = "stub"};
}

}  // namespace vlm
