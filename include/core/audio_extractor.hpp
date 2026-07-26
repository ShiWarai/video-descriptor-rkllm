#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace vlm {

/** Extract audio as WAV PCM s16le mono 16 kHz into memory (no temp files). */
[[nodiscard]] bool extractAudioWav16kMono(const std::filesystem::path& media_path,
                                          std::vector<uint8_t>& out_wav);

}  // namespace vlm
