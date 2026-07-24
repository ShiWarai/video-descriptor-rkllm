#pragma once

#include <string_view>

namespace vlm::build {

#ifndef FFMPEG_BIN_PATH
#define FFMPEG_BIN_PATH "third_party/ffmpeg-rockchip/bin"
#endif

inline constexpr std::string_view kFfmpegBinPath{FFMPEG_BIN_PATH};

}  // namespace vlm::build
