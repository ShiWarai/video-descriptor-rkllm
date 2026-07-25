#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "core/rgb_frame.hpp"

namespace vlm {

struct VideoInfo {
    double duration_sec = 0.0;
    double fps = 0.0;
    int width = 0;
    int height = 0;
};

/** How many frames to take: min(requested, frames available in the clip). 0 if invalid. */
[[nodiscard]] int planFrameCount(double duration_sec, double fps, int requested_frames);

using FrameProgressCallback = std::function<void(int current, int total)>;
using FrameReadyCallback = std::function<void(RgbFrame frame, int index, int total)>;

class FrameExtractor {
public:
    /** Default Qwen3.5-VL vision input size (letterboxed RGB). */
    static constexpr int kDefaultVisionSize = 448;

    FrameExtractor() = default;

    [[nodiscard]] bool probe(std::string_view filename, VideoInfo& info) const;

    /**
     * Decode one frame at time_sec into model-ready RGB (letterboxed to target_w x target_h).
     * Video: libav + h264_rkmpp decode, scale_rkrga resize/convert, software pad for letterbox.
     * GIF: software libav decode/scale/pad (rkmpp cannot read animated GIF).
     */
    [[nodiscard]] bool extractFrameAtTime(std::string_view filename, double time_sec,
                                          int target_w, int target_h, RgbFrame& out_rgb) const;

    /** Sample up to `requested_frames` evenly; each frame is model-ready RGB. */
    [[nodiscard]] std::vector<RgbFrame> extractFrames(
        std::string_view filename, int requested_frames,
        int target_w = kDefaultVisionSize, int target_h = kDefaultVisionSize,
        FrameProgressCallback progress = nullptr, VideoInfo* out_info = nullptr) const;

    /** Same as extractFrames but invokes `on_frame` as each frame is decoded. */
    [[nodiscard]] int extractFramesStreaming(
        std::string_view filename, int requested_frames, FrameReadyCallback on_frame,
        int target_w = kDefaultVisionSize, int target_h = kDefaultVisionSize,
        FrameProgressCallback progress = nullptr, VideoInfo* out_info = nullptr) const;
};

}  // namespace vlm
