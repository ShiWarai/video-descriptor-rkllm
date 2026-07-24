#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/opencv.hpp>

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
using FrameReadyCallback = std::function<void(cv::Mat frame, int index, int total)>;

class FrameExtractor {
public:
    explicit FrameExtractor(std::string ffmpeg_bin_path = "third_party/ffmpeg-rockchip/bin");

    [[nodiscard]] bool probe(std::string_view filename, VideoInfo& info) const;

    [[nodiscard]] bool extractFrameAtTime(std::string_view filename, double time_sec, int width,
                                          int height, cv::Mat& out_bgr) const;

    /** Sample up to `requested_frames` evenly across the video (or fewer if clip is shorter). */
    [[nodiscard]] std::vector<cv::Mat> extractFrames(std::string_view filename, int requested_frames,
                                                     FrameProgressCallback progress = nullptr,
                                                     VideoInfo* out_info = nullptr) const;

    /** Same as extractFrames but invokes `on_frame` as each frame is decoded. */
    [[nodiscard]] int extractFramesStreaming(std::string_view filename, int requested_frames,
                                             FrameReadyCallback on_frame,
                                             FrameProgressCallback progress = nullptr,
                                             VideoInfo* out_info = nullptr) const;

private:
    std::string ffmpeg_bin_path_;

    [[nodiscard]] std::string ffmpegPath(std::string_view name) const;
    [[nodiscard]] static std::string runCommand(const std::string& cmd);
};

}  // namespace vlm
