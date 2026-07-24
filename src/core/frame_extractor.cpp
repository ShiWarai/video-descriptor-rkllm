#include "core/frame_extractor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>

namespace vlm {

int planFrameCount(double duration_sec, double fps, int requested_frames)
{
    requested_frames = std::max(1, requested_frames);
    if (duration_sec <= 0.0 || fps <= 0.0) {
        return 0;
    }
    const int total_frames =
        std::max(1, static_cast<int>(std::lround(duration_sec * fps)));
    return std::min(requested_frames, total_frames);
}

FrameExtractor::FrameExtractor(std::string ffmpeg_bin_path)
    : ffmpeg_bin_path_(std::move(ffmpeg_bin_path))
{
}

std::string FrameExtractor::ffmpegPath(std::string_view name) const
{
    return ffmpeg_bin_path_ + "/" + std::string(name);
}

std::string FrameExtractor::runCommand(const std::string& cmd)
{
    std::array<char, 4096> buffer{};
    std::string result;
    const std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        return result;
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

bool FrameExtractor::probe(std::string_view filename, VideoInfo& info) const
{
    const std::string path(filename);
    const std::string cmd =
        ffmpegPath("ffprobe") + " -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate:format=duration "
        "-of default=noprint_wrappers=1:nokey=1 "
        "\"" + path + "\" 2>/dev/null";

    const std::string output = runCommand(cmd);
    if (output.empty()) {
        return false;
    }

    std::istringstream iss(output);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    if (lines.size() < 4) {
        return false;
    }

    try {
        info.width = std::stoi(lines[0]);
        info.height = std::stoi(lines[1]);
        const auto slash = lines[2].find('/');
        if (slash == std::string::npos) {
            info.fps = std::stod(lines[2]);
        } else {
            const double num = std::stod(lines[2].substr(0, slash));
            const double den = std::stod(lines[2].substr(slash + 1));
            info.fps = den > 0 ? num / den : 0.0;
        }
        info.duration_sec = std::stod(lines[3]);
    } catch (const std::exception&) {
        return false;
    }

    return info.width > 0 && info.height > 0 && info.fps > 0 && info.duration_sec > 0;
}

bool FrameExtractor::extractFrameAtTime(std::string_view filename, double time_sec, int width,
                                        int height, cv::Mat& out_bgr) const
{
    const std::string path(filename);
    const std::string cmd =
        ffmpegPath("ffmpeg") + " -hide_banner -loglevel error "
        "-ss " + std::to_string(time_sec) + " "
        "-hwaccel rkmpp -hwaccel_output_format drm_prime "
        "-i \"" + path + "\" "
        "-an -sn -frames:v 1 "
        "-vf scale_rkrga=w=" + std::to_string(width) + ":h=" + std::to_string(height) +
        ":format=bgr24:force_original_aspect_ratio=disable,hwdownload,format=bgr24 "
        "-f rawvideo -pix_fmt bgr24 pipe:1 2>/dev/null";

    const std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        return false;
    }

    const auto frame_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
    std::vector<std::uint8_t> buffer(frame_bytes);
    std::size_t read_total = 0;
    while (read_total < frame_bytes) {
        const auto n = fread(buffer.data() + read_total, 1, frame_bytes - read_total, pipe.get());
        if (n == 0) {
            break;
        }
        read_total += n;
    }

    if (read_total < frame_bytes) {
        return false;
    }

    out_bgr = cv::Mat(height, width, CV_8UC3);
    std::memcpy(out_bgr.data, buffer.data(), frame_bytes);
    return true;
}

std::vector<cv::Mat> FrameExtractor::extractFrames(std::string_view filename, int requested_frames,
                                                   FrameProgressCallback progress,
                                                   VideoInfo* out_info) const
{
    std::vector<cv::Mat> frames;
    const int got = extractFramesStreaming(
        filename, requested_frames,
        [&](cv::Mat frame, int /*index*/, int /*total*/) { frames.push_back(std::move(frame)); },
        progress, out_info);
    (void)got;
    return frames;
}

int FrameExtractor::extractFramesStreaming(std::string_view filename, int requested_frames,
                                           FrameReadyCallback on_frame, FrameProgressCallback progress,
                                           VideoInfo* out_info) const
{
    if (!on_frame || !std::filesystem::exists(filename)) {
        return 0;
    }

    VideoInfo info;
    if (!probe(filename, info)) {
        return 0;
    }
    if (out_info) {
        *out_info = info;
    }

    const int target_count = planFrameCount(info.duration_sec, info.fps, requested_frames);
    if (target_count <= 0) {
        return 0;
    }

    const double total_frames = info.duration_sec * info.fps;
    const double step = total_frames / static_cast<double>(target_count);
    int extracted = 0;

    for (int i = 0; i < target_count; ++i) {
        if (progress) {
            progress(i, target_count);
        }
        const int frame_idx = static_cast<int>(i * step);
        const double time_sec = frame_idx / info.fps;
        cv::Mat frame;
        if (!extractFrameAtTime(filename, time_sec, info.width, info.height, frame)) {
            continue;
        }
        on_frame(std::move(frame), i, target_count);
        ++extracted;
    }

    if (progress) {
        progress(target_count, target_count);
    }
    return extracted;
}

}  // namespace vlm
