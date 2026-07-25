#include <filesystem>
#include <iostream>
#include <string>

#include "core/frame_extractor.hpp"

#ifndef TEST_FIXTURES_DIR
#define TEST_FIXTURES_DIR "tests/fixtures"
#endif

namespace {

constexpr int kVision = vlm::FrameExtractor::kDefaultVisionSize;
constexpr std::size_t kRgbBytes =
    static_cast<std::size_t>(kVision) * static_cast<std::size_t>(kVision) * 3;

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

[[nodiscard]] std::filesystem::path fixturePath(std::string_view name)
{
    return std::filesystem::path(TEST_FIXTURES_DIR) / name;
}

[[nodiscard]] bool mppDecodeAvailable()
{
    return std::filesystem::exists("/dev/mpp_service");
}

void test_probe_mp4()
{
    const auto path = fixturePath("test.mp4");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip probe mp4: fixture missing\n";
        return;
    }

    vlm::FrameExtractor extractor;
    vlm::VideoInfo info;
    expect(extractor.probe(path.string(), info), "probe mp4");
    expect(info.width > 0 && info.height > 0, "positive dimensions");
    expect(info.fps > 0.0, "positive fps");
    expect(info.duration_sec > 0.0, "positive duration");
}

void test_probe_missing()
{
    vlm::FrameExtractor extractor;
    vlm::VideoInfo info;
    expect(!extractor.probe("nonexistent_file_xyz.mp4", info), "missing file");
}

void test_extract_mp4()
{
    const auto path = fixturePath("test.mp4");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip extract mp4: fixture missing\n";
        return;
    }
    if (!mppDecodeAvailable()) {
        std::cout << "skip extract mp4: no MPP device (CI / non-RK host)\n";
        return;
    }

    vlm::FrameExtractor extractor;
    int count = 0;
    const int got = extractor.extractFramesStreaming(
        path.string(), 4,
        [&](vlm::RgbFrame frame, int index, int total) {
            expect(total > 0, "positive total");
            expect(index >= 0 && index < total, "index in range");
            expect(frame.matches(kVision, kVision), "frame is 448x448 RGB");
            expect(frame.byteSize() == kRgbBytes, "RGB byte size");
            ++count;
        });
    expect(got > 0, "at least one mp4 frame");
    expect(count == got, "callback count matches return value");
}

void test_extract_gif()
{
    const auto path = fixturePath("test.gif");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip extract gif: fixture missing\n";
        return;
    }

    vlm::FrameExtractor extractor;
    vlm::VideoInfo info;
    const int got = extractor.extractFramesStreaming(
        path.string(), 2,
        [&](vlm::RgbFrame frame, int /*index*/, int /*total*/) {
            expect(frame.matches(kVision, kVision), "gif frame size");
        },
        kVision, kVision, nullptr, &info);
    expect(got > 0, "at least one gif frame");
    expect(info.duration_sec > 0.0, "gif duration");
}

void test_extract_frame_at_time()
{
    const auto path = fixturePath("test.mp4");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip single frame: fixture missing\n";
        return;
    }
    if (!mppDecodeAvailable()) {
        std::cout << "skip single frame: no MPP device (CI / non-RK host)\n";
        return;
    }

    vlm::FrameExtractor extractor;
    vlm::RgbFrame frame;
    expect(extractor.extractFrameAtTime(path.string(), 0.0, kVision, kVision, frame),
           "extract at t=0");
    expect(frame.matches(kVision, kVision), "single frame size");
}

}  // namespace

int main()
{
    test_probe_missing();
    test_probe_mp4();
    test_extract_mp4();
    test_extract_gif();
    test_extract_frame_at_time();
    std::cout << "test_frame_extractor: ok\n";
    return 0;
}
