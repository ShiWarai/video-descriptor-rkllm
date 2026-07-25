#include <iostream>

#include "core/rgb_frame.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_empty()
{
    const vlm::RgbFrame frame;
    expect(frame.empty(), "default frame is empty");
    expect(frame.width == 0 && frame.height == 0, "default dimensions are zero");
}

void test_from_raw()
{
    std::vector<std::uint8_t> raw(448 * 448 * 3, 127);
    vlm::RgbFrame frame = vlm::RgbFrame::fromRaw(448, 448, std::move(raw));
    expect(!frame.empty(), "fromRaw is not empty");
    expect(frame.matches(448, 448), "matches expected size");
    expect(frame.byteSize() == 448u * 448u * 3u, "byteSize");
    expect(frame.data() == frame.pixels.data(), "data points to storage");
}

void test_matches_rejects_wrong_size()
{
    vlm::RgbFrame frame = vlm::RgbFrame::fromRaw(10, 10, std::vector<std::uint8_t>(300));
    expect(frame.matches(10, 10), "correct match");
    expect(!frame.matches(10, 11), "height mismatch");
    expect(!frame.matches(448, 448), "dimension mismatch");
}

}  // namespace

int main()
{
    test_empty();
    test_from_raw();
    test_matches_rejects_wrong_size();
    std::cout << "test_rgb_frame: ok\n";
    return 0;
}
