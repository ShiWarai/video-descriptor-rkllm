#include <iostream>

#include "core/frame_extractor.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_plan_frame_count()
{
    // Requested 8, clip has ~21 frames → 8
    expect(vlm::planFrameCount(1.12, 100.0, 8) == 8, "short gif: take requested");
    // Requested 8, clip has 3 frames → 3
    expect(vlm::planFrameCount(0.1, 30.0, 8) == 3, "fewer than requested");
    // Requested 4, long clip → 4
    expect(vlm::planFrameCount(10.0, 30.0, 4) == 4, "long clip: take requested");
    // Invalid media
    expect(vlm::planFrameCount(0.0, 30.0, 8) == 0, "zero duration");
    expect(vlm::planFrameCount(1.0, 0.0, 8) == 0, "zero fps");
}

}  // namespace

int main()
{
    test_plan_frame_count();
    std::cout << "test_frame_plan: ok\n";
    return 0;
}
