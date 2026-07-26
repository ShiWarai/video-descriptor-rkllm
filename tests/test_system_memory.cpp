#include <filesystem>
#include <iostream>
#include <string>

#include "core/system_memory.hpp"

namespace {

#ifndef TEST_MODELS_DIR
#define TEST_MODELS_DIR "models"
#endif

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_mem_available_readable()
{
    const auto kb = vlm::readMemAvailableKb();
    expect(kb.has_value() && *kb > 0, "MemAvailable readable on linux");
}

void test_estimate_4b_large()
{
    const auto llm = std::filesystem::path(TEST_MODELS_DIR) / "qwen3.5-4b_w8a8_rk3588.rkllm";
    const auto vision = std::filesystem::path(TEST_MODELS_DIR) / "qwen3.5-4b_vision_rk3588.rknn";
    if (!std::filesystem::exists(llm) || !std::filesystem::exists(vision)) {
        std::cout << "skip estimate 4b: model files missing\n";
        return;
    }
    const std::uint64_t est =
        vlm::estimateModelRamBytes(llm.string(), vision.string(), 8192, 3);
    expect(est > 6ull * 1024 * 1024 * 1024, "4b estimate > 6 GiB");
}

void test_has_enough_ram_refuses_tight()
{
    const auto avail_kb = vlm::readMemAvailableKb();
    if (!avail_kb) {
        std::cout << "skip ram refuse test: no meminfo\n";
        return;
    }
    std::string reason;
    const std::uint64_t avail_bytes = *avail_kb * 1024;
    const bool ok = vlm::hasEnoughRamForModel(avail_bytes + 1, &reason);
    expect(!ok, "requesting more than MemAvailable fails");
    expect(!reason.empty(), "reason string set");
}

}  // namespace

int main()
{
    test_mem_available_readable();
    test_estimate_4b_large();
    test_has_enough_ram_refuses_tight();
    std::cout << "test_system_memory: ok\n";
    return 0;
}
