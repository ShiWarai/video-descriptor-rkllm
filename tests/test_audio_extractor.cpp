#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/audio_extractor.hpp"

#ifndef TEST_FIXTURES_DIR
#define TEST_FIXTURES_DIR "tests/fixtures"
#endif

namespace {

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

void expectWavHeader(const std::vector<uint8_t>& wav)
{
    expect(wav.size() >= 44, "wav at least 44 bytes");
    expect(wav[0] == 'R' && wav[1] == 'I' && wav[2] == 'F' && wav[3] == 'F', "RIFF");
    expect(wav[8] == 'W' && wav[9] == 'A' && wav[10] == 'V' && wav[11] == 'E', "WAVE");
    expect(wav[12] == 'f' && wav[13] == 'm' && wav[14] == 't' && wav[15] == ' ', "fmt");

    const uint16_t audio_format = static_cast<uint16_t>(wav[20]) |
                                (static_cast<uint16_t>(wav[21]) << 8);
    const uint16_t channels = static_cast<uint16_t>(wav[22]) |
                              (static_cast<uint16_t>(wav[23]) << 8);
    const uint32_t sample_rate = static_cast<uint32_t>(wav[24]) |
                                 (static_cast<uint32_t>(wav[25]) << 8) |
                                 (static_cast<uint32_t>(wav[26]) << 16) |
                                 (static_cast<uint32_t>(wav[27]) << 24);
    const uint16_t bits_per_sample = static_cast<uint16_t>(wav[34]) |
                                   (static_cast<uint16_t>(wav[35]) << 8);

    expect(audio_format == 1, "PCM format");
    expect(channels == 1, "mono");
    expect(sample_rate == 16000, "16 kHz");
    expect(bits_per_sample == 16, "16-bit");
}

void test_extract_audio_fixture()
{
    const auto path = fixturePath("test_audio.m4a");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip audio extract: test_audio.m4a missing\n";
        return;
    }

    std::vector<uint8_t> wav;
    expect(vlm::extractAudioWav16kMono(path, wav), "extract m4a audio");
    expectWavHeader(wav);
    expect(wav.size() > 44, "non-empty PCM payload");
}

void test_no_audio_gif()
{
    const auto path = fixturePath("test.gif");
    if (!std::filesystem::exists(path)) {
        std::cout << "skip no-audio gif: fixture missing\n";
        return;
    }

    std::vector<uint8_t> wav;
    expect(!vlm::extractAudioWav16kMono(path, wav), "gif has no audio");
    expect(wav.empty(), "wav empty on failure");
}

void test_missing_file()
{
    std::vector<uint8_t> wav;
    expect(!vlm::extractAudioWav16kMono("nonexistent_audio_xyz.mp4", wav), "missing file");
    expect(wav.empty(), "wav empty on missing file");
}

}  // namespace

int main()
{
    test_missing_file();
    test_no_audio_gif();
    test_extract_audio_fixture();
    std::cout << "test_audio_extractor: ok\n";
    return 0;
}
