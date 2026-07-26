#include <iostream>
#include <string>
#include <vector>

#include "core/vision_prompts.hpp"
#include "pipeline/audio_transcriber.hpp"
#include "types.hpp"

namespace {

using namespace vlm::prompts;

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_assign_segments_basic()
{
    const std::vector<double> times = {0.0, 1.5, 3.0};
    const std::vector<vlm::TranscriptSegment> segments = {
        {.start = 0.0, .end = 1.0, .text = "A"},
        {.start = 1.0, .end = 2.0, .text = "B"},
        {.start = 2.5, .end = 3.5, .text = "C"},
    };

    const auto speech = assignSpeechToFrameIntervals(times, segments, 4.0);
    expect(speech.size() == 3, "three intervals");
    expect(speech[0] == "A", "first interval");
    expect(speech[1] == "B", "second interval");
    expect(speech[2] == "C", "last interval after final frame");
}

void test_assign_segments_empty_interval()
{
    const std::vector<double> times = {0.0, 2.0};
    const std::vector<vlm::TranscriptSegment> segments = {
        {.start = 1.8, .end = 2.2, .text = "late"},
    };

    const auto speech = assignSpeechToFrameIntervals(times, segments, 3.0);
    expect(speech[0].empty(), "no speech before first midpoint");
    expect(speech[1] == "late", "speech after last frame");
}

void test_assign_segments_join_multiple()
{
    const std::vector<double> times = {0.0, 2.0};
    const std::vector<vlm::TranscriptSegment> segments = {
        {.start = 0.1, .end = 0.5, .text = "one"},
        {.start = 0.6, .end = 0.9, .text = "two"},
    };

    const auto speech = assignSpeechToFrameIntervals(times, segments, 2.0);
    expect(speech[0] == "one two", "joined speech in interval");
    expect(speech[1].empty(), "no speech after last frame");
}

void test_interleaved_prompt()
{
    const std::vector<double> times = {0.0, 1.5, 3.0};
    const std::vector<vlm::TranscriptSegment> segments = {
        {.start = 0.0, .end = 1.0, .text = "фраза A"},
        {.start = 1.0, .end = 2.0, .text = "фраза B"},
    };

    const std::string prompt =
        buildUserVisionPrompt("ru", "simple", times, segments, {}, 4.0);
    expect(prompt.find(kFramesInterleavedIntroRu) == 0, "interleaved intro");
    expect(prompt.find("<image> \"фраза A\"") != std::string::npos, "first frame + speech");
    expect(prompt.find("<image>\n") != std::string::npos, "frame without speech");
    expect(prompt.find("фраза B") != std::string::npos, "second interval speech");
    expect(prompt.find("[0.00s]") == std::string::npos, "no timestamps");
    expect(prompt.find("Речь:") == std::string::npos, "no speech label");
    expect(prompt.find(kSpeechPrefixRu) == std::string::npos, "no flat speech prefix");
    expect(prompt.find(kTaskSimpleRu) != std::string::npos, "task at end");
}

void test_fallback_flat_transcript()
{
    const std::string prompt = buildUserVisionPrompt("ru", "detailed", {}, {}, "плоский текст");
    expect(prompt.find(kFramesIntroRu) != std::string::npos, "flat intro");
    expect(prompt.find("плоский текст") != std::string::npos, "flat transcript");
    expect(prompt.find(kFramesInterleavedIntroRu) == std::string::npos, "no interleaved intro");
}

void test_expand_image_tags()
{
    const std::string single = "frames <image> task";
    expect(expandImageTags(single, 3) == "frames <image> <image> <image> task", "expand one tag");

    const std::string already = "<image>\n<image>\n";
    expect(expandImageTags(already, 2) == already, "leave pre-placed tags");

    expect(expandImageTags("no tags", 2) == "no tags", "unchanged without tags");
}

void test_parse_whisper_segments()
{
    const std::string body = R"({
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
            {"start": 2.0, "end": 2.5, "text": ""}
        ]
    })";

    vlm::TranscriptResult result;
    expect(vlm::parseWhisperTranscribeResponse(body, result), "parse ok");
    expect(result.text == "hello world", "full text");
    expect(result.segments.size() == 2, "skip empty segment text");
    expect(result.segments[0].text == "hello", "first segment");
    expect(result.segments[1].end == 2.0, "second segment end");
}

void test_parse_whisper_without_segments()
{
    const std::string body = R"({"text": "only text"})";
    vlm::TranscriptResult result{.segments = {{.start = 1, .end = 2, .text = "old"}}};
    expect(vlm::parseWhisperTranscribeResponse(body, result), "parse ok");
    expect(result.segments.empty(), "segments cleared when absent");
}

void test_prompt_without_transcript_on_error()
{
    const std::vector<double> times = {0.0, 1.5};
    const std::string prompt =
        buildUserVisionPrompt("ru", "detailed", times, {}, {}, 3.0);
    expect(prompt.find(kFramesIntroRu) != std::string::npos, "frames-only intro on error");
    expect(prompt.find(kFramesInterleavedIntroRu) == std::string::npos,
           "no interleaved intro without segments");
    expect(prompt.find("Речь:") == std::string::npos, "no speech block on error");
    expect(prompt.find(kSpeechPrefixRu) == std::string::npos, "no flat speech prefix on error");
}

void test_prompt_ok_text_without_segments_uses_flat()
{
    const std::vector<double> times = {0.0, 1.5};
    const std::string prompt =
        buildUserVisionPrompt("ru", "detailed", times, {}, "привет мир", 3.0);
    expect(prompt.find(kFramesIntroRu) != std::string::npos, "flat intro");
    expect(prompt.find("привет мир") != std::string::npos, "flat transcript used");
    expect(prompt.find(kFramesInterleavedIntroRu) == std::string::npos,
           "no interleaved without segments");
}

}  // namespace

int main()
{
    test_assign_segments_basic();
    test_assign_segments_empty_interval();
    test_assign_segments_join_multiple();
    test_interleaved_prompt();
    test_fallback_flat_transcript();
    test_prompt_without_transcript_on_error();
    test_prompt_ok_text_without_segments_uses_flat();
    test_expand_image_tags();
    test_parse_whisper_segments();
    test_parse_whisper_without_segments();
    std::cout << "test_asr_segments: ok\n";
    return 0;
}
