#include <iostream>
#include <string>

#include "core/llm_runtime.hpp"
#include "core/text_util.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_prompt_simple_ru()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("ru", false, "simple");
    expect(p.find("<image>") != std::string::npos, "has image tag");
    expect(p.find("Тебе даны кадры из видео") != std::string::npos, "ru frames context");
    expect(p.find("Опиши кратко и по делу видео.") != std::string::npos, "simple ru task");
    expect(p.find("/no_think") != std::string::npos, "has /no_think");
    expect(p.find("1) О чём") == std::string::npos, "no detailed sections");
}

void test_prompt_detailed_ru()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("ru", false, "detailed");
    expect(p.find("\n/no_think") != std::string::npos, "has /no_think");
    expect(p.find("\n/think") == std::string::npos, "no bare /think switch");
    expect(p.find("Тебе даны кадры из видео <image>") != std::string::npos, "frames first");
    expect(p.find("Опиши это видео.") != std::string::npos, "has describe task");
    expect(p.find("русск") != std::string::npos, "russian instruction");
    expect(p.find("1) О чём видео?") != std::string::npos, "ru about item");
    expect(p.find("2) Текст в видео") != std::string::npos, "ru on-screen text");
    expect(p.find("Только надписи в кадрах видео, не речь.") != std::string::npos,
           "ru on-screen not speech");
    expect(p.find("3) Предположительный жанр") != std::string::npos, "ru genre");
    expect(p.find("## Действия") == std::string::npos, "no old actions section");
}

void test_prompt_with_transcript()
{
    const std::string ru =
        vlm::LlmRuntime::buildUserVisionPrompt("ru", false, "detailed", "привет");
    expect(ru.find("с речью в видео: \"привет\"") != std::string::npos, "ru speech before task");
    expect(ru.find("Текст видео на английском") == std::string::npos, "no old eng label");
    const auto speech_pos = ru.find("с речью в видео:");
    const auto task_pos = ru.find("Опиши это видео.");
    expect(speech_pos != std::string::npos && task_pos != std::string::npos && speech_pos < task_pos,
           "speech before describe task");
    expect(ru.find("/no_think") != std::string::npos, "think switch after task");

    const std::string eng =
        vlm::LlmRuntime::buildUserVisionPrompt("eng", false, "simple", "hello");
    expect(eng.find("with speech in the video: \"hello\"") != std::string::npos, "eng speech");
    const auto eng_speech = eng.find("with speech in the video:");
    const auto eng_task = eng.find("Describe the video briefly");
    expect(eng_speech != std::string::npos && eng_task != std::string::npos &&
               eng_speech < eng_task,
           "eng speech before task");
}

void test_prompt_detailed_think_eng()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("eng", true, "detailed");
    expect(p.find("/no_think") == std::string::npos, "no /no_think when thinking on");
    expect(p.find("\n/think") != std::string::npos, "has /think");
    expect(p.find("You are given frames from a video <image>") != std::string::npos,
           "eng frames context");
    expect(p.find("Describe this video.") != std::string::npos, "has describe task");
    expect(p.find("Answer in English:") != std::string::npos, "english instruction");
    expect(p.find("1) What is the video about?") != std::string::npos, "eng about item");
    expect(p.find("2) On-screen text") != std::string::npos, "eng on-screen text");
    expect(p.find("Only captions/labels visible in the video frames, not speech.") !=
               std::string::npos,
           "eng on-screen not speech");
    expect(p.find("3) Likely genre") != std::string::npos, "eng genre");
}

void test_prompt_simple_eng()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("eng", true, "simple");
    expect(p.find("Describe the video briefly and to the point.") != std::string::npos,
           "simple eng text");
    expect(p.find("\n/think") != std::string::npos, "has /think");
    expect(p.find("1) What is the video about?") == std::string::npos, "no detailed sections");
}

void test_strip_empty_think()
{
    const std::string raw = "<think>\n\n</think>\n\nHello";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out.find("<think>") == std::string::npos, "stripped open");
    expect(out.find("</think>") == std::string::npos, "stripped close");
    expect(out == "Hello", "text remains");
}

void test_extract_thinking()
{
    const std::string raw = "<think>\nstep one\n</think>\nAnswer.";
    const auto blocks = vlm::extractThinkingBlocks(raw);
    expect(blocks.size() == 1, "one think block");
    expect(blocks[0].find("step one") != std::string::npos, "think content");
    expect(vlm::stripThinkingTags(raw) == "Answer.", "strip leaves answer");
}

}  // namespace

int main()
{
    test_prompt_simple_ru();
    test_prompt_detailed_ru();
    test_prompt_with_transcript();
    test_prompt_detailed_think_eng();
    test_prompt_simple_eng();
    test_strip_empty_think();
    test_extract_thinking();
    std::cout << "test_thinking_control: ok\n";
    return 0;
}
