#include <iostream>
#include <string>

#include "core/llm_runtime.hpp"
#include "core/text_util.hpp"
#include "core/vision_prompts.hpp"

namespace {

using namespace vlm::prompts;

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

std::string expectedPrompt(std::string_view frames, std::string_view task,
                           std::string_view speech_prefix = {},
                           std::string_view transcript = {})
{
    std::string prompt;
    prompt += frames;
    if (!transcript.empty()) {
        prompt += speech_prefix;
        prompt += transcript;
        prompt += '"';
    }
    prompt += '\n';
    prompt += task;
    return prompt;
}

void test_prompt_simple_ru()
{
    const std::string p = buildUserVisionPrompt("ru", "simple");
    expect(p == expectedPrompt(kFramesIntroRu, kTaskSimpleRu), "simple ru prompt");
    expect(p.find("/no_think") == std::string::npos, "no /no_think in prompt");
    expect(p.find("/think") == std::string::npos, "no /think in prompt");
    expect(p.find(kTaskDetailedRu) == std::string::npos, "no detailed sections");
}

void test_prompt_detailed_ru()
{
    const std::string p = buildUserVisionPrompt("ru", "detailed");
    expect(p == expectedPrompt(kFramesIntroRu, kTaskDetailedRu), "detailed ru prompt");
    expect(p.find("/no_think") == std::string::npos, "no /no_think in prompt");
    expect(p.find("/think") == std::string::npos, "no /think in prompt");
}

void test_prompt_with_transcript()
{
    const std::string ru = buildUserVisionPrompt("ru", "detailed", "привет");
    expect(ru == expectedPrompt(kFramesIntroRu, kTaskDetailedRu, kSpeechPrefixRu, "привет"),
           "ru with transcript");
    const auto speech_pos = ru.find(kSpeechPrefixRu);
    const auto task_pos = ru.find(kTaskDetailedRu);
    expect(speech_pos != std::string::npos && task_pos != std::string::npos && speech_pos < task_pos,
           "speech before describe task");
    expect(ru.find("/no_think") == std::string::npos, "no think switch in prompt");

    const std::string eng = buildUserVisionPrompt("eng", "simple", "hello");
    expect(eng == expectedPrompt(kFramesIntroEng, kTaskSimpleEng, kSpeechPrefixEng, "hello"),
           "eng with transcript");
    const auto eng_speech = eng.find(kSpeechPrefixEng);
    const auto eng_task = eng.find(kTaskSimpleEng);
    expect(eng_speech != std::string::npos && eng_task != std::string::npos &&
               eng_speech < eng_task,
           "eng speech before task");
}

void test_prompt_detailed_eng()
{
    const std::string p = buildUserVisionPrompt("eng", "detailed");
    expect(p == expectedPrompt(kFramesIntroEng, kTaskDetailedEng), "detailed eng prompt");
    expect(p.find("/think") == std::string::npos, "no /think in prompt");
}

void test_prompt_simple_eng()
{
    const std::string p = buildUserVisionPrompt("eng", "simple");
    expect(p == expectedPrompt(kFramesIntroEng, kTaskSimpleEng), "simple eng prompt");
    expect(p.find("/think") == std::string::npos, "no /think in prompt");
    expect(p.find(kTaskDetailedEng) == std::string::npos, "no detailed sections");
}

void test_llm_runtime_delegates_to_prompts()
{
    expect(vlm::LlmRuntime::buildUserVisionPrompt("ru", "detailed") ==
               buildUserVisionPrompt("ru", "detailed"),
           "LlmRuntime ru detailed");
    expect(vlm::LlmRuntime::buildUserVisionPrompt("en", "simple") ==
               buildUserVisionPrompt("eng", "simple"),
           "LlmRuntime normalizes lang aliases");
    expect(vlm::LlmRuntime::buildUserVisionPrompt("eng", "detailed", "hi") ==
               buildUserVisionPrompt("eng", "detailed", "hi"),
           "LlmRuntime with transcript");
}

void test_strip_thinking_block()
{
    const std::string raw =
        "\x3cthinking\x3e\nlong reasoning here\n\x3c/thinking\x3e\n\nAnswer for user.";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out.find("long reasoning") == std::string::npos, "no reasoning in output");
    expect(out == "Answer for user.", "answer preserved after thinking block");
}

void test_strip_empty_think()
{
    const std::string raw = "<think>\n\n</think>\n\nHello";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out.find("<think>") == std::string::npos, "stripped open");
    expect(out.find("</think>") == std::string::npos, "stripped close");
    expect(out == "Hello", "text remains");
}

void test_strip_echoed_soft_switch()
{
    const std::string out = vlm::stripThinkingTags("Ответ про ролик.\n/no_think");
    expect(out == "Ответ про ролик.", "stripped trailing /no_think");
    const std::string out2 = vlm::stripThinkingTags("Answer /think");
    expect(out2 == "Answer", "stripped trailing /think");
    const std::string out3 = vlm::stripThinkingTags("Текст  /no_think  продолжение");
    expect(out3 == "Текст продолжение", "stripped inline /no_think with spaces");
    const std::string out4 = vlm::stripThinkingTags("start /think end");
    expect(out4 == "start end", "stripped inline /think");
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
    test_prompt_detailed_eng();
    test_prompt_simple_eng();
    test_llm_runtime_delegates_to_prompts();
    test_strip_thinking_block();
    test_strip_empty_think();
    test_strip_echoed_soft_switch();
    test_extract_thinking();
    std::cout << "test_thinking_control: ok\n";
    return 0;
}
