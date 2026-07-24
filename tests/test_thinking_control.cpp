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
    expect(p.find("Опиши кратко и по делу видео/gif") != std::string::npos, "simple ru text");
    expect(p.find("/no_think") != std::string::npos, "has /no_think");
    expect(p.find("## Кратко") == std::string::npos, "no detailed sections");
}

void test_prompt_detailed_ru()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("ru", false, "detailed");
    expect(p.find("/no_think") != std::string::npos, "has /no_think");
    expect(p.find(" /think") == std::string::npos, "no bare /think switch");
    expect(p.find("Опиши это видео") != std::string::npos, "has describe task");
    expect(p.find("русск") != std::string::npos, "russian instruction");
    expect(p.find("## Кратко") == std::string::npos, "no old summary section");
    expect(p.find("## О чём") != std::string::npos, "ru about section");
    expect(p.find("## Действия") != std::string::npos, "ru actions section");
    expect(p.find("## Жанр") != std::string::npos, "ru genre section");
    expect(p.find("## Для следующей модели") == std::string::npos, "no downstream section");
}

void test_prompt_with_transcript()
{
    const std::string ru =
        vlm::LlmRuntime::buildUserVisionPrompt("ru", false, "detailed", "привет");
    expect(ru.find("Речь на видео:") != std::string::npos, "ru speech label");
    expect(ru.find("Текст видео на английском") == std::string::npos, "no old eng label");
    expect(ru.find("привет") != std::string::npos, "transcript text");
    expect(ru.find("/no_think") != std::string::npos, "think switch after transcript");

    const std::string eng =
        vlm::LlmRuntime::buildUserVisionPrompt("eng", false, "simple", "hello");
    expect(eng.find("Speech in the video:") != std::string::npos, "eng speech label");
    expect(eng.find("hello") != std::string::npos, "eng transcript text");
}

void test_prompt_detailed_think_eng()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("eng", true, "detailed");
    expect(p.find("/no_think") == std::string::npos, "no /no_think when thinking on");
    expect(p.find(" /think") != std::string::npos, "has /think");
    expect(p.find("Describe this video") != std::string::npos, "has describe task");
    expect(p.find("English") != std::string::npos, "english instruction");
    expect(p.find("## Summary") != std::string::npos, "eng summary section");
    expect(p.find("## Actions") != std::string::npos, "eng actions section");
    expect(p.find("## Genre") != std::string::npos, "eng genre section");
}

void test_prompt_simple_eng()
{
    const std::string p = vlm::LlmRuntime::buildUserVisionPrompt("eng", true, "simple");
    expect(p.find("Give a brief, to-the-point description of the video/GIF") != std::string::npos,
           "simple eng text");
    expect(p.find(" /think") != std::string::npos, "has /think");
    expect(p.find("## Summary") == std::string::npos, "no detailed sections");
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
