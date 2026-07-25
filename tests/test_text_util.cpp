#include <cassert>
#include <iostream>
#include <string>

#include "core/text_util.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_strip_thinking()
{
    const std::string raw =
        "<think>\n\n</think>\n\nВ видео показан процесс.";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out.find("<think>") == std::string::npos, "thinking tag removed");
    expect(out.starts_with("В видео"), "description preserved after strip");
}

void test_strip_empty()
{
    const std::string raw = "<think>foo</think>";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out.empty(), "only thinking block yields empty string");
}

void test_preserve_multiline_structure()
{
    const std::string raw =
        "1) О чём видео?\n"
        "Краткая суть.\n"
        "\n"
        "2) Надписи в видео\n"
        "Текст на экране.";
    const std::string out = vlm::stripThinkingTags(raw);
    expect(out == raw, "multiline answer preserved");
    expect(out.find('\n') != std::string::npos, "newlines kept");
}

}  // namespace

int main()
{
    test_strip_thinking();
    test_strip_empty();
    test_preserve_multiline_structure();
    std::cout << "test_text_util: ok\n";
    return 0;
}
