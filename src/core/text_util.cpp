#include "core/text_util.hpp"

#include <array>
#include <string_view>

namespace vlm {

namespace {

constexpr std::array<std::pair<std::string_view, std::string_view>, 2> kThinkTagPairs = {{
    {"<think>", "</think>"},
    {"\x3cthinking\x3e", "\x3c/thinking\x3e"},
}};

void stripThinkingBlocks(std::string& text)
{
    for (const auto& [open, close] : kThinkTagPairs) {
        for (;;) {
            const auto start = text.find(open);
            if (start == std::string::npos) {
                break;
            }
            const auto end = text.find(close, start + open.size());
            if (end == std::string::npos) {
                // Unclosed block: drop from the opening tag to the end.
                text.erase(start);
                break;
            }
            text.erase(start, end + close.size() - start);
        }
    }
}

void stripSoftSwitches(std::string& text)
{
    const auto erase_token = [&](std::string_view token) {
        for (;;) {
            const auto pos = text.find(token);
            if (pos == std::string::npos) {
                break;
            }
            text.erase(pos, token.size());
        }
    };
    erase_token("/no_think");
    erase_token("/think");
}

void collapseWhitespace(std::string& text)
{
    std::string collapsed;
    collapsed.reserve(text.size());
    bool prev_space = false;
    for (const char ch : text) {
        const bool is_space = (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r');
        if (is_space) {
            if (!prev_space && !collapsed.empty()) {
                collapsed.push_back(' ');
            }
            prev_space = true;
        } else {
            collapsed.push_back(ch);
            prev_space = false;
        }
    }
    text = std::move(collapsed);

    while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\r')) {
        text.erase(text.begin());
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
}

}  // namespace

std::vector<std::string> extractThinkingBlocks(const std::string& text)
{
    std::vector<std::string> blocks;
    for (const auto& [open, close] : kThinkTagPairs) {
        std::size_t pos = 0;
        for (;;) {
            const auto start = text.find(open, pos);
            if (start == std::string::npos) {
                break;
            }
            const auto content_start = start + open.size();
            const auto end = text.find(close, content_start);
            if (end == std::string::npos) {
                blocks.push_back(text.substr(content_start));
                break;
            }
            blocks.push_back(text.substr(content_start, end - content_start));
            pos = end + close.size();
        }
    }
    return blocks;
}

std::string stripThinkingTags(std::string text)
{
    // Remove entire thinking blocks (tags + inner reasoning); keep only the answer text.
    stripThinkingBlocks(text);
    // Qwen soft-switches echoed by the model.
    stripSoftSwitches(text);
    collapseWhitespace(text);
    return text;
}

}  // namespace vlm
