#include "core/text_util.hpp"

#include <string_view>

namespace vlm {

namespace {

constexpr std::string_view kThinkOpen = "<think>";
constexpr std::string_view kThinkClose = "</think>";

}  // namespace

std::vector<std::string> extractThinkingBlocks(const std::string& text)
{
    std::vector<std::string> blocks;
    std::size_t pos = 0;
    for (;;) {
        const auto start = text.find(kThinkOpen, pos);
        if (start == std::string::npos) {
            break;
        }
        const auto content_start = start + kThinkOpen.size();
        const auto end = text.find(kThinkClose, content_start);
        if (end == std::string::npos) {
            blocks.push_back(text.substr(content_start));
            break;
        }
        blocks.push_back(text.substr(content_start, end - content_start));
        pos = end + kThinkClose.size();
    }
    return blocks;
}

std::string stripThinkingTags(std::string text)
{
    for (;;) {
        const auto start = text.find(kThinkOpen);
        if (start == std::string::npos) {
            break;
        }
        const auto end = text.find(kThinkClose, start);
        if (end == std::string::npos) {
            text.erase(start);
            break;
        }
        text.erase(start, end + kThinkClose.size() - start);
    }

    while (!text.empty() && (text.front() == ' ' || text.front() == '\n' || text.front() == '\r')) {
        text.erase(text.begin());
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

}  // namespace vlm
