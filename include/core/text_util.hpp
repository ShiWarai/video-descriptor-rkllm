#pragma once

#include <string>
#include <vector>

namespace vlm {

[[nodiscard]] std::string stripThinkingTags(std::string text);

/** Extract inner text of each <think>...</think> block (empty if none). */
[[nodiscard]] std::vector<std::string> extractThinkingBlocks(const std::string& text);

}  // namespace vlm
