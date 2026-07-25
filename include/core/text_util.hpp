#pragma once

#include <string>
#include <vector>

namespace vlm {

[[nodiscard]] std::string stripThinkingTags(std::string text);

/** Extract inner text of each thinking block (redacted_thinking or thinking tags). */
[[nodiscard]] std::vector<std::string> extractThinkingBlocks(const std::string& text);

}  // namespace vlm
