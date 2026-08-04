#pragma once

#include <string>
#include <vector>

namespace vlm {

/** Убрать блоки обдумывания из ответа модели. */
[[nodiscard]] std::string stripThinkingTags(std::string text);

/** Внутренний текст каждого блока обдумывания (теги think / thinking). */
[[nodiscard]] std::vector<std::string> extractThinkingBlocks(const std::string& text);

}  // namespace vlm
