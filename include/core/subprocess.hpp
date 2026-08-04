#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace vlm {

// Выполнение скрипта через /bin/bash без обёртки оболочки; значения окружения не интерпретируются.
[[nodiscard]] bool runBashScript(
    const std::filesystem::path& script_path,
    const std::vector<std::pair<std::string, std::string>>& env_overrides);

}  // namespace vlm
