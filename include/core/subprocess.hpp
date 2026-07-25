#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace vlm {

// Execute script via /bin/bash without a shell wrapper; env values are not interpreted.
[[nodiscard]] bool runBashScript(
    const std::filesystem::path& script_path,
    const std::vector<std::pair<std::string, std::string>>& env_overrides);

}  // namespace vlm
