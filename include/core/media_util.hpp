#pragma once

#include <cctype>
#include <string_view>

namespace vlm {

[[nodiscard]] inline bool isGifPath(std::string_view path)
{
    const auto dot = path.find_last_of('.');
    if (dot == std::string_view::npos || dot + 4 != path.size()) {
        return false;
    }
    const char e1 = path[dot + 1];
    const char e2 = path[dot + 2];
    const char e3 = path[dot + 3];
    return (e1 == 'g' || e1 == 'G') && (e2 == 'i' || e2 == 'I') && (e3 == 'f' || e3 == 'F');
}

}  // namespace vlm
