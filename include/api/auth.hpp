#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace vlm {

/** True for OpenAI-style routes that require a key when configured. */
[[nodiscard]] inline bool authRequiredForPath(std::string_view path) noexcept
{
    return path.rfind("/v1/", 0) == 0;
}

/** Parse `Authorization: Bearer <token>` (case-sensitive prefix per OpenAI). */
[[nodiscard]] inline std::optional<std::string> parseBearerToken(std::string_view authorization)
{
    constexpr std::string_view kPrefix = "Bearer ";
    if (!authorization.starts_with(kPrefix)) {
        return std::nullopt;
    }
    const std::string_view token = authorization.substr(kPrefix.size());
    if (token.empty()) {
        return std::nullopt;
    }
    return std::string(token);
}

[[nodiscard]] inline bool secureEqual(std::string_view a, std::string_view b) noexcept
{
    if (a.size() != b.size()) {
        return false;
    }
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= static_cast<std::uint8_t>(a[i] ^ b[i]);
    }
    return diff == 0;
}

/** Empty expected_key disables auth (always valid). */
[[nodiscard]] inline bool isApiKeyValid(const std::optional<std::string>& token,
                                        std::string_view expected_key) noexcept
{
    if (expected_key.empty()) {
        return true;
    }
    if (!token) {
        return false;
    }
    return secureEqual(*token, expected_key);
}

[[nodiscard]] inline std::string unauthorizedErrorJson()
{
    return R"({"error":{"message":"Incorrect API key provided: missing or invalid Bearer token","type":"invalid_request_error","param":null,"code":"invalid_api_key"}})";
}

}  // namespace vlm
