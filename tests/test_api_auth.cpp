#include <cstdlib>
#include <iostream>

#include "api/auth.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_auth_required_paths()
{
    expect(vlm::authRequiredForPath("/v1/models"), "/v1/models");
    expect(vlm::authRequiredForPath("/v1/video/analyze"), "/v1/video/analyze");
    expect(!vlm::authRequiredForPath("/health"), "/health");
    expect(!vlm::authRequiredForPath("/ready"), "/ready");
}

void test_bearer_parse()
{
    const auto token = vlm::parseBearerToken("Bearer sk-test-key");
    expect(token.has_value(), "bearer parsed");
    expect(*token == "sk-test-key", "bearer value");
    expect(!vlm::parseBearerToken("Basic abc"), "not bearer");
    expect(!vlm::parseBearerToken("Bearer "), "empty bearer");
}

void test_api_key_validation()
{
    expect(vlm::isApiKeyValid(std::nullopt, ""), "disabled auth");
    expect(!vlm::isApiKeyValid(std::nullopt, "secret"), "missing token");
    expect(vlm::isApiKeyValid(std::string("secret"), "secret"), "valid token");
    expect(!vlm::isApiKeyValid(std::string("wrong"), "secret"), "invalid token");
}

}  // namespace

int main()
{
    test_auth_required_paths();
    test_bearer_parse();
    test_api_key_validation();
    std::cout << "test_api_auth: ok\n";
    return 0;
}
