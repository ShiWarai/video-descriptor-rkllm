#include <iostream>

#include "runtime/model_registry.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_resolve()
{
    vlm::ModelRegistry registry;
    registry.add({.id = "qwen3.5-0.8b",
                  .vision_model_path = "a.rknn",
                  .llm_model_path = "a.rkllm"});
    registry.add({.id = "qwen3.5-2b",
                  .vision_model_path = "b.rknn",
                  .llm_model_path = "b.rkllm"});
    registry.add({.id = "qwen3.5-4b",
                  .vision_model_path = "c.rknn",
                  .llm_model_path = "c.rkllm"});
    registry.setDefaultModelId("qwen3.5-0.8b");

    const auto empty = registry.resolveId("");
    expect(empty.has_value(), "empty resolves to default");
    expect(*empty == "qwen3.5-0.8b", "default model id");

    const auto alias = registry.resolveId("0.8b");
    expect(alias.has_value() && *alias == "qwen3.5-0.8b", "0.8b alias");

    const auto alias4b = registry.resolveId("4b");
    expect(alias4b.has_value() && *alias4b == "qwen3.5-4b", "4b alias");
}

}  // namespace

int main()
{
    test_resolve();
    std::cout << "test_model_registry: ok\n";
    return 0;
}
