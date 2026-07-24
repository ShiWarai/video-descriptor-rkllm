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
    registry.add({.id = "qwen3.5-0.8b-video",
                  .vision_model_path = "a.rknn",
                  .llm_model_path = "a.rkllm"});
    registry.add({.id = "qwen3.5-2b-video",
                  .vision_model_path = "b.rknn",
                  .llm_model_path = "b.rkllm"});
    registry.setDefaultModelId("qwen3.5-0.8b-video");

    const auto empty = registry.resolveId("");
    expect(empty.has_value(), "empty resolves to default");
    expect(*empty == "qwen3.5-0.8b-video", "default model id");

    const auto alias = registry.resolveId("0.8b");
    expect(alias.has_value() && *alias == "qwen3.5-0.8b-video", "0.8b alias");
}

}  // namespace

int main()
{
    test_resolve();
    std::cout << "test_model_registry: ok\n";
    return 0;
}
