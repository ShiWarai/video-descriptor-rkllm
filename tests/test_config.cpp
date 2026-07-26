#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#include "api/config.hpp"
#include "runtime/model_registry.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_registry_aliases_not_duplicated()
{
    vlm::ModelRegistry registry;
    registry.add({.id = "qwen3.5-0.8b-video",
                  .vision_model_path = "a.rknn",
                  .llm_model_path = "a.rkllm"});
    registry.add({.id = "qwen3.5-2b-video",
                  .vision_model_path = "b.rknn",
                  .llm_model_path = "b.rkllm"});
    registry.setDefaultModelId("qwen3.5-0.8b-video");

    expect(registry.models().size() == 2, "registry should list exactly 2 models");

    const auto resolved = registry.resolveId("2b");
    expect(resolved.has_value(), "alias 2b should resolve");
    expect(*resolved == "qwen3.5-2b-video", "alias 2b should map to 2b model id");

    bool has_alias_entry = false;
    for (const auto& model : registry.models()) {
        if (model.id == "2b") {
            has_alias_entry = true;
        }
    }
    expect(!has_alias_entry, "alias must not appear as separate model entry");
}

void test_config_frame_budget_and_models()
{
    const std::string json = R"({
      "default_model": "qwen3.5-0.8b-video",
      "models": [
        {"id": "qwen3.5-0.8b-video", "vision_model": "v.rknn", "llm_model": "l.rkllm",
         "temperature": 0.7, "top_k": 20, "top_p": 0.8, "presence_penalty": 1.5},
        {"id": "qwen3.5-2b-video", "vision_model": "v2.rknn", "llm_model": "l2.rkllm", "preload": true}
      ],
      "pipeline": {
        "frame_budget": 28,
        "default_context": 8192,
        "default_max_tokens": 2048,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "thinking_temperature": 0.6,
        "thinking_top_p": 0.95
      }
    })";

    nlohmann::json j = nlohmann::json::parse(json);
    const vlm::ServerConfig cfg = vlm::loadConfigFromJson(j);

    expect(cfg.registry.models().size() == 2, "config should load 2 models");
    expect(cfg.pipeline.frame_budget == 28, "frame_budget should be 28");
    expect(cfg.preload_model_id.has_value(), "preload from model entry");
    expect(*cfg.preload_model_id == "qwen3.5-2b-video", "preload model id");
    expect(cfg.pipeline.temperature == 0.7f, "pipeline temperature");
    expect(cfg.pipeline.top_k == 20, "pipeline top_k");
    expect(cfg.pipeline.top_p == 0.8f, "pipeline top_p");
    expect(cfg.pipeline.presence_penalty == 1.5f, "pipeline presence_penalty");
    expect(cfg.pipeline.thinking_temperature == 0.6f, "thinking temperature");
    const auto* m08 = cfg.registry.find("qwen3.5-0.8b-video");
    expect(m08 != nullptr, "0.8b model present");
    expect(m08->temperature.has_value() && *m08->temperature == 0.7f, "0.8b temperature override");
    expect(m08->top_k.has_value() && *m08->top_k == 20, "0.8b top_k override");
}

void test_config_whisper_url_and_workdir()
{
    const std::string json = R"({
      "workdir": "/data/vlm",
      "models": [
        {"id": "m1", "vision_model": "v.rknn", "llm_model": "l.rkllm"}
      ],
      "pipeline": {
        "whisper_url": "http://whisper-rknn.whisper-rknn.svc.cluster.local:8080",
        "whisper_api_key": "secret-whisper"
      }
    })";

    nlohmann::json j = nlohmann::json::parse(json);
    const vlm::ServerConfig cfg = vlm::loadConfigFromJson(j);

    expect(cfg.workdir == "/data/vlm", "server workdir");
    expect(cfg.pipeline.workdir == "/data/vlm", "pipeline workdir from server");
    expect(cfg.pipeline.whisper_url ==
               "http://whisper-rknn.whisper-rknn.svc.cluster.local:8080",
           "pipeline whisper_url");
    expect(cfg.pipeline.whisper_api_key == "secret-whisper", "pipeline whisper_api_key");
}

}  // namespace

int main()
{
    test_registry_aliases_not_duplicated();
    test_config_frame_budget_and_models();
    test_config_whisper_url_and_workdir();
    std::cout << "test_config: ok\n";
    return 0;
}
