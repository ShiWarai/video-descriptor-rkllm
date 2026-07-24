#pragma once

#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace vlm {

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string workdir = "/tmp/vlm_work";
    std::optional<std::string> preload_model_id;
    ModelRegistry registry;
    PipelineConfig pipeline;
};

[[nodiscard]] inline ServerConfig loadConfigFromJson(const nlohmann::json& j)
{
    ServerConfig cfg;

    if (j.contains("host")) {
        cfg.host = j["host"].get<std::string>();
    }
    if (j.contains("port")) {
        cfg.port = j["port"].get<int>();
    }
    if (j.contains("workdir")) {
        cfg.workdir = j["workdir"].get<std::string>();
        cfg.pipeline.workdir = cfg.workdir;
    }

    if (j.contains("default_model")) {
        cfg.registry.setDefaultModelId(j["default_model"].get<std::string>());
    } else if (j.contains("model_id")) {
        cfg.registry.setDefaultModelId(j["model_id"].get<std::string>());
    }
    if (j.contains("preload_model")) {
        cfg.preload_model_id = j["preload_model"].get<std::string>();
    }

    if (j.contains("models") && j["models"].is_array()) {
        for (const auto& entry : j["models"]) {
            ModelSpec spec;
            spec.id = entry.at("id").get<std::string>();
            spec.vision_model_path = entry.at("vision_model").get<std::string>();
            spec.llm_model_path = entry.at("llm_model").get<std::string>();
            if (entry.contains("temperature")) {
                spec.temperature = entry["temperature"].get<float>();
            }
            if (entry.contains("top_k")) {
                spec.top_k = entry["top_k"].get<int>();
            }
            if (entry.contains("top_p")) {
                spec.top_p = entry["top_p"].get<float>();
            }
            if (entry.contains("presence_penalty")) {
                spec.presence_penalty = entry["presence_penalty"].get<float>();
            }
            const std::string id = spec.id;
            cfg.registry.add(std::move(spec));
            if (!cfg.preload_model_id && entry.contains("preload") &&
                entry["preload"].get<bool>()) {
                cfg.preload_model_id = id;
            }
        }
    }

    const auto& p = j.contains("pipeline") ? j["pipeline"] : j;
    if (p.contains("default_frames")) {
        cfg.pipeline.default_frames = p["default_frames"].get<int>();
    }
    if (p.contains("default_context")) {
        cfg.pipeline.default_context = p["default_context"].get<int>();
    }
    if (p.contains("default_max_tokens")) {
        cfg.pipeline.default_max_tokens = p["default_max_tokens"].get<int>();
    }
    if (p.contains("frame_budget")) {
        cfg.pipeline.frame_budget = p["frame_budget"].get<int>();
    }
    if (p.contains("default_lang")) {
        cfg.pipeline.default_lang = p["default_lang"].get<std::string>();
    }
    if (p.contains("prompt_reserve_tokens")) {
        cfg.pipeline.prompt_reserve_tokens = p["prompt_reserve_tokens"].get<int>();
    }
    if (p.contains("absolute_max_frames")) {
        cfg.pipeline.absolute_max_frames = p["absolute_max_frames"].get<int>();
    }
    if (p.contains("temperature")) {
        cfg.pipeline.temperature = p["temperature"].get<float>();
    }
    if (p.contains("top_k")) {
        cfg.pipeline.top_k = p["top_k"].get<int>();
    }
    if (p.contains("top_p")) {
        cfg.pipeline.top_p = p["top_p"].get<float>();
    }
    if (p.contains("presence_penalty")) {
        cfg.pipeline.presence_penalty = p["presence_penalty"].get<float>();
    }
    if (p.contains("thinking_temperature")) {
        cfg.pipeline.thinking_temperature = p["thinking_temperature"].get<float>();
    }
    if (p.contains("thinking_top_p")) {
        cfg.pipeline.thinking_top_p = p["thinking_top_p"].get<float>();
    }
    if (p.contains("thinking_presence_penalty")) {
        cfg.pipeline.thinking_presence_penalty = p["thinking_presence_penalty"].get<float>();
    }
    if (p.contains("ffmpeg_bin_path")) {
        cfg.pipeline.ffmpeg_bin_path = p["ffmpeg_bin_path"].get<std::string>();
    }
    if (p.contains("workdir")) {
        cfg.pipeline.workdir = p["workdir"].get<std::string>();
    }
    if (p.contains("whisper_url")) {
        cfg.pipeline.whisper_url = p["whisper_url"].get<std::string>();
    }
    if (p.contains("enable_thinking")) {
        cfg.pipeline.enable_thinking = p["enable_thinking"].get<bool>();
    }
    if (p.contains("verbose")) {
        cfg.pipeline.verbose = p["verbose"].get<bool>();
    }

    // Legacy single-model config
    if (cfg.registry.models().empty() && p.contains("vision_model") && p.contains("llm_model")) {
        ModelSpec spec;
        spec.id = cfg.registry.defaultModelId().empty() ? "default" : cfg.registry.defaultModelId();
        spec.vision_model_path = p["vision_model"].get<std::string>();
        spec.llm_model_path = p["llm_model"].get<std::string>();
        const std::string id = spec.id;
        cfg.registry.add(std::move(spec));
        if (cfg.registry.defaultModelId().empty()) {
            cfg.registry.setDefaultModelId(id);
        }
    }

    if (cfg.registry.models().empty()) {
        throw std::runtime_error("No models configured (expected \"models\" array)");
    }
    if (cfg.registry.defaultModelId().empty()) {
        cfg.registry.setDefaultModelId(cfg.registry.models().front().id);
    }

    return cfg;
}

[[nodiscard]] inline ServerConfig loadConfigFile(const std::string& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open config: " + path);
    }
    nlohmann::json j;
    in >> j;
    return loadConfigFromJson(j);
}

}  // namespace vlm
