#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>

namespace vlm {

struct TranscriptResult {
    std::string text;
    std::string status;  // "provided" | "stub"
    std::optional<std::string> language;
};

struct AnalyzeRequest {
    std::filesystem::path video_path;
    std::string model;  // empty → default from registry
    int max_frames = 8;
    int frame_budget = 0;  // 0 → use pipeline.frame_budget or formula
    int max_tokens = 0;    // 0 → RKLLM init default (pipeline.default_max_tokens)
    std::string lang = "ru";
    std::string prompt_mode = "detailed";  // "simple" | "detailed"
    std::optional<bool> enable_thinking;  // nullopt → pipeline.enable_thinking
    std::optional<float> temperature;     // nullopt → pipeline.temperature
    std::optional<std::string> transcript_override;
};

struct AnalyzeResult {
    std::string model;
    std::string description;
    TranscriptResult transcript;
    int frames_used = 0;
    int frames_requested = 0;
    int frame_budget = 0;
    bool frames_capped_by_context = false;
    double duration_sec = 0.0;
    // Stage timings in milliseconds, e.g. model_load_ms, llm_generate_ms, total_ms.
    std::map<std::string, double> metrics;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

struct PipelineConfig {
    int default_frames = 8;
    int default_context = 8192;
    int default_max_tokens = 1024;
    int frame_budget = 0;  // 0 → compute from context formula
    std::string default_lang = "ru";
    int prompt_reserve_tokens = 512;
    int absolute_max_frames = 0;
    // Qwen3.5-VL non-thinking (official card); Qengineering demo uses top_k=1 only.
    float temperature = 0.7f;
    int top_k = 20;
    float top_p = 0.8f;
    float presence_penalty = 1.5f;
    // Qwen3.5-VL thinking mode
    float thinking_temperature = 0.6f;
    float thinking_top_p = 0.95f;
    float thinking_presence_penalty = 0.0f;
    std::string ffmpeg_bin_path = "third_party/ffmpeg-rockchip/bin";
    bool enable_thinking = false;
    bool verbose = false;
};

}  // namespace vlm
