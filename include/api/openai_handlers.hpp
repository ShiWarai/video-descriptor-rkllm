#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "types.hpp"

namespace vlm {

struct ParsedChatRequest {
    std::string video_url;
    int max_frames = 8;
    int frame_budget = 0;
    int max_tokens = 0;
    std::string lang = "ru";
    std::string prompt_mode = "detailed";
    std::optional<bool> enable_thinking;
    std::optional<float> temperature;
    std::optional<std::string> transcript;
    std::string model;
};

[[nodiscard]] nlohmann::json analyzeResultToJson(const AnalyzeResult& result);
[[nodiscard]] nlohmann::json buildChatCompletion(const std::string& model,
                                                 const AnalyzeResult& result);
[[nodiscard]] ParsedChatRequest parseChatCompletionRequest(const nlohmann::json& body,
                                                         const PipelineConfig& defaults);

[[nodiscard]] std::string saveVideoFromUrl(const std::string& video_url,
                                           const std::filesystem::path& workdir);
[[nodiscard]] std::filesystem::path saveUploadedFile(const std::string& filename,
                                                     const std::string& content,
                                                     const std::filesystem::path& workdir);

/** Remove all files in workdir (creates dir if missing). */
void clearWorkdir(const std::filesystem::path& workdir);
/** Delete file only if it lives under workdir (uploads/temp copies). */
void removeWorkFileIfOwned(const std::filesystem::path& file,
                           const std::filesystem::path& workdir);

}  // namespace vlm
