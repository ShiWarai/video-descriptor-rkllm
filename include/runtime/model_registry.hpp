#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vlm {

struct ModelSpec {
    std::string id;
    std::string vision_model_path;
    std::string llm_model_path;
    // Optional overrides (nullopt → pipeline defaults). From Qwen3.5-VL card / model pack.
    std::optional<float> temperature;
    std::optional<int> top_k;
    std::optional<float> top_p;
    std::optional<float> presence_penalty;
};

class ModelRegistry {
public:
    void setDefaultModelId(std::string id);
    void add(ModelSpec spec);

    [[nodiscard]] const std::string& defaultModelId() const noexcept { return default_model_id_; }
    [[nodiscard]] const std::vector<ModelSpec>& models() const noexcept { return models_; }

    [[nodiscard]] std::optional<std::string> resolveId(std::string_view requested) const;
    [[nodiscard]] const ModelSpec* find(std::string_view id) const;

private:
    [[nodiscard]] static std::string normalizeKey(std::string_view key);

    std::string default_model_id_;
    std::vector<ModelSpec> models_;
    std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace vlm
