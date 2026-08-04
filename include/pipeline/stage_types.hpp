#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/vision_encoder.hpp"
#include "types.hpp"

namespace vlm {

using VisionProgressCallback = std::function<void(int current, int total)>;

struct VisionEncodeResult {
    std::vector<float> embeddings;
    std::vector<double> frame_times;
    VisionModelInfo model_info{};
    int n_image = 0;
    double encode_ms = 0;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty() && n_image > 0; }
};

struct LlmGenerateResult {
    std::string text;
    int max_new_tokens = 0;
    int generate_tokens = 0;
    bool truncated = false;
    double generate_ms = 0;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

struct EncodeThenGenerateResult {
    std::string text;
    int max_new_tokens = 0;
    int generate_tokens = 0;
    bool truncated = false;
    int n_image = 0;
    double encode_ms = 0;
    double generate_ms = 0;
    VisionEncodeResult vision;  // populated in local mode for frame_times
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

struct EncodeThenGenerateParams {
    std::string job_id;
    std::string model_id;
    std::string prompt;
    int max_new_tokens = 0;
    float temperature = -1.0f;
    bool enable_thinking = false;
    std::string lang;
};

}  // namespace vlm
