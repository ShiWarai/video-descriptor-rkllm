#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <rkllm/rknn_api.h>

#include "core/rgb_frame.hpp"

namespace vlm {

using VisionProgressCallback = std::function<void(int current, int total)>;

struct VisionModelInfo {
    int width = 0;
    int height = 0;
    int channels = 0;
    int image_tokens = 0;
    int embed_size = 0;
};

class VisionEncoder {
public:
    VisionEncoder();
    ~VisionEncoder();

    VisionEncoder(const VisionEncoder&) = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;
    VisionEncoder(VisionEncoder&&) = delete;
    VisionEncoder& operator=(VisionEncoder&&) = delete;

    [[nodiscard]] bool load(const std::string& model_path, bool verbose = false);
    void unload();
    [[nodiscard]] bool loaded() const noexcept { return ctx_ != 0; }

    [[nodiscard]] const VisionModelInfo& modelInfo() const noexcept { return info_; }

    [[nodiscard]] int computeFrameBudget(int context_len, int max_new_tokens,
                                         int prompt_reserve) const;

    [[nodiscard]] bool encodeFrames(const std::vector<RgbFrame>& frames,
                                    VisionProgressCallback progress = nullptr);

    /** Append one model-sized RGB frame (caller must clear() first). */
    [[nodiscard]] bool appendFrame(const RgbFrame& frame);

    void clear();
    [[nodiscard]] std::size_t frameCount() const noexcept { return frame_count_; }
    [[nodiscard]] const std::vector<float>& embeddings() const noexcept { return embeddings_; }

private:
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    VisionModelInfo info_;
    std::vector<float> embeddings_;
    std::size_t frame_count_ = 0;
    bool verbose_ = false;

    [[nodiscard]] bool initFromPath(std::string_view model_path);
    [[nodiscard]] int processOneImage(const RgbFrame& rgb);
    void dumpTensorAttr(const rknn_tensor_attr& attr) const;
};

}  // namespace vlm
