#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
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

struct PendingVisionFrame {
    int index = 0;
    double time_sec = 0;
    RgbFrame frame;
};

/** Shared queue between frame extract and parallel vision workers. */
struct VisionEncodeQueue {
    std::mutex& mu;
    std::condition_variable& cv;
    std::deque<PendingVisionFrame>& pending;
    const bool& extract_done;
    const std::atomic<bool>& abort;
};

class VisionEncoder {
public:
    static constexpr int kWorkerCount = 3;

    VisionEncoder();
    ~VisionEncoder();

    VisionEncoder(const VisionEncoder&) = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;
    VisionEncoder(VisionEncoder&&) = delete;
    VisionEncoder& operator=(VisionEncoder&&) = delete;

    /** Load vision pack with worker_count contexts (1..kWorkerCount).
     *  Worker 0 does a full rknn_init; others use rknn_dup_context (shared weights).
     *  If fewer than 3, last worker uses NPU AUTO. */
    [[nodiscard]] bool load(const std::string& model_path, bool verbose = false,
                            int worker_count = kWorkerCount);
    void unload();
    [[nodiscard]] bool loaded() const noexcept;
    [[nodiscard]] int workerCount() const noexcept { return worker_count_; }

    [[nodiscard]] const VisionModelInfo& modelInfo() const noexcept { return info_; }

    [[nodiscard]] int computeFrameBudget(int context_len, int max_new_tokens,
                                         int prompt_reserve) const;

    [[nodiscard]] bool encodeFrames(const std::vector<RgbFrame>& frames,
                                    VisionProgressCallback progress = nullptr);

    /**
     * Parallel encode from a shared queue (one RKNN context per active worker / NPU core).
     * Embeddings are assembled in ascending frame index order; failed frames are skipped.
     */
    [[nodiscard]] int encodeStreaming(VisionEncodeQueue& queue, int total_hint,
                                      VisionProgressCallback progress = nullptr);

    /** Append one model-sized RGB frame (caller must clear() first). Uses worker 0. */
    [[nodiscard]] bool appendFrame(const RgbFrame& frame);

    void clear();
    [[nodiscard]] std::size_t frameCount() const noexcept { return frame_count_; }
    [[nodiscard]] const std::vector<float>& embeddings() const noexcept { return embeddings_; }
    /** Sample time (seconds) per encoded frame, same order as embeddings. */
    [[nodiscard]] const std::vector<double>& frameTimes() const noexcept { return frame_times_; }

private:
    struct WorkerSlot {
        rknn_context ctx = 0;
        rknn_core_mask core_mask = RKNN_NPU_CORE_0;
    };

    std::array<WorkerSlot, kWorkerCount> workers_{};
    int worker_count_ = 0;
    rknn_input_output_num io_num_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    VisionModelInfo info_;
    std::vector<float> embeddings_;
    std::vector<double> frame_times_;
    std::size_t frame_count_ = 0;
    bool verbose_ = false;

    [[nodiscard]] bool initFromPath(std::string_view model_path, int worker_count);
    [[nodiscard]] bool initWorker(WorkerSlot& worker, std::string_view model_path,
                                  rknn_core_mask core_mask);
    /** Duplicate primary context with shared weights; pin to core_mask. */
    [[nodiscard]] bool dupWorkerFromPrimary(WorkerSlot& worker, rknn_core_mask core_mask);
    void destroyWorkers();
    void queryModelInfoFromPrimary();
    [[nodiscard]] std::size_t floatsPerImage() const;
    [[nodiscard]] bool processOneImage(rknn_context ctx, const RgbFrame& rgb,
                                       std::vector<float>& out) const;
    void assembleEmbeddings(
        const std::vector<std::optional<std::pair<std::vector<float>, double>>>& slots);
    void dumpTensorAttr(const rknn_tensor_attr& attr) const;
};

}  // namespace vlm
