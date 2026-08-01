#include "core/vision_encoder.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace vlm {

namespace {

constexpr std::array<rknn_core_mask, VisionEncoder::kWorkerCount> kWorkerCoreMasks = {
    RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2,
};

/** Pin workers 0..n-2 to dedicated cores; last worker gets AUTO to use leftover NPU cores. */
[[nodiscard]] rknn_core_mask coreMaskForWorker(int worker_index, int worker_count)
{
    if (worker_count >= VisionEncoder::kWorkerCount) {
        return kWorkerCoreMasks[static_cast<std::size_t>(worker_index)];
    }
    if (worker_index == worker_count - 1) {
        return RKNN_NPU_CORE_AUTO;
    }
    return kWorkerCoreMasks[static_cast<std::size_t>(worker_index)];
}

const char* coreMaskName(rknn_core_mask mask)
{
    switch (mask) {
    case RKNN_NPU_CORE_AUTO:
        return "AUTO";
    case RKNN_NPU_CORE_0:
        return "CORE_0";
    case RKNN_NPU_CORE_1:
        return "CORE_1";
    case RKNN_NPU_CORE_2:
        return "CORE_2";
    case RKNN_NPU_CORE_0_1:
        return "CORE_0_1";
    case RKNN_NPU_CORE_0_1_2:
        return "CORE_0_1_2";
    default:
        return "OTHER";
    }
}

}  // namespace

VisionEncoder::VisionEncoder() = default;

VisionEncoder::~VisionEncoder()
{
    unload();
}

bool VisionEncoder::loaded() const noexcept
{
    return workers_[0].ctx != 0;
}

void VisionEncoder::destroyWorkers()
{
    // Destroy duplicates before the base context (worker 0) so shared weights stay valid.
    for (int i = worker_count_ - 1; i >= 0; --i) {
        auto& worker = workers_[static_cast<std::size_t>(i)];
        if (worker.ctx != 0) {
            rknn_destroy(worker.ctx);
            worker.ctx = 0;
        }
    }
    worker_count_ = 0;
}

void VisionEncoder::unload()
{
    destroyWorkers();
    input_attrs_.clear();
    output_attrs_.clear();
    info_ = {};
    clear();
}

void VisionEncoder::dumpTensorAttr(const rknn_tensor_attr& attr) const
{
    if (!verbose_) {
        return;
    }
    std::cerr << "  index=" << attr.index << ", name=" << attr.name << ", n_dims=" << attr.n_dims
              << ", dims=[" << attr.dims[0] << ", " << attr.dims[1] << ", " << attr.dims[2]
              << ", " << attr.dims[3] << "]\n";
}

bool VisionEncoder::initWorker(WorkerSlot& worker, std::string_view model_path,
                               rknn_core_mask core_mask)
{
    const std::string path(model_path);
    worker.core_mask = core_mask;
    int ret = rknn_init(&worker.ctx, const_cast<char*>(path.c_str()), 0, 0, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_init failed for core mask " << worker.core_mask << ": " << ret << '\n';
        return false;
    }
    ret = rknn_set_core_mask(worker.ctx, worker.core_mask);
    if (ret < 0) {
        std::cerr << "rknn_set_core_mask failed: " << ret << '\n';
        rknn_destroy(worker.ctx);
        worker.ctx = 0;
        return false;
    }
    return true;
}

bool VisionEncoder::dupWorkerFromPrimary(WorkerSlot& worker, rknn_core_mask core_mask)
{
    if (workers_[0].ctx == 0) {
        return false;
    }
    worker.core_mask = core_mask;
    worker.ctx = 0;
    int ret = rknn_dup_context(&workers_[0].ctx, &worker.ctx);
    if (ret < 0 || worker.ctx == 0) {
        std::cerr << "rknn_dup_context failed for core mask " << core_mask << ": " << ret << '\n';
        worker.ctx = 0;
        return false;
    }
    ret = rknn_set_core_mask(worker.ctx, worker.core_mask);
    if (ret < 0) {
        std::cerr << "rknn_set_core_mask failed on dup: " << ret << '\n';
        rknn_destroy(worker.ctx);
        worker.ctx = 0;
        return false;
    }
    return true;
}

void VisionEncoder::queryModelInfoFromPrimary()
{
    const rknn_context primary = workers_[0].ctx;
    rknn_query(primary, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        input_attrs_[i] = {};
        input_attrs_[i].index = i;
        rknn_query(primary, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        dumpTensorAttr(input_attrs_[i]);
    }

    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i] = {};
        output_attrs_[i].index = i;
        rknn_query(primary, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        dumpTensorAttr(output_attrs_[i]);
    }

    for (int i = 0; i < 4; ++i) {
        if (output_attrs_[0].dims[i] > 1) {
            info_.image_tokens = output_attrs_[0].dims[i];
            info_.embed_size = output_attrs_[0].dims[i + 1];
            break;
        }
    }

    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
        info_.channels = input_attrs_[0].dims[1];
        info_.height = input_attrs_[0].dims[2];
        info_.width = input_attrs_[0].dims[3];
    } else {
        info_.height = input_attrs_[0].dims[1];
        info_.width = input_attrs_[0].dims[2];
        info_.channels = input_attrs_[0].dims[3];
    }
}

bool VisionEncoder::initFromPath(std::string_view model_path, int worker_count)
{
    const int n = std::clamp(worker_count, 1, kWorkerCount);
    // One full load of vision weights, then rknn_dup_context for the rest (shared weights,
    // per-worker runtime/IO) — same pattern as whisper-rknn encoder pool.
    if (!initWorker(workers_[0], model_path, coreMaskForWorker(0, n))) {
        destroyWorkers();
        return false;
    }
    for (int i = 1; i < n; ++i) {
        const rknn_core_mask mask = coreMaskForWorker(i, n);
        if (!dupWorkerFromPrimary(workers_[static_cast<std::size_t>(i)], mask)) {
            destroyWorkers();
            return false;
        }
    }
    worker_count_ = n;
    queryModelInfoFromPrimary();
    if (verbose_) {
        std::cerr << "vision: " << worker_count_ << " RKNN worker(s) (shared weights via dup):";
        for (int i = 0; i < worker_count_; ++i) {
            std::cerr << ' ' << coreMaskName(workers_[static_cast<std::size_t>(i)].core_mask);
        }
        if (worker_count_ < kWorkerCount) {
            std::cerr << " (last=AUTO to use remaining NPU cores)";
        }
        std::cerr << '\n';
    }
    return true;
}

bool VisionEncoder::load(const std::string& model_path, bool verbose, int worker_count)
{
    unload();
    verbose_ = verbose;
    return initFromPath(model_path, worker_count);
}

int VisionEncoder::computeFrameBudget(int context_len, int max_new_tokens, int prompt_reserve) const
{
    if (!loaded() || info_.image_tokens <= 0) {
        return 1;
    }
    const int available = context_len - max_new_tokens - prompt_reserve;
    if (available <= 0) {
        return 1;
    }
    return std::max(1, available / info_.image_tokens);
}

std::size_t VisionEncoder::floatsPerImage() const
{
    return static_cast<std::size_t>(info_.image_tokens) * static_cast<std::size_t>(info_.embed_size) *
           static_cast<std::size_t>(io_num_.n_output);
}

bool VisionEncoder::processOneImage(rknn_context ctx, const RgbFrame& rgb,
                                    std::vector<float>& out) const
{
    rknn_input inputs[1]{};
    std::vector<rknn_output> outputs(io_num_.n_output);

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = static_cast<uint32_t>(rgb.byteSize());
    inputs[0].buf = const_cast<void*>(static_cast<const void*>(rgb.data()));

    int ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        return false;
    }
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        return false;
    }

    for (auto& output : outputs) {
        output.want_float = 1;
    }
    ret = rknn_outputs_get(ctx, io_num_.n_output, outputs.data(), nullptr);
    if (ret < 0) {
        return false;
    }

    out.resize(floatsPerImage());
    float* dest_ptr = out.data();

    if (io_num_.n_output == 1) {
        std::memcpy(dest_ptr, outputs[0].buf, outputs[0].size);
    } else {
        const int n_out = static_cast<int>(io_num_.n_output);
        const int embed_size = info_.embed_size;
        const int n_tokens = info_.image_tokens;
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_out; ++j) {
                const auto offset =
                    static_cast<std::size_t>(i * n_out * embed_size + j * embed_size);
                std::memcpy(dest_ptr + offset,
                            static_cast<float*>(outputs[j].buf) + i * embed_size,
                            sizeof(float) * static_cast<std::size_t>(embed_size));
            }
        }
    }

    rknn_outputs_release(ctx, io_num_.n_output, outputs.data());
    return true;
}

void VisionEncoder::assembleEmbeddings(
    const std::vector<std::optional<std::pair<std::vector<float>, double>>>& slots)
{
    embeddings_.clear();
    frame_times_.clear();
    frame_count_ = 0;
    for (const auto& slot : slots) {
        if (!slot) {
            continue;
        }
        embeddings_.insert(embeddings_.end(), slot->first.begin(), slot->first.end());
        frame_times_.push_back(slot->second);
        ++frame_count_;
    }
}

int VisionEncoder::encodeStreaming(VisionEncodeQueue& queue, int total_hint,
                                   VisionProgressCallback progress)
{
    if (!loaded()) {
        return 0;
    }

    clear();

    std::vector<std::optional<std::pair<std::vector<float>, double>>> slots;
    std::mutex slots_mu;
    int encoded = 0;
    const int progress_total = std::max(total_hint, 1);

    auto worker_fn = [&](int worker_id) {
        const rknn_context ctx = workers_[static_cast<std::size_t>(worker_id)].ctx;
        while (true) {
            if (queue.abort.load()) {
                break;
            }

            int index = 0;
            double time_sec = 0;
            RgbFrame frame;
            {
                std::unique_lock lock(queue.mu);
                queue.cv.wait(lock, [&] {
                    return queue.abort.load() || !queue.pending.empty() || queue.extract_done;
                });
                if (queue.abort.load()) {
                    break;
                }
                if (queue.pending.empty()) {
                    if (queue.extract_done) {
                        break;
                    }
                    continue;
                }
                PendingVisionFrame item = std::move(queue.pending.front());
                queue.pending.pop_front();
                index = item.index;
                time_sec = item.time_sec;
                frame = std::move(item.frame);
            }

            if (frame.empty() || !frame.matches(info_.width, info_.height)) {
                std::cerr << "Vision encode skipped invalid frame index=" << index << '\n';
                continue;
            }

            std::vector<float> embedding;
            if (!processOneImage(ctx, frame, embedding)) {
                std::cerr << "Vision encode failed for frame index=" << index << '\n';
                continue;
            }

            {
                std::lock_guard lock(slots_mu);
                if (index >= static_cast<int>(slots.size())) {
                    slots.resize(static_cast<std::size_t>(index) + 1);
                }
                slots[static_cast<std::size_t>(index)] =
                    std::make_pair(std::move(embedding), time_sec);
                ++encoded;
                if (progress) {
                    progress(encoded, progress_total);
                }
            }
        }
    };

    const int n = std::max(1, worker_count_);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (workers_[static_cast<std::size_t>(i)].ctx == 0) {
            continue;
        }
        threads.emplace_back(worker_fn, i);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    assembleEmbeddings(slots);
    return static_cast<int>(frame_count_);
}

bool VisionEncoder::appendFrame(const RgbFrame& frame)
{
    if (!loaded() || frame.empty() || !frame.matches(info_.width, info_.height)) {
        return false;
    }
    std::vector<float> embedding;
    if (!processOneImage(workers_[0].ctx, frame, embedding)) {
        return false;
    }
    embeddings_.insert(embeddings_.end(), embedding.begin(), embedding.end());
    ++frame_count_;
    return true;
}

bool VisionEncoder::encodeFrames(const std::vector<RgbFrame>& frames,
                                 VisionProgressCallback progress)
{
    if (!loaded() || frames.empty()) {
        return false;
    }

    std::mutex mu;
    std::condition_variable cv;
    std::deque<PendingVisionFrame> pending;
    for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
        pending.push_back(PendingVisionFrame{.index = i, .frame = frames[static_cast<std::size_t>(i)]});
    }
    const bool extract_done = true;
    const std::atomic<bool> abort{false};
    VisionEncodeQueue queue{mu, cv, pending, extract_done, abort};
    cv.notify_all();
    return encodeStreaming(queue, static_cast<int>(frames.size()), progress) > 0;
}

void VisionEncoder::clear()
{
    embeddings_.clear();
    embeddings_.shrink_to_fit();
    frame_times_.clear();
    frame_times_.shrink_to_fit();
    frame_count_ = 0;
}

}  // namespace vlm
