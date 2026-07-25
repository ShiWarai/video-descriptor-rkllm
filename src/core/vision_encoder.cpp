#include "core/vision_encoder.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace vlm {

VisionEncoder::VisionEncoder() = default;

VisionEncoder::~VisionEncoder()
{
    unload();
}

void VisionEncoder::unload()
{
    if (ctx_) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
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

bool VisionEncoder::initFromPath(std::string_view model_path)
{
    const std::string path(model_path);
    int ret = rknn_init(&ctx_, const_cast<char*>(path.c_str()), 0, 0, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_init failed: " << ret << '\n';
        return false;
    }

    rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1_2);
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));

    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; ++i) {
        input_attrs_[i] = {};
        input_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        dumpTensorAttr(input_attrs_[i]);
    }

    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i] = {};
        output_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
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
    return true;
}

bool VisionEncoder::load(const std::string& model_path, bool verbose)
{
    unload();
    verbose_ = verbose;
    return initFromPath(model_path);
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

int VisionEncoder::processOneImage(const RgbFrame& rgb)
{
    rknn_input inputs[1]{};
    std::vector<rknn_output> outputs(io_num_.n_output);

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = static_cast<uint32_t>(rgb.byteSize());
    inputs[0].buf = const_cast<void*>(static_cast<const void*>(rgb.data()));

    int ret = rknn_inputs_set(ctx_, 1, inputs);
    if (ret < 0) {
        return ret;
    }
    ret = rknn_run(ctx_, nullptr);
    if (ret < 0) {
        return ret;
    }

    for (auto& output : outputs) {
        output.want_float = 1;
    }
    ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
    if (ret < 0) {
        return ret;
    }

    const auto floats_per_image =
        static_cast<std::size_t>(info_.image_tokens) * static_cast<std::size_t>(info_.embed_size) *
        static_cast<std::size_t>(io_num_.n_output);
    const auto current_size = embeddings_.size();
    embeddings_.resize(current_size + floats_per_image);
    float* dest_ptr = embeddings_.data() + current_size;

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

    rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
    return 0;
}

bool VisionEncoder::appendFrame(const RgbFrame& frame)
{
    if (!loaded() || frame.empty() || !frame.matches(info_.width, info_.height)) {
        return false;
    }
    if (processOneImage(frame) != 0) {
        return false;
    }
    ++frame_count_;
    return true;
}

bool VisionEncoder::encodeFrames(const std::vector<RgbFrame>& frames,
                                 VisionProgressCallback progress)
{
    clear();
    const int total = static_cast<int>(frames.size());
    for (int i = 0; i < total; ++i) {
        if (progress) {
            progress(i, total);
        }
        if (!appendFrame(frames[static_cast<std::size_t>(i)])) {
            std::cerr << "Vision encode failed for frame " << i << '\n';
        }
    }
    if (progress) {
        progress(total, total);
    }
    return frame_count_ > 0;
}

void VisionEncoder::clear()
{
    embeddings_.clear();
    embeddings_.shrink_to_fit();
    frame_count_ = 0;
}

}  // namespace vlm
