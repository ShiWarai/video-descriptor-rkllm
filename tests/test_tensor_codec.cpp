#include <cmath>
#include <iostream>
#include <vector>

#include "core/rgb_frame.hpp"
#include "core/vision_encoder.hpp"
#include "grpc/tensor_codec.hpp"

namespace {

template <typename Fn>
bool expect(bool condition, const char* message, Fn&& on_fail)
{
    if (!condition) {
        on_fail(message);
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    int failures = 0;
    auto fail = [&](const char* message) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    };

    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto tensor = vlm::grpc_util::floatVectorToTensor(data, {2, 2});
  if (!expect(tensor.shape_size() == 2, "shape size", fail) ||
        !expect(tensor.dtype() == "float32", "dtype", fail)) {
        return 1;
    }
    const auto roundtrip = vlm::grpc_util::tensorToFloatVector(tensor);
    if (!expect(roundtrip.size() == data.size(), "roundtrip size", fail)) {
        return 1;
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!expect(std::fabs(roundtrip[i] - data[i]) < 1e-6f, "roundtrip value", fail)) {
            return 1;
        }
    }

    vlm::VisionModelInfo info{.width = 448, .height = 448, .channels = 3, .image_tokens = 196,
                              .embed_size = 2048};
    const auto proto = vlm::grpc_util::toProto(info);
    const auto restored = vlm::grpc_util::fromProto(proto);
    if (!expect(restored.width == info.width && restored.image_tokens == info.image_tokens,
                "vision model info roundtrip", fail)) {
        return 1;
    }

    std::vector<vlm::PendingVisionFrame> frames;
    frames.push_back(vlm::PendingVisionFrame{
        .index = 0,
        .time_sec = 1.5,
        .frame = vlm::RgbFrame::fromRaw(2, 2, std::vector<std::uint8_t>(12, 7)),
    });
    const auto batch = vlm::grpc_util::framesToBatch(frames);
    const auto decoded = vlm::grpc_util::batchToFrames(batch);
    if (!expect(decoded.size() == 1, "batch size", fail) ||
        !expect(std::fabs(decoded[0].time_sec - 1.5) < 1e-6, "frame time", fail) ||
        !expect(decoded[0].frame.width == 2, "frame width", fail)) {
        return 1;
    }

    if (failures == 0) {
        std::cout << "test_tensor_codec: ok\n";
        return 0;
    }
    return 1;
}
