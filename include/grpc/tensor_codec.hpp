#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/rgb_frame.hpp"
#include "core/vision_encoder.hpp"
#include "vlm/v1/worker.pb.h"
#include "pipeline/stage_types.hpp"

namespace vlm::grpc_util {

constexpr int kDefaultMaxMessageBytes = 256 * 1024 * 1024;

[[nodiscard]] vlm::v1::Tensor floatVectorToTensor(const std::vector<float>& data,
                                                  const std::vector<std::int64_t>& shape);
[[nodiscard]] std::vector<float> tensorToFloatVector(const vlm::v1::Tensor& tensor);

[[nodiscard]] vlm::v1::VisionModelInfo toProto(const VisionModelInfo& info);
[[nodiscard]] VisionModelInfo fromProto(const vlm::v1::VisionModelInfo& info);

[[nodiscard]] vlm::v1::RgbFrameBatch framesToBatch(const std::vector<PendingVisionFrame>& frames);
[[nodiscard]] std::vector<PendingVisionFrame> batchToFrames(const vlm::v1::RgbFrameBatch& batch);

[[nodiscard]] std::vector<std::string> parseTargets(const char* env_name,
                                                    const char* default_value = "");
[[nodiscard]] std::vector<std::string> expandTargets(const std::vector<std::string>& targets);

}  // namespace vlm::grpc_util
