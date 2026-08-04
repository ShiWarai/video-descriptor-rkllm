#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <grpcpp/grpcpp.h>

#include "core/llm_runtime.hpp"
#include "core/vision_encoder.hpp"
#include "grpc/grpc_client.hpp"
#include "pipeline/stage_types.hpp"
#include "vlm/v1/worker.grpc.pb.h"
#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace vlm {

class VisionWorkerService final : public vlm::v1::VisionService::Service {
public:
    VisionWorkerService(ModelRegistry registry, PipelineConfig config);

    grpc::Status Encode(grpc::ServerContext* context, const vlm::v1::EncodeRequest* request,
                        vlm::v1::EncodeResponse* response) override;
    grpc::Status EncodeThenGenerate(grpc::ServerContext* context,
                                    const vlm::v1::EncodeThenGenerateRequest* request,
                                    vlm::v1::EncodeThenGenerateResponse* response) override;
    grpc::Status Health(grpc::ServerContext* context, const vlm::v1::HealthRequest* request,
                        vlm::v1::HealthResponse* response) override;

private:
    [[nodiscard]] bool ensureVisionModel(std::string_view model_id, std::string* error_out);
    [[nodiscard]] VisionEncodeResult encodeFrames(const std::vector<PendingVisionFrame>& frames);

    ModelRegistry registry_;
    PipelineConfig config_;
    VisionEncoder vision_;
    std::string loaded_model_id_;
    std::atomic<int> inflight_{0};
    std::mutex mu_;
};

class LlmWorkerService final : public vlm::v1::LlmService::Service {
public:
    LlmWorkerService(ModelRegistry registry, PipelineConfig config);

    grpc::Status Generate(grpc::ServerContext* context, const vlm::v1::GenerateRequest* request,
                          vlm::v1::GenerateResponse* response) override;
    grpc::Status Health(grpc::ServerContext* context, const vlm::v1::HealthRequest* request,
                        vlm::v1::HealthResponse* response) override;

private:
    [[nodiscard]] bool ensureLlmModel(std::string_view model_id, std::string* error_out);

    ModelRegistry registry_;
    PipelineConfig config_;
    LlmRuntime llm_;
    std::string loaded_model_id_;
    std::atomic<int> inflight_{0};
    std::mutex mu_;
};

void runVisionWorkerServer(const ModelRegistry& registry, const PipelineConfig& config,
                           const std::string& host, int port);
void runLlmWorkerServer(const ModelRegistry& registry, const PipelineConfig& config,
                        const std::string& host, int port);

}  // namespace vlm
