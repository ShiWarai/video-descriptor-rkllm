#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "vlm/v1/worker.grpc.pb.h"

namespace vlm {

class VisionClientPool {
public:
    explicit VisionClientPool(std::vector<std::string> targets);

    void connect();
    void close();

    [[nodiscard]] vlm::v1::EncodeResponse encode(const vlm::v1::EncodeRequest& request);
    [[nodiscard]] vlm::v1::EncodeThenGenerateResponse encodeThenGenerate(
        const vlm::v1::EncodeThenGenerateRequest& request);
    [[nodiscard]] std::vector<vlm::v1::HealthResponse> health() const;

private:
    struct Endpoint {
        explicit Endpoint(std::string target_in) : target(std::move(target_in)) {}
        std::string target;
        std::atomic<int> inflight{0};
        std::shared_ptr<grpc::Channel> channel;
        std::unique_ptr<vlm::v1::VisionService::Stub> stub;
    };

    [[nodiscard]] Endpoint& pick();

    mutable std::mutex mu_;
    std::vector<std::unique_ptr<Endpoint>> endpoints_;
};

class LlmClientPool {
public:
    explicit LlmClientPool(std::vector<std::string> targets);

    void connect();
    void close();

    [[nodiscard]] std::string acquireTarget();
    void releaseTarget(const std::string& target);

    [[nodiscard]] vlm::v1::GenerateResponse generate(const vlm::v1::GenerateRequest& request);
    [[nodiscard]] std::vector<vlm::v1::HealthResponse> health() const;

private:
    struct Endpoint {
        explicit Endpoint(std::string target_in) : target(std::move(target_in)) {}
        std::string target;
        std::atomic<int> inflight{0};
        std::shared_ptr<grpc::Channel> channel;
        std::unique_ptr<vlm::v1::LlmService::Stub> stub;
    };

    [[nodiscard]] Endpoint& pick();

    mutable std::mutex mu_;
    std::vector<std::unique_ptr<Endpoint>> endpoints_;
};

}  // namespace vlm
