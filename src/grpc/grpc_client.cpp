#include "grpc/grpc_client.hpp"

#include <chrono>
#include <iostream>

#include "grpc/tensor_codec.hpp"

namespace vlm {

namespace {

grpc::ChannelArguments channelArgs()
{
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(grpc_util::kDefaultMaxMessageBytes);
    args.SetMaxSendMessageSize(grpc_util::kDefaultMaxMessageBytes);
    return args;
}

}  // namespace

VisionClientPool::VisionClientPool(std::vector<std::string> targets)
{
    for (auto& target : targets) {
        endpoints_.push_back(std::make_unique<Endpoint>(std::move(target)));
    }
}

void VisionClientPool::connect()
{
    const auto args = channelArgs();
    for (auto& endpoint : endpoints_) {
        endpoint->channel = grpc::CreateCustomChannel(endpoint->target, grpc::InsecureChannelCredentials(),
                                                      args);
        endpoint->stub = vlm::v1::VisionService::NewStub(endpoint->channel);
    }
}

void VisionClientPool::close()
{
    for (auto& endpoint : endpoints_) {
        endpoint->stub.reset();
        endpoint->channel.reset();
    }
}

VisionClientPool::Endpoint& VisionClientPool::pick()
{
    std::lock_guard lock(mu_);
    Endpoint* best = endpoints_.front().get();
    for (auto& endpoint : endpoints_) {
        if (endpoint->inflight.load() < best->inflight.load()) {
            best = endpoint.get();
        }
    }
    best->inflight.fetch_add(1);
    return *best;
}

vlm::v1::EncodeResponse VisionClientPool::encode(const vlm::v1::EncodeRequest& request)
{
    Endpoint& endpoint = pick();
    vlm::v1::EncodeResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));
    const grpc::Status status = endpoint.stub->Encode(&context, request, &response);
    endpoint.inflight.fetch_sub(1);
    if (!status.ok()) {
        response.set_error(status.error_message());
    }
    return response;
}

vlm::v1::EncodeThenGenerateResponse VisionClientPool::encodeThenGenerate(
    const vlm::v1::EncodeThenGenerateRequest& request)
{
    Endpoint& endpoint = pick();
    vlm::v1::EncodeThenGenerateResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));
    const grpc::Status status = endpoint.stub->EncodeThenGenerate(&context, request, &response);
    endpoint.inflight.fetch_sub(1);
    if (!status.ok()) {
        response.set_error(status.error_message());
    }
    return response;
}

std::vector<vlm::v1::HealthResponse> VisionClientPool::health() const
{
    std::vector<vlm::v1::HealthResponse> responses;
    for (const auto& endpoint : endpoints_) {
        vlm::v1::HealthResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        const grpc::Status status =
            endpoint->stub->Health(&context, vlm::v1::HealthRequest{}, &response);
        if (!status.ok()) {
            response.set_ready(false);
            response.set_service(endpoint->target);
        }
        responses.push_back(std::move(response));
    }
    return responses;
}

LlmClientPool::LlmClientPool(std::vector<std::string> targets)
{
    for (auto& target : targets) {
        endpoints_.push_back(std::make_unique<Endpoint>(std::move(target)));
    }
}

void LlmClientPool::connect()
{
    const auto args = channelArgs();
    for (auto& endpoint : endpoints_) {
        endpoint->channel = grpc::CreateCustomChannel(endpoint->target, grpc::InsecureChannelCredentials(),
                                                      args);
        endpoint->stub = vlm::v1::LlmService::NewStub(endpoint->channel);
    }
}

void LlmClientPool::close()
{
    for (auto& endpoint : endpoints_) {
        endpoint->stub.reset();
        endpoint->channel.reset();
    }
}

LlmClientPool::Endpoint& LlmClientPool::pick()
{
    std::lock_guard lock(mu_);
    Endpoint* best = endpoints_.front().get();
    for (auto& endpoint : endpoints_) {
        if (endpoint->inflight.load() < best->inflight.load()) {
            best = endpoint.get();
        }
    }
    best->inflight.fetch_add(1);
    return *best;
}

std::string LlmClientPool::acquireTarget()
{
    return pick().target;
}

void LlmClientPool::releaseTarget(const std::string& target)
{
    std::lock_guard lock(mu_);
    for (auto& endpoint : endpoints_) {
        if (endpoint->target == target) {
            endpoint->inflight.fetch_sub(1);
            return;
        }
    }
}

vlm::v1::GenerateResponse LlmClientPool::generate(const vlm::v1::GenerateRequest& request)
{
    Endpoint& endpoint = pick();
    vlm::v1::GenerateResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));
    const grpc::Status status = endpoint.stub->Generate(&context, request, &response);
    endpoint.inflight.fetch_sub(1);
    if (!status.ok()) {
        response.set_error(status.error_message());
    }
    return response;
}

std::vector<vlm::v1::HealthResponse> LlmClientPool::health() const
{
    std::vector<vlm::v1::HealthResponse> responses;
    for (const auto& endpoint : endpoints_) {
        vlm::v1::HealthResponse response;
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        const grpc::Status status =
            endpoint->stub->Health(&context, vlm::v1::HealthRequest{}, &response);
        if (!status.ok()) {
            response.set_ready(false);
            response.set_service(endpoint->target);
        }
        responses.push_back(std::move(response));
    }
    return responses;
}

}  // namespace vlm
