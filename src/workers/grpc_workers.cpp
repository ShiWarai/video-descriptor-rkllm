#include "workers/grpc_workers.hpp"

#include <chrono>
#include <iostream>

#include "core/system_memory.hpp"
#include "core/text_util.hpp"
#include "grpc/tensor_codec.hpp"

namespace vlm {

namespace {

[[nodiscard]] double elapsedMs(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

[[nodiscard]] float memAvailableMib()
{
    if (const auto kb = readMemAvailableKb()) {
        return static_cast<float>(*kb) / 1024.0f;
    }
    return 0.0f;
}

}  // namespace

VisionWorkerService::VisionWorkerService(ModelRegistry registry, PipelineConfig config)
    : registry_(std::move(registry)), config_(std::move(config))
{
}

bool VisionWorkerService::ensureVisionModel(std::string_view model_id, std::string* error_out)
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        if (error_out) {
            *error_out = "Unknown model: " + std::string(model_id);
        }
        return false;
    }
    if (loaded_model_id_ == *resolved && vision_.loaded()) {
        return true;
    }
    const ModelSpec* spec = registry_.find(*resolved);
    if (!spec) {
        if (error_out) {
            *error_out = "Model spec not found";
        }
        return false;
    }

    std::string ram_reason;
    const int workers = pickVisionWorkerCount(spec->llm_model_path, spec->vision_model_path,
                                              config_.default_context, VisionEncoder::kWorkerCount,
                                              0, &ram_reason);
    if (workers <= 0) {
        if (error_out) {
            *error_out = ram_reason;
        }
        return false;
    }

    vision_.unload();
    if (!vision_.load(spec->vision_model_path, config_.verbose, workers)) {
        if (error_out) {
            *error_out = "Failed to load vision model";
        }
        return false;
    }
    loaded_model_id_ = spec->id;
    return true;
}

VisionEncodeResult VisionWorkerService::encodeFrames(
    const std::vector<PendingVisionFrame>& frames)
{
    VisionEncodeResult result;
    const auto t0 = std::chrono::steady_clock::now();
    vision_.clear();
    int encoded = 0;
    for (const auto& pending : frames) {
        if (!vision_.appendFrame(pending.frame)) {
            result.error = "Vision appendFrame failed";
            return result;
        }
        ++encoded;
    }
    result.encode_ms = elapsedMs(t0);
    result.embeddings = vision_.embeddings();
    result.frame_times = vision_.frameTimes();
    result.model_info = vision_.modelInfo();
    result.n_image = encoded;
    return result;
}

grpc::Status VisionWorkerService::Encode(grpc::ServerContext* context,
                                         const vlm::v1::EncodeRequest* request,
                                         vlm::v1::EncodeResponse* response)
{
    (void)context;
    ++inflight_;
    struct Guard {
        std::atomic<int>& counter;
        ~Guard() { counter.fetch_sub(1); }
    } guard{inflight_};

    std::lock_guard lock(mu_);
    response->set_job_id(request->job_id());
    std::string error;
    if (!ensureVisionModel(request->model_id(), &error)) {
        response->set_error(error);
        return grpc::Status::OK;
    }

    const auto frames = grpc_util::batchToFrames(request->frames());
    const auto encoded = encodeFrames(frames);
    if (!encoded.ok()) {
        response->set_error(encoded.error);
        return grpc::Status::OK;
    }

    const std::vector<std::int64_t> shape = {
        encoded.n_image, encoded.model_info.image_tokens, encoded.model_info.embed_size};
    *response->mutable_embeddings() =
        grpc_util::floatVectorToTensor(encoded.embeddings, shape);
    for (double time : encoded.frame_times) {
        response->add_frame_times(time);
    }
    *response->mutable_model_info() = grpc_util::toProto(encoded.model_info);
    response->set_n_image(encoded.n_image);
    response->set_encode_ms(static_cast<float>(encoded.encode_ms));
    return grpc::Status::OK;
}

grpc::Status VisionWorkerService::EncodeThenGenerate(
    grpc::ServerContext* context, const vlm::v1::EncodeThenGenerateRequest* request,
    vlm::v1::EncodeThenGenerateResponse* response)
{
    (void)context;
    ++inflight_;
    struct Guard {
        std::atomic<int>& counter;
        ~Guard() { counter.fetch_sub(1); }
    } guard{inflight_};

    std::lock_guard lock(mu_);
    response->set_job_id(request->job_id());
    std::string error;
    if (!ensureVisionModel(request->model_id(), &error)) {
        response->set_error(error);
        return grpc::Status::OK;
    }

    const auto frames = grpc_util::batchToFrames(request->frames());
    const auto encoded = encodeFrames(frames);
    if (!encoded.ok()) {
        response->set_error(encoded.error);
        return grpc::Status::OK;
    }

    vlm::v1::GenerateRequest gen_request;
    gen_request.set_job_id(request->job_id());
    gen_request.set_model_id(request->model_id());
    const std::vector<std::int64_t> shape = {
        encoded.n_image, encoded.model_info.image_tokens, encoded.model_info.embed_size};
    *gen_request.mutable_embeddings() =
        grpc_util::floatVectorToTensor(encoded.embeddings, shape);
    for (double time : encoded.frame_times) {
        gen_request.add_frame_times(time);
    }
    *gen_request.mutable_model_info() = grpc_util::toProto(encoded.model_info);
    gen_request.set_n_image(encoded.n_image);
    gen_request.set_prompt(request->prompt());
    gen_request.set_max_new_tokens(request->max_new_tokens());
    gen_request.set_temperature(request->temperature());
    gen_request.set_enable_thinking(request->enable_thinking());
    gen_request.set_lang(request->lang());

    grpc::ChannelArguments channel_args;
    channel_args.SetMaxReceiveMessageSize(grpc_util::kDefaultMaxMessageBytes);
    channel_args.SetMaxSendMessageSize(grpc_util::kDefaultMaxMessageBytes);
    auto channel = grpc::CreateCustomChannel(request->llm_target(),
                                             grpc::InsecureChannelCredentials(), channel_args);
    auto stub = vlm::v1::LlmService::NewStub(channel);
    vlm::v1::GenerateResponse gen_response;
    grpc::ClientContext client_context;
    client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::minutes(30));
    const grpc::Status status = stub->Generate(&client_context, gen_request, &gen_response);
    if (!status.ok()) {
        response->set_error(status.error_message());
        return grpc::Status::OK;
    }
    if (!gen_response.error().empty()) {
        response->set_error(gen_response.error());
        return grpc::Status::OK;
    }

    response->set_text(gen_response.text());
    response->set_max_new_tokens(gen_response.max_new_tokens());
    response->set_generate_tokens(gen_response.generate_tokens());
    response->set_truncated(gen_response.truncated());
    response->set_encode_ms(static_cast<float>(encoded.encode_ms));
    response->set_generate_ms(gen_response.generate_ms());
    response->set_n_image(encoded.n_image);
    return grpc::Status::OK;
}

grpc::Status VisionWorkerService::Health(grpc::ServerContext* context,
                                         const vlm::v1::HealthRequest* request,
                                         vlm::v1::HealthResponse* response)
{
    (void)context;
    (void)request;
    response->set_ready(vision_.loaded());
    response->set_inflight(inflight_.load());
    response->set_mem_available_mib(memAvailableMib());
    response->set_model_id(loaded_model_id_);
    response->set_service("vision");
    return grpc::Status::OK;
}

LlmWorkerService::LlmWorkerService(ModelRegistry registry, PipelineConfig config)
    : registry_(std::move(registry)), config_(std::move(config))
{
}

bool LlmWorkerService::ensureLlmModel(std::string_view model_id, std::string* error_out)
{
    const auto resolved = registry_.resolveId(model_id);
    if (!resolved) {
        if (error_out) {
            *error_out = "Unknown model: " + std::string(model_id);
        }
        return false;
    }
    if (loaded_model_id_ == *resolved && llm_.loaded()) {
        return true;
    }
    const ModelSpec* spec = registry_.find(*resolved);
    if (!spec) {
        if (error_out) {
            *error_out = "Model spec not found";
        }
        return false;
    }

    llm_.unload();
    const int top_k = spec->top_k.value_or(config_.top_k);
    const float top_p = spec->top_p.value_or(config_.top_p);
    const float temperature = spec->temperature.value_or(config_.temperature);
    const float presence = spec->presence_penalty.value_or(config_.presence_penalty);
    llm_.setSamplingDefaults(top_k, top_p, temperature, presence);
    llm_.setThinkingSamplingDefaults(config_.thinking_temperature, config_.thinking_top_p,
                                    config_.thinking_presence_penalty);
    if (!llm_.load(spec->llm_model_path, config_.default_max_tokens, config_.default_context,
                   config_.default_lang, config_.enable_thinking, config_.verbose)) {
        if (error_out) {
            *error_out = "Failed to load LLM";
        }
        return false;
    }
    loaded_model_id_ = spec->id;
    return true;
}

grpc::Status LlmWorkerService::Generate(grpc::ServerContext* context,
                                        const vlm::v1::GenerateRequest* request,
                                        vlm::v1::GenerateResponse* response)
{
    (void)context;
    ++inflight_;
    struct Guard {
        std::atomic<int>& counter;
        ~Guard() { counter.fetch_sub(1); }
    } guard{inflight_};

    std::lock_guard lock(mu_);
    response->set_job_id(request->job_id());
    std::string error;
    if (!ensureLlmModel(request->model_id(), &error)) {
        response->set_error(error);
        return grpc::Status::OK;
    }

    const auto embeddings = grpc_util::tensorToFloatVector(request->embeddings());
    const auto model_info = grpc_util::fromProto(request->model_info());
    llm_.setOutputLang(request->lang());
    llm_.setEnableThinking(request->enable_thinking());

    const auto t0 = std::chrono::steady_clock::now();
    llm_.clearKvCache();
    const std::string text = llm_.generateMultimodal(
        request->prompt(), embeddings, model_info, static_cast<std::size_t>(request->n_image()),
        request->max_new_tokens(), request->temperature());
    response->set_text(stripThinkingTags(text));
    response->set_max_new_tokens(llm_.lastMaxNewTokens());
    response->set_generate_tokens(llm_.lastGenerateTokens());
    response->set_truncated(llm_.lastTruncatedByMaxTokens());
    response->set_generate_ms(static_cast<float>(elapsedMs(t0)));
    llm_.clearKvCache();
    return grpc::Status::OK;
}

grpc::Status LlmWorkerService::Health(grpc::ServerContext* context,
                                      const vlm::v1::HealthRequest* request,
                                      vlm::v1::HealthResponse* response)
{
    (void)context;
    (void)request;
    response->set_ready(llm_.loaded());
    response->set_inflight(inflight_.load());
    response->set_mem_available_mib(memAvailableMib());
    response->set_model_id(loaded_model_id_);
    response->set_service("llm");
    return grpc::Status::OK;
}

void runVisionWorkerServer(const ModelRegistry& registry, const PipelineConfig& config,
                           const std::string& host, int port)
{
    VisionWorkerService service(registry, config);
    std::string address = host + ':' + std::to_string(port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(grpc_util::kDefaultMaxMessageBytes);
    builder.SetMaxSendMessageSize(grpc_util::kDefaultMaxMessageBytes);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cerr << "vision worker listening on " << address << '\n';
    server->Wait();
}

void runLlmWorkerServer(const ModelRegistry& registry, const PipelineConfig& config,
                        const std::string& host, int port)
{
    LlmWorkerService service(registry, config);
    std::string address = host + ':' + std::to_string(port);
    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(grpc_util::kDefaultMaxMessageBytes);
    builder.SetMaxSendMessageSize(grpc_util::kDefaultMaxMessageBytes);
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cerr << "llm worker listening on " << address << '\n';
    server->Wait();
}

}  // namespace vlm
