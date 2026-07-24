#include "api/http_server.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>

#include "api/openai_handlers.hpp"

namespace vlm {

namespace {

std::string statusToString(ServiceStatus s)
{
    switch (s) {
        case ServiceStatus::Idle:
            return "idle";
        case ServiceStatus::Busy:
            return "busy";
    }
    return "unknown";
}

nlohmann::json statusJson(const StatusSnapshot& snap)
{
    return {{"status", statusToString(snap.status)},
            {"current_job_id", snap.current_job_id},
            {"loaded_model_id", snap.loaded_model_id},
            {"current_job_elapsed_sec", snap.current_job_elapsed_sec},
            {"uptime_sec", snap.uptime_sec},
            {"model_loaded", snap.model_loaded}};
}

std::string makeJobId()
{
    static std::atomic<std::uint64_t> counter{0};
    const auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "job-" << now << '-' << counter.fetch_add(1);
    return oss.str();
}

std::optional<std::string> formField(const httplib::Request& req, const std::string& key)
{
    if (req.has_param(key)) {
        return req.get_param_value(key);
    }
    if (req.has_file(key)) {
        const auto& field = req.get_file_value(key);
        if (field.filename.empty()) {
            return field.content;
        }
    }
    return std::nullopt;
}

}  // namespace

HttpServer::HttpServer(ServerConfig config, std::shared_ptr<VideoContextPipeline> pipeline,
                       std::shared_ptr<ServiceState> state)
    : config_(std::move(config)),
      pipeline_(std::move(pipeline)),
      state_(std::move(state))
{
}

AnalyzeResult HttpServer::runInference(AnalyzeRequest request)
{
    const std::string job_id = makeJobId();
    const std::string model_label = request.model.empty()
                                        ? pipeline_->registry().defaultModelId()
                                        : request.model;
    const auto video_path = request.video_path;

    std::cerr << '[' << job_id << "] analyze start model=" << model_label
              << " video=" << video_path << " frames=" << request.max_frames
              << " frame_budget=" << (request.frame_budget > 0 ? request.frame_budget
                                                               : config_.pipeline.frame_budget)
              << '\n';

    const auto wall_start = std::chrono::steady_clock::now();
    const auto queue_start = wall_start;
    std::lock_guard lock(inference_mutex_);
    const double queue_wait_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - queue_start)
            .count();

    state_->onJobStarted(job_id, request.model);
    AnalyzeResult result = pipeline_->analyze(request);
    state_->setLoadedModelId(pipeline_->loadedModelId());
    state_->onJobFinished();

    removeWorkFileIfOwned(video_path, config_.workdir);

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start)
            .count();
    result.metrics["queue_wait_ms"] = queue_wait_ms;
    result.metrics["wall_ms"] = wall_ms;

    std::cerr << '[' << job_id << "] analyze done ok=" << (result.ok() ? "yes" : "no")
              << " model=" << result.model << " frames_used=" << result.frames_used
              << " frame_budget=" << result.frame_budget << " wall_ms=" << wall_ms;
    if (const auto it = result.metrics.find("total_ms"); it != result.metrics.end()) {
        std::cerr << " pipeline_ms=" << it->second;
    }
    if (!result.ok()) {
        std::cerr << " error=\"" << result.error << '"';
    }
    std::cerr << '\n';

    return result;
}

void HttpServer::run()
{
    clearWorkdir(config_.workdir);
    std::cerr << "workdir cleared: " << config_.workdir << '\n';

    httplib::Server svr;

    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path == "/health") {
            return;
        }
        std::cerr << req.method << ' ' << req.path << " -> " << res.status << '\n';
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(statusJson(state_->snapshot()).dump(), "application/json");
    });

    svr.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(statusJson(state_->snapshot()).dump(), "application/json");
    });

    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        nlohmann::json data = nlohmann::json::array();
        for (const auto& model : pipeline_->registry().models()) {
            data.push_back({{"id", model.id},
                            {"object", "model"},
                            {"owned_by", "vlm"},
                            {"input_modalities", nlohmann::json::array({"video"})}});
        }
        res.set_content(nlohmann::json{{"object", "list"}, {"data", data}}.dump(),
                        "application/json");
    });

    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        std::cerr << "POST /v1/chat/completions received (" << req.body.size() << " bytes)\n";

        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception&) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON"})", "application/json");
            return;
        }

        const auto parsed = parseChatCompletionRequest(body, config_.pipeline);
        if (parsed.video_url.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"video_url required in messages"})", "application/json");
            return;
        }

        std::string video_path;
        try {
            video_path = saveVideoFromUrl(parsed.video_url, config_.workdir);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        AnalyzeRequest areq{
            .video_path = video_path,
            .model = parsed.model,
            .max_frames = parsed.max_frames,
            .frame_budget = parsed.frame_budget,
            .max_tokens = parsed.max_tokens,
            .lang = parsed.lang,
            .prompt_mode = parsed.prompt_mode,
            .enable_thinking = parsed.enable_thinking,
            .temperature = parsed.temperature,
            .transcript_override = parsed.transcript,
        };

        const auto result = runInference(std::move(areq));
        const std::string model = result.model.empty()
                                      ? pipeline_->registry().defaultModelId()
                                      : result.model;
        if (!result.ok()) {
            res.status = 500;
        }
        res.set_content(buildChatCompletion(model, result).dump(), "application/json");
    });

    svr.Post("/v1/video/analyze", [this](const httplib::Request& req, httplib::Response& res) {
        std::cerr << "POST /v1/video/analyze received";
        if (req.has_file("file")) {
            std::cerr << " file=" << req.get_file_value("file").filename;
        }
        std::cerr << '\n';

        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(R"({"error":"file field required"})", "application/json");
            return;
        }

        const auto& file = req.get_file_value("file");
        if (file.filename.empty() || file.content.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"empty video upload"})", "application/json");
            return;
        }

        std::filesystem::path saved;
        try {
            saved = saveUploadedFile(file.filename, file.content, config_.workdir);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        AnalyzeRequest areq{.video_path = saved};
        if (const auto model = formField(req, "model")) {
            areq.model = *model;
        }
        if (const auto frames = formField(req, "frames")) {
            areq.max_frames = std::stoi(*frames);
        } else {
            areq.max_frames = config_.pipeline.default_frames;
        }
        if (const auto frame_budget = formField(req, "frame_budget")) {
            areq.frame_budget = std::stoi(*frame_budget);
        }
        if (const auto lang = formField(req, "lang")) {
            areq.lang = *lang;
        }
        if (const auto prompt_mode = formField(req, "prompt_mode")) {
            areq.prompt_mode = *prompt_mode;
        }
        if (const auto max_tokens = formField(req, "max_tokens")) {
            areq.max_tokens = std::stoi(*max_tokens);
        }
        if (const auto thinking = formField(req, "enable_thinking")) {
            const std::string v = *thinking;
            areq.enable_thinking = (v == "1" || v == "true" || v == "yes" || v == "on");
        }
        if (const auto temperature = formField(req, "temperature")) {
            areq.temperature = std::stof(*temperature);
        }
        if (const auto transcript = formField(req, "transcript")) {
            areq.transcript_override = *transcript;
        }

        std::cerr << "  params: model=" << (areq.model.empty() ? "(default)" : areq.model)
                  << " frames=" << areq.max_frames << " frame_budget=" << areq.frame_budget
                  << " max_tokens=" << areq.max_tokens << " lang=" << areq.lang
                  << " prompt_mode=" << areq.prompt_mode
                  << " thinking="
                  << (areq.enable_thinking
                          ? (*areq.enable_thinking ? "true" : "false")
                          : "config")
                  << " temperature="
                  << (areq.temperature ? std::to_string(*areq.temperature) : "config")
                  << '\n';

        const auto result = runInference(std::move(areq));
        if (!result.ok()) {
            res.status = 500;
        }
        res.set_content(analyzeResultToJson(result).dump(), "application/json");
    });

    std::cerr << "vlm_api_server listening on " << config_.host << ':' << config_.port << '\n';
    svr.listen(config_.host.c_str(), config_.port);
}

}  // namespace vlm
