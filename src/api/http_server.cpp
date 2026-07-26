#include "api/http_server.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "api/openai_handlers.hpp"
#include "api/auth.hpp"
#include "core/subprocess.hpp"
#include "runtime/job_progress.hpp"

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
    const std::string status = snap.ready ? statusToString(snap.status) : "loading";
    return {{"status", status},
            {"ready", snap.ready},
            {"current_job_id", snap.current_job_id},
            {"loaded_model_id", snap.loaded_model_id},
            {"current_job_elapsed_sec", snap.current_job_elapsed_sec},
            {"uptime_sec", snap.uptime_sec},
            {"model_loaded", snap.model_loaded}};
}

bool downloadModelsIfRequested()
{
    const char* download = std::getenv("VLM_DOWNLOAD_MODELS");
    if (download == nullptr || download[0] == '\0' || std::string_view(download) == "0") {
        return true;
    }

    const char* app_dir = std::getenv("APP_DIR");
    const char* models_dir = std::getenv("MODELS_DIR");
    const std::string app = app_dir != nullptr ? app_dir : "/app";
    const std::string models = models_dir != nullptr ? models_dir : "/app/models";
    const char* model_urls = std::getenv("VLM_MODEL_URLS");
    const auto script = std::filesystem::path(app) / "scripts" / "download_models.sh";

    std::vector<std::pair<std::string, std::string>> env_overrides{
        {"MODELS_DIR", models},
        {"VLM_DOWNLOAD_MODELS", download},
    };
    if (model_urls != nullptr && model_urls[0] != '\0') {
        env_overrides.emplace_back("VLM_MODEL_URLS", model_urls);
    }

    std::cerr << "startup: downloading models (" << download << ") into " << models << '\n';
    return runBashScript(script, env_overrides);
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

    struct JobGuard {
        ServiceState* state = nullptr;
        std::filesystem::path video_path;
        std::filesystem::path workdir;
        bool started = false;
        bool finished = false;
        ~JobGuard()
        {
            if (started && state && !finished) {
                state->onJobFinished(false, "job interrupted");
            }
            removeWorkFileIfOwned(video_path, workdir);
        }
    } guard{state_.get(), video_path, config_.workdir, false};

    request.job_id = job_id;
    request.on_progress = [this, job_id](const JobProgressUpdate& update) {
        state_->updateJobProgress(job_id, update);
    };

    state_->onJobStarted(job_id, request.model);
    guard.started = true;

    AnalyzeResult result;
    try {
        result = pipeline_->analyze(request);
    } catch (const std::exception& e) {
        result.job_id = job_id;
        result.error = e.what();
        std::cerr << '[' << job_id << "] analyze exception: " << e.what() << '\n';
    } catch (...) {
        result.job_id = job_id;
        result.error = "Unknown analyze failure";
        std::cerr << '[' << job_id << "] analyze unknown exception" << '\n';
    }
    state_->setLoadedModelId(pipeline_->loadedModelId());
    state_->onJobFinished(result.ok(), result.error);
    guard.finished = true;
    result.job_id = job_id;

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

void HttpServer::runStartup()
{
    if (!downloadModelsIfRequested()) {
        std::cerr << "startup: model download failed\n";
        return;
    }

    if (!pipeline_->initialize(config_.preload_model_id)) {
        std::cerr << "startup: pipeline initialization failed\n";
        return;
    }

    state_->setLoadedModelId(pipeline_->loadedModelId());
    state_->setReady(true);
    std::cerr << "startup: service ready\n";
}

void HttpServer::run()
{
    clearWorkdir(config_.workdir);
    std::cerr << "workdir cleared: " << config_.workdir << '\n';

    httplib::Server svr;
    // Cap upload size so concurrent buffered bodies cannot grow unbounded.
    constexpr std::size_t kMaxUploadBytes = 512ull * 1024ull * 1024ull;  // 512 MiB
    svr.set_payload_max_length(kMaxUploadBytes);

    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path == "/health" || req.path == "/ready" || req.path == "/v1/status"
            || req.path.rfind("/v1/jobs/", 0) == 0) {
            return;
        }
        std::cerr << req.method << ' ' << req.path << " -> " << res.status << '\n';
    });

    const std::string api_key = config_.api_key;
    if (!api_key.empty()) {
        std::cerr << "api auth: Bearer token required for /v1/*\n";
    }

    svr.set_pre_routing_handler([&api_key](const httplib::Request& req, httplib::Response& res) {
        if (!authRequiredForPath(req.path)) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        const std::optional<std::string> token =
            req.has_header("Authorization")
                ? parseBearerToken(req.get_header_value("Authorization"))
                : std::nullopt;
        if (!isApiKeyValid(token, api_key)) {
            res.status = 401;
            res.set_content(unauthorizedErrorJson(), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(statusJson(state_->snapshot()).dump(), "application/json");
    });

    svr.Get("/ready", [this](const httplib::Request&, httplib::Response& res) {
        const auto snap = state_->snapshot();
        res.status = snap.ready ? 200 : 503;
        res.set_content(statusJson(snap).dump(), "application/json");
    });

    svr.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(statusJson(state_->snapshot()).dump(), "application/json");
    });

    svr.Get(R"(/v1/jobs/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) {
        if (!state_->isReady()) {
            res.status = 503;
            res.set_content(R"({"error":"service starting"})", "application/json");
            return;
        }
        const std::string job_id = req.matches.size() > 1 ? req.matches[1].str() : "";
        if (job_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"job id required"})", "application/json");
            return;
        }
        const auto progress = state_->jobProgress(job_id);
        if (!progress) {
            res.status = 404;
            res.set_content(nlohmann::json{{"error", "job not found"}, {"job_id", job_id}}.dump(),
                            "application/json");
            return;
        }
        res.set_content(jobProgressToJson(*progress).dump(), "application/json");
    });

    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        if (!state_->isReady()) {
            res.status = 503;
            res.set_content(R"({"error":"service starting"})", "application/json");
            return;
        }
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
        if (!state_->isReady()) {
            res.status = 503;
            res.set_content(R"({"error":"service starting"})", "application/json");
            return;
        }
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
        if (!state_->isReady()) {
            res.status = 503;
            res.set_content(R"({"error":"service starting"})", "application/json");
            return;
        }
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
        try {
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
        } catch (const std::exception& e) {
            removeWorkFileIfOwned(saved, config_.workdir);
            res.status = 400;
            res.set_content(nlohmann::json{{"error", std::string("invalid form field: ") + e.what()}}
                                .dump(),
                            "application/json");
            return;
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

    if (!svr.bind_to_port(config_.host.c_str(), config_.port)) {
        std::cerr << "Failed to bind " << config_.host << ':' << config_.port << '\n';
        return;
    }

    std::cerr << "vlm_api_server listening on " << config_.host << ':' << config_.port << '\n';
    std::thread server_thread([&] { svr.listen_after_bind(); });

    runStartup();

    server_thread.join();
}

}  // namespace vlm
