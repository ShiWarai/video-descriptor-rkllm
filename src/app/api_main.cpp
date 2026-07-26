#include <cstdlib>
#include <iostream>
#include <string>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "api/config.hpp"
#include "api/http_server.hpp"
#include "pipeline/video_context_pipeline.hpp"
#include "runtime/service_state.hpp"

namespace {

void configureAllocator()
{
#if defined(__GLIBC__)
    // Fewer arenas → less retained RSS under multi-threaded httplib.
    mallopt(M_ARENA_MAX, 2);
    // Return free pages to the OS more eagerly after large RKLLM/vision allocations.
    mallopt(M_TRIM_THRESHOLD, 64 * 1024);
    mallopt(M_MMAP_THRESHOLD, 256 * 1024);
#endif
}

void logStartup(const vlm::ServerConfig& cfg, const vlm::VideoContextPipeline& pipeline)
{
    const auto& registry = pipeline.registry();
    std::cerr << "vlm_api_server config: " << registry.models().size() << " model(s)\n";
    for (const auto& model : registry.models()) {
        std::cerr << "  - " << model.id << '\n';
    }
    std::cerr << "default_model: " << registry.defaultModelId() << '\n';
    if (cfg.preload_model_id) {
        std::cerr << "preload: " << *cfg.preload_model_id;
        if (!pipeline.loadedModelId().empty()) {
            std::cerr << " (loaded)";
        }
        std::cerr << '\n';
    } else {
        std::cerr << "preload: none (lazy load on first request)\n";
    }
    if (cfg.pipeline.verbose) {
        std::cerr << "verbose: on\n";
    }
    if (!cfg.pipeline.whisper_url.empty()) {
        std::cerr << "whisper: " << cfg.pipeline.whisper_url;
        if (!cfg.pipeline.whisper_api_key.empty()) {
            std::cerr << " (auth)";
        }
        std::cerr << '\n';
    } else {
        std::cerr << "whisper: disabled\n";
    }
    if (!cfg.api_key.empty()) {
        std::cerr << "api auth: enabled (Bearer token)\n";
    }
}

}  // namespace

int main(int argc, char** argv)
{
    configureAllocator();
    std::string config_path = "config.json";
    bool verbose_cli = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            verbose_cli = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: vlm_api_server [--config config.json] [--verbose]\n";
            return 0;
        }
    }

    vlm::ServerConfig cfg;
    try {
        cfg = vlm::loadConfigFile(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << '\n';
        return 1;
    }
    if (verbose_cli) {
        cfg.pipeline.verbose = true;
    }
    if (const char* whisper_url = std::getenv("WHISPER_RKNN_URL")) {
        if (whisper_url[0] != '\0') {
            cfg.pipeline.whisper_url = whisper_url;
        }
    }
    if (const char* whisper_api_key = std::getenv("WHISPER_API_KEY")) {
        if (whisper_api_key[0] != '\0') {
            cfg.pipeline.whisper_api_key = whisper_api_key;
        }
    }
    if (const char* verbose_env = std::getenv("VLM_VERBOSE")) {
        if (verbose_env[0] == '1' || verbose_env[0] == 't' || verbose_env[0] == 'T' ||
            verbose_env[0] == 'y' || verbose_env[0] == 'Y') {
            cfg.pipeline.verbose = true;
        }
    }
    if (const char* api_key = std::getenv("VLM_API_KEY")) {
        if (api_key[0] != '\0') {
            cfg.api_key = api_key;
        }
    }
    if (cfg.pipeline.workdir.empty()) {
        cfg.pipeline.workdir = cfg.workdir;
    }

    auto pipeline =
        std::make_shared<vlm::VideoContextPipeline>(std::move(cfg.registry), cfg.pipeline);

    auto state = std::make_shared<vlm::ServiceState>();

    logStartup(cfg, *pipeline);

    vlm::HttpServer server(std::move(cfg), pipeline, state);
    server.run();
    return 0;
}
