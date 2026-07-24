#include <iostream>
#include <string>

#include "api/config.hpp"
#include "api/http_server.hpp"
#include "pipeline/video_context_pipeline.hpp"
#include "runtime/service_state.hpp"

namespace {

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
}

}  // namespace

int main(int argc, char** argv)
{
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

    auto pipeline =
        std::make_shared<vlm::VideoContextPipeline>(std::move(cfg.registry), cfg.pipeline);
    if (!pipeline->initialize(cfg.preload_model_id)) {
        std::cerr << "Failed to initialize pipeline\n";
        return 1;
    }

    logStartup(cfg, *pipeline);

    auto state = std::make_shared<vlm::ServiceState>();
    state->setLoadedModelId(pipeline->loadedModelId());

    vlm::HttpServer server(std::move(cfg), pipeline, state);
    server.run();
    return 0;
}
