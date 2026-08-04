#include <cstdlib>
#include <iostream>
#include <string>

#include "api/config.hpp"
#include "workers/grpc_workers.hpp"

namespace {

[[nodiscard]] std::string envOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string(fallback);
}

[[nodiscard]] int envPort(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::stoi(value);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string config_path = "config.json";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cerr << "Usage: vlm_vision_worker [--config config.json]\n";
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

    const std::string host = envOr("GRPC_HOST", "0.0.0.0");
    const int port = envPort("GRPC_PORT", 50051);
    vlm::runVisionWorkerServer(cfg.registry, cfg.pipeline, host, port);
    return 0;
}
