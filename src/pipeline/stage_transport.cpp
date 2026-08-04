#include "pipeline/stage_transport.hpp"

#include <cstdlib>
#include <memory>

#include "pipeline/grpc_stage_transport.hpp"
#include "pipeline/local_stage_transport.hpp"

namespace vlm {

namespace {

[[nodiscard]] bool isDistributedRuntime()
{
    const char* runtime = std::getenv("VLM_RUNTIME");
    if (runtime == nullptr || runtime[0] == '\0') {
        return false;
    }
    const std::string mode(runtime);
    return mode == "distributed";
}

}  // namespace

std::unique_ptr<StageTransport> makeStageTransport(const ModelRegistry& registry,
                                                   const PipelineConfig& config,
                                                   bool distributed)
{
    if (distributed || isDistributedRuntime()) {
        return std::make_unique<GrpcStageTransport>(registry, config);
    }
    return std::make_unique<LocalStageTransport>(registry, config);
}

}  // namespace vlm
