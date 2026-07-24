#pragma once

#include <memory>
#include <mutex>

#include "api/config.hpp"
#include "pipeline/video_context_pipeline.hpp"
#include "runtime/service_state.hpp"

namespace vlm {

class HttpServer {
public:
    HttpServer(ServerConfig config, std::shared_ptr<VideoContextPipeline> pipeline,
               std::shared_ptr<ServiceState> state);

    void run();

private:
    [[nodiscard]] AnalyzeResult runInference(AnalyzeRequest request);

    ServerConfig config_;
    std::shared_ptr<VideoContextPipeline> pipeline_;
    std::shared_ptr<ServiceState> state_;
    std::mutex inference_mutex_;
};

}  // namespace vlm
