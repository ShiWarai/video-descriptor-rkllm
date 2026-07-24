#pragma once

#include <chrono>
#include <mutex>
#include <string>

namespace vlm {

enum class ServiceStatus { Idle, Busy };

struct StatusSnapshot {
    ServiceStatus status = ServiceStatus::Idle;
    std::string current_job_id;
    std::string loaded_model_id;
    double current_job_elapsed_sec = 0.0;
    double uptime_sec = 0.0;
    bool model_loaded = false;
};

class ServiceState {
public:
    ServiceState() = default;

    void setModelLoaded(bool loaded) noexcept;
    void onJobStarted(const std::string& job_id, const std::string& model_id);
    void onJobFinished();
    void setLoadedModelId(const std::string& model_id);

    [[nodiscard]] StatusSnapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    ServiceStatus status_ = ServiceStatus::Idle;
    std::string current_job_id_;
    std::string loaded_model_id_;
    std::chrono::steady_clock::time_point job_started_;
    std::chrono::steady_clock::time_point started_at_{std::chrono::steady_clock::now()};
    bool model_loaded_ = false;
};

}  // namespace vlm
