#include "runtime/service_state.hpp"

namespace vlm {

void ServiceState::setModelLoaded(bool loaded) noexcept
{
    std::lock_guard lock(mutex_);
    model_loaded_ = loaded;
}

void ServiceState::setReady(bool ready) noexcept
{
    std::lock_guard lock(mutex_);
    ready_ = ready;
}

bool ServiceState::isReady() const noexcept
{
    std::lock_guard lock(mutex_);
    return ready_;
}

void ServiceState::onJobStarted(const std::string& job_id, const std::string& /*model_id*/)
{
    std::lock_guard lock(mutex_);
    status_ = ServiceStatus::Busy;
    current_job_id_ = job_id;
    job_started_ = std::chrono::steady_clock::now();
    jobs_.beginJob(job_id);
}

void ServiceState::onJobFinished(bool ok, std::string_view error)
{
    std::lock_guard lock(mutex_);
    if (!current_job_id_.empty()) {
        jobs_.finishJob(current_job_id_, ok, error);
    }
    current_job_id_.clear();
    status_ = ServiceStatus::Idle;
}

void ServiceState::updateJobProgress(const std::string& job_id, const JobProgressUpdate& update)
{
    std::lock_guard lock(mutex_);
    jobs_.updateJob(job_id, update);
}

std::optional<JobProgressSnapshot> ServiceState::jobProgress(std::string_view job_id) const
{
    std::lock_guard lock(mutex_);
    return jobs_.snapshot(job_id);
}

void ServiceState::setLoadedModelId(const std::string& model_id)
{
    std::lock_guard lock(mutex_);
    loaded_model_id_ = model_id;
    model_loaded_ = !model_id.empty();
}

StatusSnapshot ServiceState::snapshot() const
{
    std::lock_guard lock(mutex_);
    StatusSnapshot snap;
    snap.status = status_;
    snap.current_job_id = current_job_id_;
    snap.loaded_model_id = loaded_model_id_;
    snap.model_loaded = model_loaded_;
    snap.ready = ready_;
    snap.uptime_sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at_).count();
    if (!current_job_id_.empty()) {
        snap.current_job_elapsed_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - job_started_).count();
    }
    return snap;
}

}  // namespace vlm
