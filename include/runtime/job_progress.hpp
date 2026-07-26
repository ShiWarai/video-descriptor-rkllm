#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace vlm {

/** Client-visible pipeline stage (machine id). */
inline constexpr std::string_view kJobStageQueued = "queued";
inline constexpr std::string_view kJobStageLoadingModel = "loading_model";
inline constexpr std::string_view kJobStageExtractingFrames = "extracting_frames";
inline constexpr std::string_view kJobStageEncodingVision = "encoding_vision";
inline constexpr std::string_view kJobStageTranscribing = "transcribing";
inline constexpr std::string_view kJobStageGenerating = "generating";
inline constexpr std::string_view kJobStageDone = "done";
inline constexpr std::string_view kJobStageFailed = "failed";

struct JobProgressUpdate {
    std::string stage;
    int frames_done = -1;
    int frames_total = -1;
    int vision_done = -1;
    int vision_total = -1;
    int generate_tokens = -1;
    int max_new_tokens = -1;
    bool model_load_done = false;
    bool transcript_done = false;
};

struct JobProgressSnapshot {
    std::string job_id;
    std::string status;  // running | done | failed
    std::string stage;
    std::string stage_label;
    double elapsed_sec = 0.0;
    double progress_percent = 0.0;
    nlohmann::json details = nlohmann::json::object();
};

using JobProgressCallback = void (*)(void* userdata, const JobProgressUpdate& update);

[[nodiscard]] std::string jobStageLabel(std::string_view stage);

/** Estimate 0..100 from stage + partial counters (RK3588 typical timings). */
[[nodiscard]] double estimateJobProgressPercent(const JobProgressUpdate& state,
                                                double elapsed_sec) noexcept;

[[nodiscard]] nlohmann::json jobProgressToJson(const JobProgressSnapshot& snap);

class JobProgressTracker {
public:
    void beginJob(std::string job_id);
    void updateJob(std::string_view job_id, const JobProgressUpdate& update);
    void finishJob(std::string_view job_id, bool ok, std::string_view error = {});
    [[nodiscard]] std::optional<JobProgressSnapshot> snapshot(std::string_view job_id) const;

private:
    struct StoredJob {
        std::string job_id;
        JobProgressUpdate state;
        std::chrono::steady_clock::time_point started_at{};
        std::chrono::steady_clock::time_point finished_at{};
        bool running = false;
        bool ok = true;
        std::string error;
    };

    [[nodiscard]] JobProgressSnapshot buildSnapshot(const StoredJob& job) const;

    mutable std::mutex mutex_;
    std::string active_job_id_;
    StoredJob active_;
    StoredJob last_finished_;
};

}  // namespace vlm
