#include "runtime/job_progress.hpp"

#include <algorithm>
#include <cmath>

namespace vlm {

namespace {

constexpr double kTypicalModelLoadSec = 10.0;
constexpr double kTypicalWhisperSec = 50.0;
constexpr double kTypicalVisionSec = 20.0;
constexpr double kTypicalLlmSec = 48.0;

[[nodiscard]] double clampPercent(double value) noexcept
{
    return std::clamp(value, 0.0, 100.0);
}

[[nodiscard]] double ratio(int done, int total) noexcept
{
    if (total <= 0 || done < 0) {
        return 0.0;
    }
    return std::clamp(static_cast<double>(done) / static_cast<double>(total), 0.0, 1.0);
}

[[nodiscard]] double timeRatio(double elapsed_sec, double typical_sec) noexcept
{
    if (typical_sec <= 0.0) {
        return 0.0;
    }
    return std::clamp(elapsed_sec / typical_sec, 0.0, 1.0);
}

void mergeUpdate(JobProgressUpdate& dst, const JobProgressUpdate& src)
{
    if (!src.stage.empty()) {
        dst.stage = src.stage;
    }
    if (src.frames_done >= 0) {
        dst.frames_done = src.frames_done;
    }
    if (src.frames_total >= 0) {
        dst.frames_total = src.frames_total;
    }
    if (src.vision_done >= 0) {
        dst.vision_done = src.vision_done;
    }
    if (src.vision_total >= 0) {
        dst.vision_total = src.vision_total;
    }
    if (src.generate_tokens >= 0) {
        dst.generate_tokens = src.generate_tokens;
    }
    if (src.max_new_tokens >= 0) {
        dst.max_new_tokens = src.max_new_tokens;
    }
    if (src.model_load_done) {
        dst.model_load_done = true;
    }
    if (src.transcript_done) {
        dst.transcript_done = true;
    }
}

}  // namespace

std::string jobStageLabel(std::string_view stage)
{
    if (stage == kJobStageQueued) {
        return "В очереди";
    }
    if (stage == kJobStageLoadingModel) {
        return "Загрузка модели";
    }
    if (stage == kJobStageExtractingFrames) {
        return "Извлечение кадров";
    }
    if (stage == kJobStageEncodingVision) {
        return "Vision encode";
    }
    if (stage == kJobStageTranscribing) {
        return "Whisper ASR";
    }
    if (stage == kJobStageGenerating) {
        return "LLM генерация";
    }
    if (stage == kJobStageDone) {
        return "Готово";
    }
    if (stage == kJobStageFailed) {
        return "Ошибка";
    }
    return std::string(stage);
}

double estimateJobProgressPercent(const JobProgressUpdate& state, double elapsed_sec) noexcept
{
    if (state.stage == kJobStageDone) {
        return 100.0;
    }
    if (state.stage == kJobStageFailed) {
        return 100.0;
    }
    if (state.stage == kJobStageQueued) {
        return 0.0;
    }

    if (state.stage == kJobStageLoadingModel) {
        const double t = state.model_load_done ? 1.0 : timeRatio(elapsed_sec, kTypicalModelLoadSec);
        return clampPercent(12.0 * t);
    }

    if (state.stage == kJobStageGenerating) {
        if (state.max_new_tokens > 0 && state.generate_tokens >= 0) {
            return clampPercent(50.0 + 49.0 * ratio(state.generate_tokens, state.max_new_tokens));
        }
        return clampPercent(50.0 + 49.0 * timeRatio(elapsed_sec, kTypicalLlmSec));
    }

    // Parallel prep: model/frames/vision vs whisper — take the furthest track (0..50%).
    double vision_track = 12.0;
    if (state.model_load_done) {
        vision_track = 20.0;
    }
    if (state.frames_total > 0) {
        vision_track = std::max(vision_track, 20.0 + 5.0 * ratio(state.frames_done, state.frames_total));
    }
    if (state.stage == kJobStageEncodingVision || state.vision_total > 0) {
        vision_track = std::max(vision_track, 25.0 + 25.0 * ratio(state.vision_done, state.vision_total));
    }

    double whisper_track = 12.0;
    if (state.stage == kJobStageTranscribing || state.transcript_done) {
        const double t = state.transcript_done ? 1.0 : timeRatio(elapsed_sec, kTypicalWhisperSec);
        whisper_track = 12.0 + 38.0 * t;
    }

    return clampPercent(std::max(vision_track, whisper_track));
}

nlohmann::json jobProgressToJson(const JobProgressSnapshot& snap)
{
    return nlohmann::json{{"job_id", snap.job_id},
                          {"status", snap.status},
                          {"stage", snap.stage},
                          {"stage_label", snap.stage_label},
                          {"elapsed_sec", snap.elapsed_sec},
                          {"progress_percent", snap.progress_percent},
                          {"details", snap.details}};
}

void JobProgressTracker::beginJob(std::string job_id)
{
    std::lock_guard lock(mutex_);
    active_job_id_ = std::move(job_id);
    active_ = StoredJob{};
    active_.job_id = active_job_id_;
    active_.running = true;
    active_.started_at = std::chrono::steady_clock::now();
    active_.state.stage = std::string(kJobStageQueued);
}

void JobProgressTracker::updateJob(std::string_view job_id, const JobProgressUpdate& update)
{
    std::lock_guard lock(mutex_);
    if (!active_.running || active_job_id_ != job_id) {
        return;
    }
    mergeUpdate(active_.state, update);
}

void JobProgressTracker::finishJob(std::string_view job_id, bool ok, std::string_view error)
{
    std::lock_guard lock(mutex_);
    if (!active_.running || active_job_id_ != job_id) {
        return;
    }
    active_.running = false;
    active_.ok = ok;
    active_.error = error;
    active_.state.stage = ok ? std::string(kJobStageDone) : std::string(kJobStageFailed);
    active_.finished_at = std::chrono::steady_clock::now();
    last_finished_ = active_;
}

std::optional<JobProgressSnapshot> JobProgressTracker::snapshot(std::string_view job_id) const
{
    std::lock_guard lock(mutex_);
    if (active_.job_id == job_id && !active_.state.stage.empty()) {
        return buildSnapshot(active_);
    }
    if (last_finished_.job_id == job_id && !last_finished_.state.stage.empty()) {
        return buildSnapshot(last_finished_);
    }
    return std::nullopt;
}

JobProgressSnapshot JobProgressTracker::buildSnapshot(const StoredJob& job) const
{
    const auto now = std::chrono::steady_clock::now();
    const auto end = job.running ? now : job.finished_at;
    const double elapsed =
        std::chrono::duration<double>(end - job.started_at).count();

    JobProgressSnapshot snap;
    snap.job_id = job.job_id;
    snap.stage = job.state.stage;
    snap.stage_label = jobStageLabel(snap.stage);
    snap.elapsed_sec = elapsed;
    snap.progress_percent = estimateJobProgressPercent(job.state, elapsed);
    snap.status = job.running ? "running" : (job.ok ? "done" : "failed");

    snap.details = {
        {"model_load_done", job.state.model_load_done},
        {"transcript_done", job.state.transcript_done},
    };
    if (job.state.frames_total > 0) {
        snap.details["frames_done"] = job.state.frames_done;
        snap.details["frames_total"] = job.state.frames_total;
    }
    if (job.state.vision_total > 0) {
        snap.details["vision_done"] = job.state.vision_done;
        snap.details["vision_total"] = job.state.vision_total;
    }
    if (job.state.max_new_tokens > 0) {
        snap.details["generate_tokens"] = job.state.generate_tokens;
        snap.details["max_new_tokens"] = job.state.max_new_tokens;
    }
    if (!job.ok && !job.error.empty()) {
        snap.details["error"] = job.error;
    }
    return snap;
}

}  // namespace vlm
