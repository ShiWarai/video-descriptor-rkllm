#include <iostream>

#include "runtime/job_progress.hpp"

namespace {

void expect(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_progress_stages()
{
    expect(vlm::estimateJobProgressPercent({.stage = std::string(vlm::kJobStageQueued)}, 0.0) ==
               0.0,
           "queued 0%");
    expect(vlm::estimateJobProgressPercent(
               {.stage = std::string(vlm::kJobStageLoadingModel), .model_load_done = true}, 5.0) >=
               10.0,
           "model loaded");
    expect(vlm::estimateJobProgressPercent(
               {.stage = std::string(vlm::kJobStageEncodingVision),
                .vision_done = 8,
                .vision_total = 16,
                .model_load_done = true},
               10.0) >= 35.0,
           "vision half");
    expect(vlm::estimateJobProgressPercent(
               {.stage = std::string(vlm::kJobStageGenerating),
                .generate_tokens = 512,
                .max_new_tokens = 1024},
               10.0) >= 70.0,
           "llm half tokens");
    expect(vlm::estimateJobProgressPercent({.stage = std::string(vlm::kJobStageDone)}, 1.0) ==
               100.0,
           "done 100%");
}

void test_tracker_lifecycle()
{
    vlm::JobProgressTracker tracker;
    tracker.beginJob("job-test-1");
    tracker.updateJob("job-test-1",
                      {.stage = std::string(vlm::kJobStageEncodingVision),
                       .vision_done = 4,
                       .vision_total = 8,
                       .model_load_done = true});
    const auto running = tracker.snapshot("job-test-1");
    expect(running.has_value(), "running snapshot");
    expect(running->status == "running", "status running");
    expect(running->stage == vlm::kJobStageEncodingVision, "vision stage");
    expect(running->progress_percent > 0.0, "percent > 0");

    tracker.finishJob("job-test-1", true);
    const auto done = tracker.snapshot("job-test-1");
    expect(done.has_value(), "done snapshot");
    expect(done->status == "done", "status done");
    expect(done->progress_percent == 100.0, "done percent");

    expect(!tracker.snapshot("job-other").has_value(), "unknown job");
}

}  // namespace

int main()
{
    test_progress_stages();
    test_tracker_lifecycle();
    std::cout << "test_job_progress: ok\n";
    return 0;
}
