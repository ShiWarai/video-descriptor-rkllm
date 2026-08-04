#include "pipeline/video_context_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "core/frame_extractor.hpp"
#include "core/media_util.hpp"
#include "core/system_memory.hpp"
#include "core/llm_runtime.hpp"
#include "core/text_util.hpp"
#include "core/vision_encoder.hpp"
#include "runtime/job_progress.hpp"

namespace vlm {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

VideoContextPipeline::VideoContextPipeline(ModelRegistry registry, PipelineConfig config,
                                           std::unique_ptr<StageTransport> transport)
    : registry_(std::move(registry)),
      config_(std::move(config)),
      transport_(transport ? std::move(transport)
                           : makeStageTransport(registry_, config_, false)),
      transcriber_(makeAudioTranscriber(config_))
{
}

bool VideoContextPipeline::initialize(std::optional<std::string_view> preload_model_id)
{
    if (registry_.models().empty()) {
        std::cerr << "Model registry is empty\n";
        return false;
    }

    if (preload_model_id && !preload_model_id->empty() && !transport_->distributed()) {
        if (!ensureModel(*preload_model_id)) {
            std::cerr << "Failed to preload model: " << *preload_model_id << '\n';
            return false;
        }
        if (config_.verbose) {
            std::cerr << "Preloaded model: " << loadedModelId() << '\n';
        }
    }

    ready_ = true;
    return true;
}

bool VideoContextPipeline::ensureModel(std::string_view model_id, std::string* error_out)
{
    return transport_->loadModel(model_id, error_out);
}

int VideoContextPipeline::planVisionWorkers(std::string_view model_id,
                                            std::string* error_out) const
{
    return transport_->planVisionWorkers(model_id, error_out);
}

void VideoContextPipeline::releaseModels()
{
    transport_->releaseModels();
}

int VideoContextPipeline::resolveFrameBudget(const AnalyzeRequest& request) const
{
    const int cap = request.frame_budget > 0 ? request.frame_budget : config_.frame_budget;
    return transport_->resolveFrameBudget(config_, cap);
}

int VideoContextPipeline::effectiveMaxFrames(const AnalyzeRequest& request) const
{
    const int budget = resolveFrameBudget(request);
    int max_frames = request.max_frames;
    if (config_.absolute_max_frames > 0) {
        max_frames = std::min(max_frames, config_.absolute_max_frames);
    }
    return std::min(max_frames, budget);
}

AnalyzeResult VideoContextPipeline::analyze(const AnalyzeRequest& request)
{
    AnalyzeResult result;
    result.job_id = request.job_id;
    const auto t_total = Clock::now();
    auto report = [&](const JobProgressUpdate& update) {
        if (request.on_progress) {
            request.on_progress(update);
        }
    };
    auto finish = [&]() -> AnalyzeResult& {
        transport_->clearVision();
        result.metrics["total_ms"] = elapsedMs(t_total);
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        return result;
    };

    if (!ready_) {
        result.error = "Pipeline not initialized";
        report({.stage = std::string(kJobStageFailed)});
        return finish();
    }

    const std::string model_id =
        request.model.empty() ? registry_.defaultModelId() : request.model;

    const bool thinking =
        request.enable_thinking.value_or(config_.enable_thinking);
    std::cerr << "analyze: enable_thinking=" << (thinking ? "true" : "false")
              << " lang=" << request.lang << " prompt_mode=" << request.prompt_mode
              << " max_tokens=" << (request.max_tokens > 0 ? request.max_tokens
                                                           : config_.default_max_tokens)
              << (request.max_tokens > 0 ? " (request)" : " (config)")
              << " temperature="
              << (request.temperature ? *request.temperature : config_.temperature)
              << (request.temperature ? " (request)" : " (config)")
              << " runtime=" << (transport_->distributed() ? "distributed" : "local") << '\n';

    const int budget = resolveFrameBudget(request);
    const int effective = effectiveMaxFrames(request);
    result.frames_requested = request.max_frames;
    result.frame_budget = budget;
    result.frames_capped_by_context = effective < request.max_frames;

    {
        std::string ram_reason;
        if (planVisionWorkers(model_id, &ram_reason) <= 0) {
            std::cerr << "Refusing analyze for " << model_id << ": " << ram_reason << '\n';
            result.error = ram_reason;
            report({.stage = std::string(kJobStageFailed)});
            return finish();
        }
    }

    report({.stage = std::string(kJobStageLoadingModel), .frames_total = effective});

    struct PrepState {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<PendingVisionFrame> pending_frames;
        bool extract_done = false;
        int extract_count = 0;
        VideoInfo video_info{};
        std::atomic<bool> model_load_finished{false};
        std::atomic<bool> encode_abort{false};
        bool model_ok = false;
        std::string load_error;
        double model_load_ms = 0;
        double extract_ms = 0;
        double transcript_ms = 0;
        TranscriptResult transcript;
    } prep;

    if (config_.verbose) {
        if (isGifPath(request.video_path.string())) {
            std::cerr << "parallel prep: model load + frame extract (ASR skipped: GIF)\n";
        } else {
            std::cerr << "parallel prep: model load + frame extract + transcript\n";
        }
    }

    const auto t_prep = Clock::now();
    const bool skip_asr = isGifPath(request.video_path.string());
    std::future<void> transcript_fut;
    if (skip_asr) {
        prep.transcript = TranscriptResult{.text = "", .status = "skipped"};
        prep.transcript_ms = 0;
    } else {
        transcript_fut = std::async(std::launch::async, [&] {
            report({.stage = std::string(kJobStageTranscribing)});
            const auto t0 = Clock::now();
            prep.transcript = transcriber_->transcribe(
                request.video_path, request.transcript_override, request.lang);
            prep.transcript_ms = elapsedMs(t0);
            report({.stage = std::string(kJobStageTranscribing), .transcript_done = true});
        });
    }

    std::future<bool> model_fut;
    if (!transport_->distributed()) {
        model_fut = std::async(std::launch::async, [&] {
            const auto t0 = Clock::now();
            std::string load_error;
            report({.stage = std::string(kJobStageLoadingModel)});
            const bool ok = ensureModel(model_id, &load_error);
            prep.model_load_ms = elapsedMs(t0);
            prep.model_ok = ok;
            prep.model_load_finished = true;
            if (ok) {
                report({.stage = std::string(kJobStageEncodingVision),
                        .vision_total = effective,
                        .model_load_done = true});
            }
            if (!ok && !load_error.empty()) {
                std::lock_guard lock(prep.mu);
                prep.load_error = std::move(load_error);
            }
            prep.cv.notify_all();
            return ok;
        });
    } else {
        prep.model_ok = true;
        prep.model_load_finished = true;
        prep.model_load_ms = 0;
    }

    FrameProgressCallback extract_progress;
    extract_progress = [&](int cur, int total) {
        if (config_.verbose) {
            std::cerr << "\rFrame extract: " << cur << '/' << total;
            if (cur == total) {
                std::cerr << '\n';
            }
        }
        report({.stage = std::string(kJobStageExtractingFrames),
                .frames_done = cur,
                .frames_total = total,
                .model_load_done = prep.model_load_finished.load() && prep.model_ok});
    };

    auto extract_fut = std::async(std::launch::async, [&] {
        const auto t0 = Clock::now();
        VideoInfo info;
        const int got = extractor_.extractFramesStreaming(
            request.video_path.string(), effective,
            [&](RgbFrame frame, int index, int /*total*/, double time_sec) {
                std::lock_guard lock(prep.mu);
                prep.pending_frames.push_back(
                    PendingVisionFrame{.index = index,
                                       .time_sec = time_sec,
                                       .frame = std::move(frame)});
                ++prep.extract_count;
                prep.cv.notify_all();
            },
            FrameExtractor::kDefaultVisionSize, FrameExtractor::kDefaultVisionSize,
            extract_progress, &info);
        {
            std::lock_guard lock(prep.mu);
            prep.video_info = info;
            prep.extract_done = true;
            prep.cv.notify_all();
        }
        prep.extract_ms = elapsedMs(t0);
        return got;
    });

    VisionProgressCallback vision_progress = [&](int cur, int total) {
        if (config_.verbose) {
            std::cerr << "\rVision encode: " << cur << '/' << total;
            if (cur == total) {
                std::cerr << '\n';
            }
        }
        report({.stage = std::string(kJobStageEncodingVision),
                .vision_done = cur,
                .vision_total = total,
                .model_load_done = true});
    };

    if (!transport_->distributed()) {
        std::unique_lock lock(prep.mu);
        prep.cv.wait(lock, [&] { return prep.model_load_finished.load(); });
    }

    const auto t_encode = Clock::now();
    VisionEncodeResult vision_result;
    int encoded = 0;

    if (transport_->distributed()) {
        const int extracted = extract_fut.get();
        if (transcript_fut.valid()) {
            transcript_fut.get();
        }
        prep.extract_done = true;

        std::vector<PendingVisionFrame> frames;
        {
            std::lock_guard lock(prep.mu);
            frames.assign(prep.pending_frames.begin(), prep.pending_frames.end());
        }

        if (extracted == 0 || frames.empty()) {
            result.metrics["frame_extract_ms"] = prep.extract_ms;
            result.metrics["image_prep_ms"] = elapsedMs(t_prep);
            result.metrics["transcript_ms"] = prep.transcript_ms;
            result.metrics["audio_extract_ms"] = prep.transcript.audio_extract_ms;
            result.metrics["whisper_ms"] = prep.transcript.whisper_ms;
            result.transcript = std::move(prep.transcript);
            result.duration_sec = prep.video_info.duration_sec;
            result.error = "No frames extracted from video";
            return finish();
        }

        const bool has_timed_speech =
            (prep.transcript.status == "ok") && !prep.transcript.segments.empty();
        const std::string_view flat_transcript =
            (!has_timed_speech &&
             (prep.transcript.status == "ok" || prep.transcript.status == "provided"))
                ? std::string_view(prep.transcript.text)
                : std::string_view{};
        std::vector<double> frame_times;
        frame_times.reserve(frames.size());
        for (const auto& frame : frames) {
            frame_times.push_back(frame.time_sec);
        }
        static const std::vector<TranscriptSegment> kEmptySegments;
        const std::vector<TranscriptSegment>& segments =
            has_timed_speech ? prep.transcript.segments : kEmptySegments;
        const std::string prompt = LlmRuntime::buildUserVisionPrompt(
            request.lang, request.prompt_mode, frame_times, segments, flat_transcript,
            prep.video_info.duration_sec);

        EncodeThenGenerateParams params{
            .job_id = request.job_id,
            .model_id = model_id,
            .prompt = prompt,
            .max_new_tokens =
                request.max_tokens > 0 ? request.max_tokens : config_.default_max_tokens,
            .temperature = request.temperature.value_or(-1.0f),
            .enable_thinking = thinking,
            .lang = request.lang,
        };

        report({.stage = std::string(kJobStageEncodingVision),
                .vision_total = static_cast<int>(frames.size())});
        const auto etg = transport_->encodeThenGenerate(frames, params, vision_progress);
        if (!etg.ok()) {
            result.metrics["frame_extract_ms"] = prep.extract_ms;
            result.metrics["vision_encode_ms"] = etg.encode_ms;
            result.metrics["image_prep_ms"] = elapsedMs(t_prep);
            result.metrics["transcript_ms"] = prep.transcript_ms;
            result.metrics["audio_extract_ms"] = prep.transcript.audio_extract_ms;
            result.metrics["whisper_ms"] = prep.transcript.whisper_ms;
            result.transcript = std::move(prep.transcript);
            result.duration_sec = prep.video_info.duration_sec;
            result.error = etg.error.empty() ? "Distributed inference failed" : etg.error;
            return finish();
        }

        encoded = etg.n_image;
        result.description = stripThinkingTags(etg.text);
        result.frames_used = encoded;
        result.model = model_id;
        result.duration_sec = prep.video_info.duration_sec;
        result.transcript = std::move(prep.transcript);
        result.metrics["frame_extract_ms"] = prep.extract_ms;
        result.metrics["vision_encode_ms"] = etg.encode_ms;
        result.metrics["llm_generate_ms"] = etg.generate_ms;
        result.metrics["image_prep_ms"] = elapsedMs(t_prep);
        result.metrics["transcript_ms"] = prep.transcript_ms;
        result.metrics["audio_extract_ms"] = result.transcript.audio_extract_ms;
        result.metrics["whisper_ms"] = result.transcript.whisper_ms;
        result.metrics["max_new_tokens"] = etg.max_new_tokens;
        result.metrics["generate_tokens"] = etg.generate_tokens;
        if (etg.truncated) {
            result.metrics["truncated"] = 1;
        }
        report({.stage = std::string(kJobStageGenerating),
                .generate_tokens = etg.generate_tokens,
                .max_new_tokens = etg.max_new_tokens,
                .model_load_done = true,
                .transcript_done = true});
        return finish();
    }

    if (prep.model_ok) {
        transport_->clearVision();
        VisionEncodeQueue queue{prep.mu, prep.cv, prep.pending_frames, prep.extract_done,
                                prep.encode_abort};
        prep.cv.notify_all();
        vision_result = transport_->encodeStreaming(queue, effective, vision_progress);
        encoded = vision_result.n_image;
    } else {
        prep.encode_abort = true;
        prep.cv.notify_all();
    }

    const int extracted = extract_fut.get();
    const bool model_ok = model_fut.get();

    result.metrics["model_load_ms"] = prep.model_load_ms;
    result.metrics["frame_extract_ms"] = prep.extract_ms;
    result.metrics["vision_encode_ms"] = elapsedMs(t_encode);
    result.metrics["image_prep_ms"] = elapsedMs(t_prep);
    result.model = loadedModelId();
    result.duration_sec = prep.video_info.duration_sec;

    if (!model_ok) {
        if (transcript_fut.valid()) {
            transcript_fut.get();
        }
        result.metrics["transcript_ms"] = prep.transcript_ms;
        result.metrics["audio_extract_ms"] = prep.transcript.audio_extract_ms;
        result.metrics["whisper_ms"] = prep.transcript.whisper_ms;
        result.transcript = std::move(prep.transcript);
        result.error = prep.load_error.empty() ? ("Failed to load model: " + model_id)
                                               : prep.load_error;
        return finish();
    }

    if (config_.verbose) {
        const VideoInfo& info = prep.video_info;
        const int planned = planFrameCount(info.duration_sec, info.fps, effective);
        const auto times = planFrameTimes(info.duration_sec, info.fps, effective);
        std::cerr << "frame sampling: duration=" << info.duration_sec << "s fps=" << info.fps
                  << " requested=" << effective << " planned=" << planned
                  << " got=" << extracted << " encoded=" << encoded << '\n';
        if (!times.empty()) {
            std::cerr << "  times_sec:";
            for (double t : times) {
                std::cerr << ' ' << t;
            }
            std::cerr << '\n';
        }
    }

    if (encoded == 0 || !vision_result.ok()) {
        if (transcript_fut.valid()) {
            transcript_fut.get();
        }
        result.metrics["transcript_ms"] = prep.transcript_ms;
        result.metrics["audio_extract_ms"] = prep.transcript.audio_extract_ms;
        result.metrics["whisper_ms"] = prep.transcript.whisper_ms;
        result.transcript = std::move(prep.transcript);
        result.error = extracted == 0 ? "No frames extracted from video"
                                      : (vision_result.error.empty() ? "Vision encoding failed"
                                                                     : vision_result.error);
        return finish();
    }

    result.frames_used = encoded;

    if (transcript_fut.valid()) {
        transcript_fut.get();
    }
    result.metrics["transcript_ms"] = prep.transcript_ms;
    result.metrics["audio_extract_ms"] = prep.transcript.audio_extract_ms;
    result.metrics["whisper_ms"] = prep.transcript.whisper_ms;
    result.transcript = std::move(prep.transcript);

    const bool has_timed_speech =
        (result.transcript.status == "ok") && !result.transcript.segments.empty();
    const std::string_view flat_transcript =
        (!has_timed_speech &&
         (result.transcript.status == "ok" || result.transcript.status == "provided"))
            ? std::string_view(result.transcript.text)
            : std::string_view{};
    static const std::vector<TranscriptSegment> kEmptySegments;
    const std::vector<TranscriptSegment>& segments =
        has_timed_speech ? result.transcript.segments : kEmptySegments;
    const std::string prompt = LlmRuntime::buildUserVisionPrompt(
        request.lang, request.prompt_mode, vision_result.frame_times, segments, flat_transcript,
        result.duration_sec);
    const float temperature = request.temperature.value_or(-1.0f);
    const int max_new_tokens =
        request.max_tokens > 0 ? request.max_tokens : config_.default_max_tokens;
    report({.stage = std::string(kJobStageGenerating),
            .generate_tokens = 0,
            .max_new_tokens = max_new_tokens,
            .model_load_done = true,
            .transcript_done = skip_asr || prep.transcript.status == "ok" ||
                               prep.transcript.status == "skipped" ||
                               prep.transcript.status == "provided"});
    {
        const auto t0 = Clock::now();
        auto gen = transport_->generate(prompt, vision_result, request.max_tokens, temperature,
                                        thinking, request.lang);
        result.description = stripThinkingTags(gen.text);
        result.metrics["llm_generate_ms"] = elapsedMs(t0);
        result.metrics["max_new_tokens"] = gen.max_new_tokens;
        result.metrics["generate_tokens"] = gen.generate_tokens;
        report({.stage = std::string(kJobStageGenerating),
                .generate_tokens = gen.generate_tokens,
                .max_new_tokens = gen.max_new_tokens,
                .model_load_done = true,
                .transcript_done = true});
        if (gen.truncated) {
            result.metrics["truncated"] = 1;
        }
        transport_->clearVision();
    }
    return finish();
}

}  // namespace vlm
