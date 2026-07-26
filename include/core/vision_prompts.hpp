#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "types.hpp"

namespace vlm::prompts {

// Frames (+ optional speech) first, then the task — otherwise the model mixes them up.
// Thinking mode is controlled by RKLLMInput.enable_thinking, not /think|/no_think in the
// user prompt (those soft switches are echoed by the model and are redundant here).

inline constexpr std::string_view kFramesIntroRu = "Тебе даны кадры из видео <image>";
inline constexpr std::string_view kFramesIntroEng = "You are given frames from a video <image>";

inline constexpr std::string_view kFramesInterleavedIntroRu =
    "Тебе даны кадры из видео. Между кадрами указана речь за этот промежуток:";
inline constexpr std::string_view kFramesInterleavedIntroEng =
    "You are given frames from a video. Speech between frames is shown for each interval:";

inline constexpr std::string_view kSpeechPrefixRu = " с речью в видео: \"";
inline constexpr std::string_view kSpeechPrefixEng = " with speech in the video: \"";

inline constexpr std::string_view kSpeechBetweenRu = "Речь: \"";
inline constexpr std::string_view kSpeechBetweenEng = "Speech: \"";

inline constexpr std::string_view kTaskSimpleRu = "Опиши кратко и по делу видео.";
inline constexpr std::string_view kTaskSimpleEng = "Describe the video briefly and to the point.";

inline constexpr std::string_view kTaskDetailedRu =
    "Опиши это видео. Только факты, без выдумок. Ответ на русском:\n"
    "1) О чём видео?\n"
    "Его краткая суть.\n"
    "2) Надписи в видео\n"
    "Только надписи в кадрах видео, не речь и не субтитры.\n"
    "3) Предположительный жанр";

inline constexpr std::string_view kTaskDetailedEng =
    "Describe this video. Factual only, no invention. Answer in English:\n"
    "1) What is the video about?\n"
    "Brief summary.\n"
    "2) On-screen text\n"
    "Only captions/labels visible in the video frames, not speech and not subtitles.\n"
    "3) Likely genre";

/**
 * Assign whisper segments to speech after each frame timestamp.
 * Interval i is [frame_times[i], frame_times[i+1]) or, for the last frame,
 * [frame_times[i], duration_end]. A segment is placed when its midpoint lies in the interval.
 */
[[nodiscard]] std::vector<std::string> assignSpeechToFrameIntervals(
    const std::vector<double>& frame_times, const std::vector<TranscriptSegment>& segments,
    double duration_sec);

/** Build multimodal user prompt. prompt_mode: "simple" | "detailed". */
[[nodiscard]] std::string buildUserVisionPrompt(
    std::string_view lang_normalized, std::string_view prompt_mode,
    const std::vector<double>& frame_times = {},
    const std::vector<TranscriptSegment>& segments = {},
    std::string_view flat_transcript_fallback = {}, double duration_sec = 0.0);

/** Expand a single <image> tag to `image_count` tags; leave prompt unchanged if already correct. */
[[nodiscard]] std::string expandImageTags(const std::string& prompt, std::size_t image_count);

}  // namespace vlm::prompts
