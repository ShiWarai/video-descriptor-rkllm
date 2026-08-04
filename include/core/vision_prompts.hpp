#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "types.hpp"

namespace vlm::prompts {

// Frames (+ optional speech) first, then the task — otherwise the model mixes them up.
// Speech is wrapped as [речь: ...] so bare quotes next to <image> are not taken as captions.
// Thinking: RKLLMInput.enable_thinking, not /think|/no_think in the user prompt.

inline constexpr std::string_view kFramesIntroRu = "Тебе даны кадры из видео <image>";
inline constexpr std::string_view kFramesIntroEng = "You are given frames from a video <image>";

inline constexpr std::string_view kFramesInterleavedIntroRu =
    "Тебе даны кадры из видео ([речь: ...] — аудио, не надписи):";
inline constexpr std::string_view kFramesInterleavedIntroEng =
    "You are given frames from a video ([speech: ...] is audio, not on-screen text):";

inline constexpr std::string_view kSpeechBracketOpenRu = " [речь: ";
inline constexpr std::string_view kSpeechBracketOpenEng = " [speech: ";
inline constexpr std::string_view kSpeechBracketClose = "]";

inline constexpr std::string_view kSpeechPrefixRu = "\n[речь: ";
inline constexpr std::string_view kSpeechPrefixEng = "\n[speech: ";
inline constexpr std::string_view kSpeechSuffix = "]";

inline constexpr std::string_view kTaskSimpleRu = "Опиши кратко и по делу видео.";
inline constexpr std::string_view kTaskSimpleEng = "Describe the video briefly and to the point.";

inline constexpr std::string_view kTaskDetailedRu =
    "Опиши это видео. Только факты, без выдумок. Ответ на русском:\n"
    "1) О чём видео?\n"
    "Его краткая суть.\n"
    "2) Надписи в видео\n"
    "Только текст, видимый в кадрах. [речь: ...] игнорируй. Если надписей нет — «нет».\n"
    "3) Предположительный жанр";

inline constexpr std::string_view kTaskDetailedEng =
    "Describe this video. Factual only, no invention. Answer in English:\n"
    "1) What is the video about?\n"
    "Brief summary.\n"
    "2) On-screen text\n"
    "Only text visible in the frames. Ignore [speech: ...]. If none — write \"none\".\n"
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
