#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "types.hpp"

namespace vlm::prompts {

// Frames (+ optional speech) first, then the task — otherwise the model mixes them up.
// ASR text must be clearly marked as metadata (brackets), otherwise VLMs treat bare
// quotes next to <image> as on-screen captions.
// Thinking mode is controlled by RKLLMInput.enable_thinking, not /think|/no_think in the
// user prompt (those soft switches are echoed by the model and are redundant here).

inline constexpr std::string_view kFramesIntroRu = "Тебе даны кадры из видео <image>";
inline constexpr std::string_view kFramesIntroEng = "You are given frames from a video <image>";

inline constexpr std::string_view kFramesInterleavedIntroRu =
    "Тебе даны кадры из видео. Текст в [квадратных скобках] после кадра — транскрипт речи "
    "(ASR), это данные аудиодорожки, НЕ надписи на экране:";
inline constexpr std::string_view kFramesInterleavedIntroEng =
    "You are given frames from a video. Text in [square brackets] after a frame is the speech "
    "transcript (ASR) from the audio track — metadata, NOT on-screen captions:";

/** Prefix/suffix wrapping per-frame ASR speech: [речь: …] / [speech: …]. */
inline constexpr std::string_view kSpeechBracketOpenRu = " [речь: ";
inline constexpr std::string_view kSpeechBracketOpenEng = " [speech: ";
inline constexpr std::string_view kSpeechBracketClose = "]";

/** Flat (non-interleaved) transcript block after frames. */
inline constexpr std::string_view kSpeechPrefixRu = "\n[транскрипт речи (ASR): ";
inline constexpr std::string_view kSpeechPrefixEng = "\n[speech transcript (ASR): ";
inline constexpr std::string_view kSpeechSuffix = "]";

inline constexpr std::string_view kTaskSimpleRu = "Опиши кратко и по делу видео.";
inline constexpr std::string_view kTaskSimpleEng = "Describe the video briefly and to the point.";

inline constexpr std::string_view kTaskDetailedRu =
    "Опиши это видео. Только факты, без выдумок. Ответ на русском:\n"
    "1) О чём видео?\n"
    "Его краткая суть. Речь из [речь: …] можно учитывать как контекст, но не как надписи.\n"
    "2) Надписи в видео\n"
    "Только визуальные надписи/титулы, видимые в самих кадрах. "
    "Не копируй сюда текст из [речь: …] — это ASR, не графика.\n"
    "3) Предположительный жанр";

inline constexpr std::string_view kTaskDetailedEng =
    "Describe this video. Factual only, no invention. Answer in English:\n"
    "1) What is the video about?\n"
    "Brief summary. You may use [speech: …] as audio context, but not as on-screen text.\n"
    "2) On-screen text\n"
    "Only captions/labels/titles visible in the video frames themselves. "
    "Do not copy text from [speech: …] — that is ASR, not graphics.\n"
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
