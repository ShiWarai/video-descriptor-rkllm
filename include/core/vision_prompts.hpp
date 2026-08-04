#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "types.hpp"

namespace vlm::prompts {

// Кадры (+ речевой трек) идут до задания — иначе модель их смешивает.
// Речь в <р>...</р>, чтобы кавычки рядом с <image> не воспринимались как подписи.
// Обдумывание включается через RKLLMInput.enable_thinking, а не /think|/no_think в промпте.

inline constexpr std::string_view kFramesIntroRu = "Тебе даны кадры из видео <image>";
inline constexpr std::string_view kFramesIntroEng = "You are given frames from a video <image>";

inline constexpr std::string_view kFramesInterleavedIntroRu =
    "Тебе даны кадры из видео (<р> </р> — речь, не надписи на кадрах):";
inline constexpr std::string_view kFramesInterleavedIntroEng =
    "You are given frames from a video (<s> </s> — speech, not on-screen text on frames):";

inline constexpr std::string_view kSpeechBracketOpenRu = " <р>";
inline constexpr std::string_view kSpeechBracketOpenEng = " <s>";
inline constexpr std::string_view kSpeechBracketClose = "</р>";
inline constexpr std::string_view kSpeechBracketCloseEng = "</s>";

inline constexpr std::string_view kSpeechPrefixRu = "\n<р>";
inline constexpr std::string_view kSpeechPrefixEng = "\n<s>";
inline constexpr std::string_view kSpeechSuffix = "</р>";
inline constexpr std::string_view kSpeechSuffixEng = "</s>";

inline constexpr std::string_view kTaskSimpleRu = "Опиши кратко и по делу видео.";
inline constexpr std::string_view kTaskSimpleEng = "Describe the video briefly and to the point.";

inline constexpr std::string_view kTaskDetailedRu =
    "Опиши это видео. Только факты, без выдумок. Все 3 пункта обязательны. Ответ на русском:\n"
    "1) О чём видео?\n"
    "Его краткая суть.\n"
    "2) Надписи в видео\n"
    "Только надписи в кадре. <р> не перечислять. Если нет надписей пиши «нет».\n"
    "3) Предположительный жанр";

inline constexpr std::string_view kTaskDetailedEng =
    "Describe this video. Factual only, no invention. All 3 sections required. Answer in English:\n"
    "1) What is the video about?\n"
    "Brief summary.\n"
    "2) On-screen text\n"
    "Only on-screen text/captions. Do not list <s> content. If none write \"none\".\n"
    "3) Likely genre";

/** Распределить сегменты whisper по временным интервалам между кадрами. */
[[nodiscard]] std::vector<std::string> assignSpeechToFrameIntervals(
    const std::vector<double>& frame_times, const std::vector<TranscriptSegment>& segments,
    double duration_sec);

/** Собрать промпт для модели: кадры (+речь) + задание. */
[[nodiscard]] std::string buildUserVisionPrompt(
    std::string_view lang_normalized, std::string_view prompt_mode,
    const std::vector<double>& frame_times = {},
    const std::vector<TranscriptSegment>& segments = {},
    std::string_view flat_transcript_fallback = {}, double duration_sec = 0.0);

/** Заменить один <image> на нужное количество тегов; если уже совпадает — вернуть без изменений. */
[[nodiscard]] std::string expandImageTags(const std::string& prompt, std::size_t image_count);

}  // namespace vlm::prompts
