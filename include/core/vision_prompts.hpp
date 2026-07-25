#pragma once

#include <string>
#include <string_view>

namespace vlm::prompts {

// Frames + speech first, then the task — otherwise the model mixes them up.
// Do NOT call rkllm_set_chat_template: it disables input.enable_thinking (RKLLM warning).

inline constexpr std::string_view kFramesIntroRu = "Тебе даны кадры из видео <image>";
inline constexpr std::string_view kFramesIntroEng = "You are given frames from a video <image>";

inline constexpr std::string_view kSpeechPrefixRu = " с речью в видео: \"";
inline constexpr std::string_view kSpeechPrefixEng = " with speech in the video: \"";

inline constexpr std::string_view kTaskSimpleRu = "Опиши кратко и по делу видео.";
inline constexpr std::string_view kTaskSimpleEng = "Describe the video briefly and to the point.";

inline constexpr std::string_view kTaskDetailedRu =
    "Опиши это видео. Только факты, без выдумок. Ответ на русском:\n"
    "1) О чём видео?\n"
    "Его краткая суть.\n"
    "2) Текст в видео\n"
    "Только надписи в кадрах видео, не речь.\n"
    "3) Предположительный жанр";

inline constexpr std::string_view kTaskDetailedEng =
    "Describe this video. Factual only, no invention. Answer in English:\n"
    "1) What is the video about?\n"
    "Brief summary.\n"
    "2) On-screen text\n"
    "Only captions/labels visible in the video frames, not speech.\n"
    "3) Likely genre";

inline constexpr std::string_view kThinkOn = "\n/think";
inline constexpr std::string_view kThinkOff = "\n/no_think";

/** Build multimodal user prompt. prompt_mode: "simple" | "detailed".
 *  Optional ASR transcript is inserted with the frames, before the task. */
[[nodiscard]] inline std::string buildUserVisionPrompt(std::string_view lang_normalized,
                                                       bool enable_thinking,
                                                       std::string_view prompt_mode,
                                                       std::string_view transcript = {})
{
    const bool simple = (prompt_mode == "simple");
    const bool eng = (lang_normalized == "eng");

    std::string prompt;
    prompt.reserve(256 + transcript.size());
    prompt += eng ? kFramesIntroEng : kFramesIntroRu;
    if (!transcript.empty()) {
        prompt += eng ? kSpeechPrefixEng : kSpeechPrefixRu;
        prompt += transcript;
        prompt += '"';
    }
    prompt += '\n';

    if (simple) {
        prompt += eng ? kTaskSimpleEng : kTaskSimpleRu;
    } else {
        prompt += eng ? kTaskDetailedEng : kTaskDetailedRu;
    }
    prompt += enable_thinking ? kThinkOn : kThinkOff;
    return prompt;
}

}  // namespace vlm::prompts
