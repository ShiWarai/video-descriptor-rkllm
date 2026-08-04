#include "core/vision_prompts.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace vlm::prompts {

namespace {

[[nodiscard]] std::string_view taskForMode(std::string_view prompt_mode, bool eng)
{
    const bool simple = (prompt_mode == "simple");
    if (eng) {
        return simple ? kTaskSimpleEng : kTaskDetailedEng;
    }
    return simple ? kTaskSimpleRu : kTaskDetailedRu;
}

[[nodiscard]] double intervalEnd(std::size_t frame_index, const std::vector<double>& frame_times,
                                 double duration_sec,
                                 const std::vector<TranscriptSegment>& segments)
{
    if (frame_index + 1 < frame_times.size()) {
        return frame_times[frame_index + 1];
    }

    double end = frame_times[frame_index];
    if (duration_sec > 0.0) {
        end = std::max(end, duration_sec);
    }
    for (const auto& segment : segments) {
        end = std::max(end, segment.end);
    }
    return end;
}

[[nodiscard]] std::string joinTexts(const std::vector<std::string>& parts)
{
    std::string out;
    for (const auto& part : parts) {
        if (part.empty()) {
            continue;
        }
        if (!out.empty()) {
            out += ' ';
        }
        out += part;
    }
    return out;
}

}  // namespace

std::vector<std::string> assignSpeechToFrameIntervals(
    const std::vector<double>& frame_times, const std::vector<TranscriptSegment>& segments,
    double duration_sec)
{
    std::vector<std::string> speech_after_frames(frame_times.size());
    if (frame_times.empty() || segments.empty()) {
        return speech_after_frames;
    }

    for (std::size_t i = 0; i < frame_times.size(); ++i) {
        const double interval_start = frame_times[i];
        const double interval_end = intervalEnd(i, frame_times, duration_sec, segments);
        const bool last_interval = (i + 1 >= frame_times.size());

        std::vector<std::string> texts;
        for (const auto& segment : segments) {
            if (segment.text.empty()) {
                continue;
            }
            const double midpoint = 0.5 * (segment.start + segment.end);
            const bool in_interval = last_interval ? (midpoint >= interval_start &&
                                                      midpoint <= interval_end)
                                                   : (midpoint >= interval_start &&
                                                      midpoint < interval_end);
            if (in_interval) {
                texts.push_back(segment.text);
            }
        }
        speech_after_frames[i] = joinTexts(texts);
    }

    return speech_after_frames;
}

std::string buildUserVisionPrompt(std::string_view lang_normalized, std::string_view prompt_mode,
                                  const std::vector<double>& frame_times,
                                  const std::vector<TranscriptSegment>& segments,
                                  std::string_view flat_transcript_fallback, double duration_sec)
{
    const bool eng = (lang_normalized == "eng");
    const std::string_view task = taskForMode(prompt_mode, eng);

    if (!frame_times.empty() && !segments.empty()) {
        const auto speech = assignSpeechToFrameIntervals(frame_times, segments, duration_sec);
        std::string prompt;
        prompt.reserve(512);
        prompt += eng ? kFramesInterleavedIntroEng : kFramesInterleavedIntroRu;
        prompt += '\n';

        for (std::size_t i = 0; i < frame_times.size(); ++i) {
            prompt += "<image>";
            if (!speech[i].empty()) {
                prompt += eng ? kSpeechBracketOpenEng : kSpeechBracketOpenRu;
                prompt += speech[i];
                prompt += kSpeechBracketClose;
            }
            prompt += '\n';
        }

        prompt += task;
        return prompt;
    }

    std::string prompt;
    prompt.reserve(256 + flat_transcript_fallback.size());
    prompt += eng ? kFramesIntroEng : kFramesIntroRu;
    if (!flat_transcript_fallback.empty()) {
        prompt += eng ? kSpeechPrefixEng : kSpeechPrefixRu;
        prompt += flat_transcript_fallback;
        prompt += kSpeechSuffix;
    }
    prompt += '\n';
    prompt += task;
    return prompt;
}

std::string expandImageTags(const std::string& prompt, std::size_t image_count)
{
    if (image_count == 0) {
        return prompt;
    }

    std::size_t tag_count = 0;
    std::size_t search_from = 0;
    while (true) {
        const auto pos = prompt.find("<image>", search_from);
        if (pos == std::string::npos) {
            break;
        }
        ++tag_count;
        search_from = pos + 7;
    }

    if (tag_count == image_count) {
        return prompt;
    }
    if (tag_count != 1 || image_count <= 1) {
        return prompt;
    }

    std::string out = prompt;
    const auto pos = out.find("<image>");
    if (pos == std::string::npos) {
        return out;
    }
    std::string tags;
    tags.reserve(image_count * 8);
    for (std::size_t i = 0; i < image_count; ++i) {
        if (i) {
            tags += ' ';
        }
        tags += "<image>";
    }
    out.replace(pos, 7, tags);
    return out;
}

}  // namespace vlm::prompts
