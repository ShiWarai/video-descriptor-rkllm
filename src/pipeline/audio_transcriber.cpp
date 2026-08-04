#include "pipeline/audio_transcriber.hpp"

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/audio_extractor.hpp"

namespace vlm {

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct ParsedUrl {
    std::string host;
    int port = 80;
    bool use_ssl = false;
};

[[nodiscard]] bool parseBaseUrl(const std::string& url, ParsedUrl& out)
{
    std::string rest = url;
    if (rest.rfind("http://", 0) == 0) {
        rest = rest.substr(7);
        out.use_ssl = false;
        out.port = 80;
    } else if (rest.rfind("https://", 0) == 0) {
        rest = rest.substr(8);
        out.use_ssl = true;
        out.port = 443;
    } else {
        return false;
    }

    const auto slash = rest.find('/');
    if (slash != std::string::npos) {
        rest = rest.substr(0, slash);
    }
    if (rest.empty()) {
        return false;
    }

    const auto colon = rest.rfind(':');
    if (colon != std::string::npos && colon > 0) {
        out.host = rest.substr(0, colon);
        try {
            out.port = std::stoi(rest.substr(colon + 1));
        } catch (...) {
            return false;
        }
    } else {
        out.host = rest;
    }
    return !out.host.empty();
}

}  // namespace

bool parseWhisperTranscribeResponse(std::string_view body, TranscriptResult& result)
{
    const auto j = nlohmann::json::parse(body);
    result.text = j.value("text", "");
    result.segments.clear();
    if (j.contains("language") && j["language"].is_string()) {
        const auto lang = j["language"].get<std::string>();
        if (!lang.empty() && lang != "unknown") {
            result.language = lang;
        }
    }
    if (j.contains("segments") && j["segments"].is_array()) {
        for (const auto& seg : j["segments"]) {
            TranscriptSegment segment;
            segment.start = seg.value("start", 0.0);
            segment.end = seg.value("end", 0.0);
            segment.text = seg.value("text", "");
            if (!segment.text.empty()) {
                result.segments.push_back(std::move(segment));
            }
        }
    }
    return true;
}

TranscriptResult StubAudioTranscriber::transcribe(const std::filesystem::path& /*video_path*/,
                                                  const std::optional<std::string>& override_text,
                                                  std::string_view /*language*/)
{
    if (override_text && !override_text->empty()) {
        return TranscriptResult{.text = *override_text, .status = "provided"};
    }
    return TranscriptResult{.text = "", .status = "stub"};
}

HttpWhisperTranscriber::HttpWhisperTranscriber(std::string base_url, std::string api_key)
    : base_url_(std::move(base_url)), api_key_(std::move(api_key))
{
    while (!base_url_.empty() && base_url_.back() == '/') {
        base_url_.pop_back();
    }
}

TranscriptResult HttpWhisperTranscriber::transcribe(
    const std::filesystem::path& video_path, const std::optional<std::string>& override_text,
    std::string_view language)
{
    if (override_text && !override_text->empty()) {
        return TranscriptResult{.text = *override_text, .status = "provided"};
    }

    TranscriptResult result{.status = "error"};
    ParsedUrl parsed;
    if (!parseBaseUrl(base_url_, parsed)) {
        std::cerr << "whisper: invalid base URL: " << base_url_ << '\n';
        return result;
    }

    const auto t_extract = Clock::now();
    std::vector<uint8_t> wav_bytes;
    if (!extractAudioWav16kMono(video_path, wav_bytes)) {
        std::cerr << "whisper: audio extract failed for " << video_path << '\n';
        result.audio_extract_ms = elapsedMs(t_extract);
        return result;
    }
    result.audio_extract_ms = elapsedMs(t_extract);

    std::string audio_payload(reinterpret_cast<const char*>(wav_bytes.data()), wav_bytes.size());

    httplib::MultipartFormDataItems items;
    items.push_back(
        {.name = "file",
         .content = std::move(audio_payload),
         .filename = "audio.wav",
         .content_type = "audio/wav"});
    items.push_back({.name = "model", .content = "whisper-1", .filename = "", .content_type = ""});
    items.push_back(
        {.name = "response_format", .content = "verbose_json", .filename = "", .content_type = ""});
    if (!language.empty()) {
        items.push_back(
            {.name = "language", .content = std::string(language), .filename = "", .content_type = ""});
    }

    const auto t_whisper = Clock::now();

    httplib::Headers headers;
    if (!api_key_.empty()) {
        headers.emplace("Authorization", "Bearer " + api_key_);
    }

    constexpr const char* kTranscribePath = "/v1/audio/transcriptions";

    auto post_transcribe = [&](auto& client) -> httplib::Result {
        client.set_connection_timeout(30, 0);
        client.set_read_timeout(300, 0);
        client.set_write_timeout(30, 0);
        if (headers.empty()) {
            return client.Post(kTranscribePath, items);
        }
        return client.Post(kTranscribePath, headers, items);
    };

    httplib::Result response;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parsed.use_ssl) {
        httplib::SSLClient client(parsed.host, parsed.port);
        response = post_transcribe(client);
    } else {
        httplib::Client client(parsed.host, parsed.port);
        response = post_transcribe(client);
    }
#else
    if (parsed.use_ssl) {
        std::cerr << "whisper: HTTPS not supported (build without OpenSSL)\n";
        return result;
    }
    httplib::Client client(parsed.host, parsed.port);
    response = post_transcribe(client);
#endif
    result.whisper_ms = elapsedMs(t_whisper);

    if (!response) {
        std::cerr << "whisper: HTTP request failed to " << base_url_
                  << " error=" << static_cast<int>(response.error()) << '\n';
        return result;
    }
    if (response->status != 200) {
        std::cerr << "whisper: HTTP " << response->status << ": " << response->body << '\n';
        return result;
    }

    try {
        if (!parseWhisperTranscribeResponse(response->body, result)) {
            result.status = "error";
            return result;
        }
        result.status = "ok";
    } catch (const std::exception& e) {
        std::cerr << "whisper: invalid JSON response: " << e.what() << '\n';
        result.status = "error";
    }
    return result;
}

std::unique_ptr<AudioTranscriber> makeAudioTranscriber(const PipelineConfig& config)
{
    if (config.whisper_url.empty()) {
        return std::make_unique<StubAudioTranscriber>();
    }
    return std::make_unique<HttpWhisperTranscriber>(config.whisper_url, config.whisper_api_key);
}

}  // namespace vlm
