#include "pipeline/audio_transcriber.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

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

[[nodiscard]] std::string ffmpegPath(const std::string& bin_path, std::string_view name)
{
    return bin_path + "/" + std::string(name);
}

[[nodiscard]] bool runCommand(const std::string& cmd)
{
    return std::system(cmd.c_str()) == 0;
}

[[nodiscard]] std::filesystem::path makeTempAudioPath(const std::filesystem::path& workdir)
{
    std::error_code ec;
    std::filesystem::create_directories(workdir, ec);
    const auto stamp =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count();
    std::ostringstream oss;
    oss << "whisper_" << stamp << ".ogg";
    return workdir / oss.str();
}

[[nodiscard]] bool readFileBytes(const std::filesystem::path& path, std::string& out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

}  // namespace

TranscriptResult StubAudioTranscriber::transcribe(const std::filesystem::path& /*video_path*/,
                                                  const std::optional<std::string>& override_text)
{
    if (override_text && !override_text->empty()) {
        return TranscriptResult{.text = *override_text, .status = "provided"};
    }
    return TranscriptResult{.text = "", .status = "stub"};
}

HttpWhisperTranscriber::HttpWhisperTranscriber(std::string base_url, std::string ffmpeg_bin_path,
                                               std::string workdir)
    : base_url_(std::move(base_url)),
      ffmpeg_bin_path_(std::move(ffmpeg_bin_path)),
      workdir_(std::move(workdir))
{
    while (!base_url_.empty() && base_url_.back() == '/') {
        base_url_.pop_back();
    }
}

TranscriptResult HttpWhisperTranscriber::transcribe(
    const std::filesystem::path& video_path, const std::optional<std::string>& override_text)
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

    // mono 16 kHz ogg/opus — compact for whisper 25 MB upload limit
    const auto audio_path = makeTempAudioPath(workdir_);
    const auto t_extract = Clock::now();
    const std::string extract_cmd =
        ffmpegPath(ffmpeg_bin_path_, "ffmpeg") + " -y -v error -i \"" + video_path.string() +
        "\" -vn -ac 1 -ar 16000 -c:a libopus \"" + audio_path.string() + "\" 2>/dev/null";
    if (!runCommand(extract_cmd)) {
        std::cerr << "whisper: audio extract failed for " << video_path << '\n';
        std::error_code ec;
        std::filesystem::remove(audio_path, ec);
        result.audio_extract_ms = elapsedMs(t_extract);
        return result;
    }
    result.audio_extract_ms = elapsedMs(t_extract);

    std::string audio_bytes;
    if (!readFileBytes(audio_path, audio_bytes)) {
        std::cerr << "whisper: failed to read extracted audio\n";
        std::error_code ec;
        std::filesystem::remove(audio_path, ec);
        return result;
    }
    std::error_code ec;
    std::filesystem::remove(audio_path, ec);

    httplib::MultipartFormDataItems items;
    items.push_back(
        {.name = "file",
         .content = std::move(audio_bytes),
         .filename = "audio.ogg",
         .content_type = "audio/ogg"});

    const auto t_whisper = Clock::now();

    auto post_transcribe = [&](auto& client) -> httplib::Result {
        client.set_connection_timeout(30, 0);
        client.set_read_timeout(300, 0);
        client.set_write_timeout(30, 0);
        return client.Post("/transcribe", items);
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
        const auto j = nlohmann::json::parse(response->body);
        result.text = j.value("text", "");
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
    return std::make_unique<HttpWhisperTranscriber>(config.whisper_url, config.ffmpeg_bin_path,
                                                   config.workdir);
}

}  // namespace vlm
