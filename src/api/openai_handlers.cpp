#include "api/openai_handlers.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vlm {

namespace {

std::string randomId()
{
    static std::mt19937_64 rng{std::random_device{}()};
    std::ostringstream oss;
    oss << "chatcmpl-" << std::hex << rng();
    return oss.str();
}

std::string base64Decode(const std::string& input)
{
    static const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> table{};
    table.fill(-1);
    for (int i = 0; i < 64; ++i) {
        table[static_cast<unsigned char>(chars[static_cast<std::size_t>(i)])] = i;
    }

    std::string out;
    int val = 0;
    int valb = -8;
    for (unsigned char c : input) {
        if (std::isspace(c)) {
            continue;
        }
        if (c == '=') {
            break;
        }
        const int d = table[c];
        if (d == -1) {
            continue;
        }
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

}  // namespace

nlohmann::json analyzeResultToJson(const AnalyzeResult& result)
{
    nlohmann::json j;
    j["model"] = result.model;
    j["description"] = result.description;
    j["transcript"] = {{"text", result.transcript.text}, {"status", result.transcript.status}};
    if (!result.transcript.segments.empty()) {
        nlohmann::json segments = nlohmann::json::array();
        for (const auto& segment : result.transcript.segments) {
            segments.push_back(
                {{"start", segment.start}, {"end", segment.end}, {"text", segment.text}});
        }
        j["transcript"]["segments"] = std::move(segments);
    }
    if (result.transcript.language) {
        j["transcript"]["language"] = *result.transcript.language;
    }
    j["frames_used"] = result.frames_used;
    j["frames_requested"] = result.frames_requested;
    j["frame_budget"] = result.frame_budget;
    j["frames_capped_by_context"] = result.frames_capped_by_context;
    j["duration_sec"] = result.duration_sec;
    j["metrics"] = result.metrics;
    if (!result.ok()) {
        j["error"] = result.error;
    }
    return j;
}

nlohmann::json buildChatCompletion(const std::string& model, const AnalyzeResult& result)
{
    const auto payload = analyzeResultToJson(result);
    return nlohmann::json{
        {"id", randomId()},
        {"object", "chat.completion"},
        {"created", std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count()},
        {"model", model},
        {"choices",
         nlohmann::json::array({{{"index", 0},
                                 {"message", {{"role", "assistant"}, {"content", payload.dump()}}},
                                 {"finish_reason", result.ok() ? "stop" : "error"}}})},
        {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}},
    };
}

ParsedChatRequest parseChatCompletionRequest(const nlohmann::json& body,
                                             const PipelineConfig& defaults)
{
    ParsedChatRequest req{
        .max_frames = defaults.default_frames,
        .frame_budget = defaults.frame_budget,
        .lang = defaults.default_lang,
    };

    if (body.contains("model")) {
        req.model = body["model"].get<std::string>();
    }
    if (body.contains("max_tokens")) {
        req.max_tokens = body["max_tokens"].get<int>();
    }

    const nlohmann::json* extra = nullptr;
    if (body.contains("extra_body") && body["extra_body"].is_object()) {
        extra = &body["extra_body"];
    }
    if (extra) {
        if (extra->contains("frames")) {
            req.max_frames = (*extra)["frames"].get<int>();
        }
        if (extra->contains("frame_budget")) {
            req.frame_budget = (*extra)["frame_budget"].get<int>();
        }
        if (extra->contains("max_tokens")) {
            req.max_tokens = (*extra)["max_tokens"].get<int>();
        }
        if (extra->contains("lang")) {
            req.lang = (*extra)["lang"].get<std::string>();
        }
        if (extra->contains("prompt_mode")) {
            req.prompt_mode = (*extra)["prompt_mode"].get<std::string>();
        }
        if (extra->contains("enable_thinking")) {
            req.enable_thinking = (*extra)["enable_thinking"].get<bool>();
        }
        if (extra->contains("temperature")) {
            req.temperature = (*extra)["temperature"].get<float>();
        }
        if (extra->contains("transcript")) {
            req.transcript = (*extra)["transcript"].get<std::string>();
        }
    }

    if (body.contains("temperature") && !req.temperature) {
        req.temperature = body["temperature"].get<float>();
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        return req;
    }

    const auto& msg = body["messages"].back();
    if (!msg.contains("content") || !msg["content"].is_array()) {
        return req;
    }
    for (const auto& part : msg["content"]) {
        if (!part.contains("type")) {
            continue;
        }
        const auto type = part["type"].get<std::string>();
        if (type == "video_url" && part.contains("video_url") &&
            part["video_url"].contains("url")) {
            req.video_url = part["video_url"]["url"].get<std::string>();
        }
    }
    return req;
}

std::filesystem::path saveUploadedFile(const std::string& filename, const std::string& content,
                                       const std::filesystem::path& workdir)
{
    if (filename.empty()) {
        throw std::runtime_error("Empty upload filename");
    }
    if (content.empty()) {
        throw std::runtime_error("Empty upload content");
    }
    std::filesystem::create_directories(workdir);
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    const auto out = workdir / (std::to_string(stamp) + "_" + filename);
    std::ofstream ofs(out, std::ios::binary);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
    return out;
}

std::string saveVideoFromUrl(const std::string& video_url, const std::filesystem::path& workdir)
{
    std::filesystem::create_directories(workdir);
    const auto stamp = std::chrono::system_clock::now().time_since_epoch().count();

    if (video_url.starts_with("file://")) {
        return video_url.substr(7);
    }

    if (video_url.starts_with("data:")) {
        const auto comma = video_url.find(',');
        if (comma == std::string::npos) {
            throw std::runtime_error("Invalid data URL");
        }
        const std::string bytes = base64Decode(video_url.substr(comma + 1));
        if (bytes.empty()) {
            throw std::runtime_error("Empty data URL payload");
        }
        const auto out = workdir / (std::to_string(stamp) + ".mp4");
        std::ofstream ofs(out, std::ios::binary);
        ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        return out.string();
    }

    if (video_url.starts_with("http://") || video_url.starts_with("https://")) {
        const auto out = workdir / (std::to_string(stamp) + ".mp4");
        const std::string cmd =
            "curl -fsSL -o \"" + out.string() + "\" \"" + video_url + "\"";
        if (std::system(cmd.c_str()) != 0) {
            std::error_code ec;
            std::filesystem::remove(out, ec);
            throw std::runtime_error("Failed to download video");
        }
        if (!std::filesystem::exists(out) || std::filesystem::file_size(out) == 0) {
            std::error_code ec;
            std::filesystem::remove(out, ec);
            throw std::runtime_error("Downloaded video is empty");
        }
        return out.string();
    }

    if (std::filesystem::exists(video_url)) {
        return video_url;
    }

    throw std::runtime_error("Unsupported or missing video_url");
}

void clearWorkdir(const std::filesystem::path& workdir)
{
    std::error_code ec;
    std::filesystem::create_directories(workdir, ec);
    if (!std::filesystem::exists(workdir)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(workdir, ec)) {
        if (ec) {
            break;
        }
        std::filesystem::remove_all(entry.path(), ec);
    }
}

void removeWorkFileIfOwned(const std::filesystem::path& file, const std::filesystem::path& workdir)
{
    if (file.empty() || workdir.empty()) {
        return;
    }
    std::error_code ec;
    const auto canon_file = std::filesystem::weakly_canonical(file, ec);
    if (ec) {
        return;
    }
    const auto canon_dir = std::filesystem::weakly_canonical(workdir, ec);
    if (ec) {
        return;
    }
    const auto rel = std::filesystem::relative(canon_file, canon_dir, ec);
    if (ec || rel.empty() || rel.native().starts_with("..")) {
        return;
    }
    std::filesystem::remove(canon_file, ec);
}

}  // namespace vlm
