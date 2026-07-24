#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "pipeline/video_context_pipeline.hpp"
#include "runtime/model_registry.hpp"
#include "types.hpp"

namespace {

bool hasExtension(std::string_view filename, std::string_view ext)
{
    if (filename.size() < ext.size()) {
        return false;
    }
    const auto f_ext = filename.substr(filename.size() - ext.size());
    std::string lower(f_ext);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == ext;
}

bool isVideoFile(std::string_view filename)
{
    return hasExtension(filename, ".mp4") || hasExtension(filename, ".avi") ||
           hasExtension(filename, ".mov") || hasExtension(filename, ".mkv");
}

bool fileExists(const std::string& path)
{
    std::ifstream f(path);
    return f.good();
}

void printUsage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " vlm_model llm_model <video> [options]\n\n"
        << "Options:\n"
        << "  --context N       RKLLM context at load (default: 8192)\n"
        << "  --max-tokens N    RKLLM max tokens at load (default: 1024)\n"
        << "  --frames N        Max frames to extract (default: 8)\n"
        << "  --frame-budget N  Frame cap override (0 = from config/formula)\n"
        << "  --lang ru|eng     Output language (default: ru)\n"
        << "  --thinking        Enable Qwen thinking mode\n"
        << "  --verbose         Log progress to stderr\n"
        << "  -h, --help        Show this help\n";
}

struct CliOptions {
    int context_len = 8192;
    int max_tokens = 1024;
    int max_frames = 8;
    int frame_budget = 0;
    bool thinking = false;
    bool verbose = false;
    std::string lang = "ru";
};

bool parseCli(int argc, char** argv, int start, CliOptions& opts)
{
    for (int i = start; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        if (arg == "--thinking") {
            opts.thinking = true;
            continue;
        }
        if (arg == "--verbose") {
            opts.verbose = true;
            continue;
        }
        if (arg == "--lang" && i + 1 < argc) {
            opts.lang = argv[++i];
            continue;
        }
        if ((arg == "--context" || arg == "--max-tokens" || arg == "--frames" ||
             arg == "--frame-budget") &&
            i + 1 < argc) {
            const int v = std::stoi(argv[++i]);
            if (arg == "--context") {
                opts.context_len = v;
            } else if (arg == "--max-tokens") {
                opts.max_tokens = v;
            } else if (arg == "--frames") {
                opts.max_frames = v;
            } else {
                opts.frame_budget = v;
            }
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << '\n';
            return false;
        }
    }
    opts.context_len = std::clamp(opts.context_len, 512, 16384);
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    CliOptions opts;
    if (!parseCli(argc, argv, 4, opts)) {
        return 1;
    }

    vlm::ModelRegistry registry;
    registry.add({.id = "cli",
                  .vision_model_path = argv[1],
                  .llm_model_path = argv[2]});
    registry.setDefaultModelId("cli");

    vlm::PipelineConfig cfg{
        .default_frames = opts.max_frames,
        .default_context = opts.context_len,
        .default_max_tokens = opts.max_tokens,
        .frame_budget = opts.frame_budget,
        .default_lang = opts.lang,
        .enable_thinking = opts.thinking,
        .verbose = opts.verbose,
    };

    vlm::VideoContextPipeline pipeline(std::move(registry), cfg);
    if (!pipeline.initialize("cli")) {
        std::cerr << "Failed to load models\n";
        return 1;
    }

    const std::string media = argv[3];
    if (!fileExists(media) || !isVideoFile(media)) {
        std::cerr << "Not a valid video file: " << media << '\n';
        return 1;
    }

    vlm::AnalyzeRequest req{
        .video_path = media,
        .model = "cli",
        .max_frames = opts.max_frames,
        .lang = opts.lang,
    };

    const auto result = pipeline.analyze(req);
    if (!result.ok()) {
        std::cerr << result.error << '\n';
        return 1;
    }

    std::cout << result.description << '\n';
    if (opts.verbose) {
        std::cerr << "frames_used=" << result.frames_used << " budget=" << result.frame_budget
                  << " capped=" << (result.frames_capped_by_context ? "yes" : "no")
                  << " transcript_status=" << result.transcript.status << '\n';
        for (const auto& [key, value] : result.metrics) {
            std::cerr << "  metric " << key << '=' << value << '\n';
        }
    }
    return 0;
}
