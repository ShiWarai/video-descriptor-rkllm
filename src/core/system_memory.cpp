#include "core/system_memory.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace vlm {

namespace {

[[nodiscard]] bool parseMeminfoValueKb(std::string_view key, std::uint64_t& out_kb)
{
    std::ifstream in("/proc/meminfo");
    if (!in) {
        return false;
    }
    std::string line;
    const std::string prefix(key);
    while (std::getline(in, line)) {
        if (line.rfind(prefix, 0) != 0) {
            continue;
        }
        std::istringstream iss(line.substr(prefix.size()));
        std::uint64_t value = 0;
        std::string unit;
        if (!(iss >> value)) {
            return false;
        }
        iss >> unit;
        if (unit == "kB" || unit == "KB") {
            out_kb = value;
            return true;
        }
        return false;
    }
    return false;
}

}  // namespace

std::optional<std::uint64_t> readMemAvailableKb()
{
    std::uint64_t kb = 0;
    if (!parseMeminfoValueKb("MemAvailable:", kb)) {
        return std::nullopt;
    }
    return kb;
}

std::optional<std::uint64_t> fileSizeBytes(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size == 0) {
        return std::nullopt;
    }
    return size;
}

std::uint64_t estimateModelRamBytes(std::string_view llm_model_path,
                                    std::string_view vision_model_path, int context_len,
                                    int vision_worker_count)
{
    std::uint64_t total = 0;
    if (const auto llm = fileSizeBytes(llm_model_path)) {
        // Weights are typically mmap'd close to file size; add headroom for runtime.
        total += static_cast<std::uint64_t>(*llm * 1.10);
    }
    if (const auto vision = fileSizeBytes(vision_model_path)) {
        // Each RKNN context loads a vision pack (last worker may be AUTO when count < 3).
        total += static_cast<std::uint64_t>(*vision) *
                 static_cast<std::uint64_t>(std::max(1, vision_worker_count)) * 1.15;
    }
    if (context_len > 0) {
        // Rough KV / activation headroom (W8A8, multimodal prefill).
        total += static_cast<std::uint64_t>(context_len) * 80 * 1024;
    }
    return total;
}

int pickVisionWorkerCount(std::string_view llm_model_path, std::string_view vision_model_path,
                          int context_len, int max_workers, std::uint64_t credit_bytes,
                          std::string* reason)
{
    const int capped = std::clamp(max_workers, 1, 3);
    const auto avail_kb = readMemAvailableKb();
    if (!avail_kb) {
        return capped;
    }
    const std::uint64_t avail_bytes = *avail_kb * 1024 + credit_bytes;

    for (int n = capped; n >= 1; --n) {
        const std::uint64_t need =
            estimateModelRamBytes(llm_model_path, vision_model_path, context_len, n);
        if (avail_bytes >= need) {
            return n;
        }
    }

    if (reason) {
        const std::uint64_t need1 =
            estimateModelRamBytes(llm_model_path, vision_model_path, context_len, 1);
        std::ostringstream oss;
        oss << "insufficient RAM: need ~" << (need1 / (1024 * 1024))
            << " MiB even with 1 vision worker, MemAvailable ~" << (*avail_kb / 1024) << " MiB";
        if (credit_bytes > 0) {
            oss << " (+" << (credit_bytes / (1024 * 1024)) << " MiB credit from unload)";
        }
        *reason = oss.str();
    }
    return 0;
}

bool hasEnoughRamForModel(std::uint64_t required_bytes, std::string* reason)
{
    const auto avail_kb = readMemAvailableKb();
    if (!avail_kb) {
        return true;
    }
    const std::uint64_t avail_bytes = *avail_kb * 1024;
    if (avail_bytes >= required_bytes) {
        return true;
    }
    if (reason) {
        std::ostringstream oss;
        oss << "insufficient RAM: need ~" << (required_bytes / (1024 * 1024))
            << " MiB (estimated model RSS), MemAvailable ~" << (*avail_kb / 1024) << " MiB";
        *reason = oss.str();
    }
    return false;
}

}  // namespace vlm
