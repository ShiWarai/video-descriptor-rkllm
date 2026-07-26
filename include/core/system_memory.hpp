#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace vlm {

/** MemAvailable from /proc/meminfo (kB), or nullopt if unreadable. */
[[nodiscard]] std::optional<std::uint64_t> readMemAvailableKb();

/** File size in bytes, or nullopt if missing/empty. */
[[nodiscard]] std::optional<std::uint64_t> fileSizeBytes(const std::filesystem::path& path);

/**
 * Conservative RSS estimate for one loaded model pack (LLM mmap + 3× RKNN vision + KV).
 * Intentionally high — acts as headroom without a separate system reserve.
 */
[[nodiscard]] std::uint64_t estimateModelRamBytes(std::string_view llm_model_path,
                                                  std::string_view vision_model_path,
                                                  int context_len, int vision_worker_count);

/** True if MemAvailable (or unknown → allow) is enough for estimated model RSS. */
[[nodiscard]] bool hasEnoughRamForModel(std::uint64_t required_bytes, std::string* reason = nullptr);

}  // namespace vlm
