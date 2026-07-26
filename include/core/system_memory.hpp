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
 * Conservative RSS estimate for one loaded model pack (LLM mmap + N× RKNN vision + KV).
 * Intentionally high — acts as headroom without a separate system reserve.
 */
[[nodiscard]] std::uint64_t estimateModelRamBytes(std::string_view llm_model_path,
                                                  std::string_view vision_model_path,
                                                  int context_len, int vision_worker_count);

/**
 * Largest vision worker count in [1, max_workers] that fits MemAvailable (+ optional credit
 * for memory freed by unloading the current model). Returns 0 if even 1 worker does not fit.
 * If MemAvailable is unreadable → returns max_workers (allow load).
 */
[[nodiscard]] int pickVisionWorkerCount(std::string_view llm_model_path,
                                        std::string_view vision_model_path, int context_len,
                                        int max_workers, std::uint64_t credit_bytes = 0,
                                        std::string* reason = nullptr);

/** True if MemAvailable (or unknown → allow) is enough for estimated model RSS. */
[[nodiscard]] bool hasEnoughRamForModel(std::uint64_t required_bytes, std::string* reason = nullptr);

}  // namespace vlm
