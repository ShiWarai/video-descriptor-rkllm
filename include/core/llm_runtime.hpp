#pragma once

#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>

#include <rkllm/rkllm.h>
#include "core/vision_encoder.hpp"

namespace vlm {

class LlmRuntime {
public:
    LlmRuntime();
    ~LlmRuntime();

    LlmRuntime(const LlmRuntime&) = delete;
    LlmRuntime& operator=(const LlmRuntime&) = delete;

    [[nodiscard]] bool load(const std::string& llm_model_path, int32_t max_new_tokens,
                            int32_t context_len, const std::string& lang = "ru",
                            bool enable_thinking = false, bool verbose = false);

    void unload();
    [[nodiscard]] bool loaded() const noexcept { return handle_ != nullptr; }

    void setEnableThinking(bool enable) noexcept { enable_thinking_ = enable; }
    [[nodiscard]] bool enableThinking() const noexcept { return enable_thinking_; }
    [[nodiscard]] bool setOutputLang(std::string_view lang);
    [[nodiscard]] const std::string& outputLang() const noexcept { return output_lang_; }

    [[nodiscard]] std::string generateMultimodal(const std::string& prompt,
                                                 const VisionEncoder& vision,
                                                 int max_new_tokens = 0,
                                                 float temperature = -1.0f);
    void clearKvCache();

    /** Stats from the last generateMultimodal() call. */
    [[nodiscard]] int lastMaxNewTokens() const noexcept { return last_max_new_tokens_; }
    [[nodiscard]] int lastGenerateTokens() const noexcept { return last_generate_tokens_; }
    [[nodiscard]] bool lastTruncatedByMaxTokens() const noexcept {
        return last_max_new_tokens_ > 0 && last_generate_tokens_ >= last_max_new_tokens_;
    }

    void setSamplingDefaults(int top_k, float top_p, float temperature,
                             float presence_penalty = 1.5f) noexcept;
    void setThinkingSamplingDefaults(float temperature, float top_p,
                                     float presence_penalty) noexcept;

    /** User multimodal prompt. prompt_mode: "simple" | "detailed".
     *  Texts live in core/vision_prompts.hpp; optional ASR goes with frames, before the task. */
    [[nodiscard]] static std::string buildUserVisionPrompt(
        std::string_view lang, bool enable_thinking, std::string_view prompt_mode = "detailed",
        std::string_view transcript = {});

private:
    LLMHandle handle_ = nullptr;
    RKLLMParam param_{};
    RKLLMInput input_{};
    RKLLMInferParam infer_params_{};
    RKLLMSamplingParam sampling_{};
    RKLLMCallback callback_{};

    std::string output_lang_ = "ru";
    bool enable_thinking_ = false;
    bool verbose_ = false;
    int default_top_k_ = 20;
    float default_top_p_ = 0.8f;
    float default_temperature_ = 0.7f;
    float default_presence_penalty_ = 1.5f;
    float thinking_temperature_ = 0.6f;
    float thinking_top_p_ = 0.95f;
    float thinking_presence_penalty_ = 0.0f;

    std::string response_buffer_;
    std::string multimodal_prompt_;
    std::mutex response_mutex_;
    std::condition_variable response_cv_;
    bool response_ready_ = false;
    bool thinking_open_printed_ = false;
    bool stream_started_ = false;
    int last_max_new_tokens_ = 0;
    int last_generate_tokens_ = 0;
    int last_prefill_tokens_ = 0;

    static int staticCallback(RKLLMResult* result, void* userdata, LLMCallState state);
    int instanceCallback(RKLLMResult* result, LLMCallState state);
    [[nodiscard]] static std::string normalizeLang(std::string_view lang);
    [[nodiscard]] static std::string expandImageTags(const std::string& prompt,
                                                     std::size_t image_count);
};

}  // namespace vlm
