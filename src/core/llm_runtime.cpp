#include "core/llm_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string_view>

#include "core/text_util.hpp"

namespace vlm {

namespace {

constexpr std::string_view kThinkingOpenTag = "<think>\n";

}  // namespace

LlmRuntime::LlmRuntime()
{
    input_ = {};
    infer_params_ = {};
    sampling_ = {};
    callback_ = {};
    param_ = rkllm_createDefaultParam();
    // Qwen3.5-VL non-thinking card defaults (not Qengineering's greedy top_k=1).
    param_.top_k = default_top_k_;
    param_.top_p = default_top_p_;
    param_.temperature = default_temperature_;
    param_.presence_penalty = default_presence_penalty_;
    param_.skip_special_token = true;
    param_.extend_param.base_domain_id = 1;
}

void LlmRuntime::setSamplingDefaults(int top_k, float top_p, float temperature,
                                     float presence_penalty) noexcept
{
    default_top_k_ = std::max(1, top_k);
    default_top_p_ = top_p;
    default_temperature_ = temperature;
    default_presence_penalty_ = presence_penalty;
    param_.top_k = default_top_k_;
    param_.top_p = default_top_p_;
    param_.temperature = default_temperature_;
    param_.presence_penalty = default_presence_penalty_;
}

void LlmRuntime::setThinkingSamplingDefaults(float temperature, float top_p,
                                             float presence_penalty) noexcept
{
    thinking_temperature_ = temperature;
    thinking_top_p_ = top_p;
    thinking_presence_penalty_ = presence_penalty;
}

LlmRuntime::~LlmRuntime()
{
    unload();
}

void LlmRuntime::unload()
{
    if (handle_) {
        rkllm_destroy(handle_);
        handle_ = nullptr;
    }
}

std::string LlmRuntime::normalizeLang(std::string_view lang)
{
    std::string v(lang);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "eng" || v == "en" || v == "english") {
        return "eng";
    }
    return "ru";
}

bool LlmRuntime::setOutputLang(std::string_view lang)
{
    const std::string normalized = normalizeLang(lang);
    if (normalized != "ru" && normalized != "eng") {
        return false;
    }
    output_lang_ = normalized;
    return true;
}

std::string LlmRuntime::buildUserVisionPrompt(std::string_view lang, bool enable_thinking,
                                              std::string_view prompt_mode,
                                              std::string_view transcript)
{
    const std::string normalized = normalizeLang(lang);
    const bool simple = (prompt_mode == "simple");
    // Do NOT call rkllm_set_chat_template: it disables input.enable_thinking (RKLLM warning).
    std::string prompt = "<image>\n";
    if (simple) {
        prompt += (normalized == "eng") ? "Give a brief, to-the-point description of the video/GIF."
                                        : "Опиши кратко и по делу видео/gif.";
    } else if (normalized == "eng") {
        prompt +=
            "Describe this video from the frames in time order. Factual only, no invention. "
            "Answer in English:\n"
            "## Summary\nWhat the clip is about.\n"
            "## Actions\nWhat changes across frames, in order.\n"
            "## On-screen text\nQuote or none.\n"
            "## Genre\nOne short label.";
    } else {
        prompt +=
            "Опиши это видео по кадрам по порядку. Только факты, без выдумок. Ответ на русском:\n"
            "## О чём\nСуть ролика.\n"
            "## Действия\nЧто меняется между кадрами, по порядку.\n"
            "## Текст на экране\nЦитата или «нет».\n"
            "## Жанр\nОдин ярлык.";
    }
    if (!transcript.empty()) {
        prompt += (normalized == "eng") ? "\n\nSpeech in the video:\n" : "\n\nРечь на видео:\n";
        prompt += transcript;
    }
    prompt += enable_thinking ? " /think" : " /no_think";
    return prompt;
}

bool LlmRuntime::load(const std::string& llm_model_path, int32_t max_new_tokens,
                      int32_t context_len, const std::string& lang, bool enable_thinking,
                      bool verbose)
{
    verbose_ = verbose;
    enable_thinking_ = enable_thinking;
    output_lang_ = normalizeLang(lang);

    param_.model_path = llm_model_path.c_str();
    param_.max_new_tokens = max_new_tokens;
    param_.max_context_len = context_len;

    callback_.result_callback = LlmRuntime::staticCallback;
    callback_.result_userdata = this;

    const int ret = rkllm_init(&handle_, &param_, &callback_);
    if (ret != 0) {
        std::cerr << "rkllm_init failed: " << ret << '\n';
        return false;
    }
    if (verbose_) {
        std::cerr << "rkllm init success (enable_thinking="
                  << (enable_thinking_ ? "true" : "false")
                  << ", using built-in chat template)\n";
    }

    infer_params_.mode = RKLLM_INFER_GENERATE;
    infer_params_.keep_history = 0;
    // Intentionally NOT calling rkllm_set_chat_template — it disables enable_thinking.
    return true;
}

int LlmRuntime::staticCallback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    if (!userdata) {
        return -1;
    }
    return static_cast<LlmRuntime*>(userdata)->instanceCallback(result, state);
}

int LlmRuntime::instanceCallback(RKLLMResult* result, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH) {
        if (result) {
            last_generate_tokens_ = result->perf.generate_tokens;
            last_prefill_tokens_ = result->perf.prefill_tokens;
        }
        if (verbose_ && stream_started_) {
            std::cerr << "\n----- end stream -----\n" << std::flush;
            std::cerr << "rkllm perf: prefill_tokens=" << last_prefill_tokens_
                      << " generate_tokens=" << last_generate_tokens_
                      << " max_new_tokens=" << last_max_new_tokens_;
            if (lastTruncatedByMaxTokens()) {
                std::cerr << " truncated=yes (hit max_new_tokens hard stop)";
            }
            std::cerr << '\n';
        }
        std::lock_guard lock(response_mutex_);
        response_ready_ = true;
        response_cv_.notify_all();
    } else if (state == RKLLM_RUN_ERROR) {
        if (verbose_ && stream_started_) {
            std::cerr << "\n----- stream error -----\n" << std::flush;
        }
        std::lock_guard lock(response_mutex_);
        response_buffer_ += "[Error during inference]";
        response_ready_ = true;
        response_cv_.notify_all();
    } else if (state == RKLLM_RUN_NORMAL && result && result->text) {
        if (enable_thinking_ && !thinking_open_printed_) {
            thinking_open_printed_ = true;
            if (!std::string_view(result->text).starts_with("<think>")) {
                response_buffer_ += std::string(kThinkingOpenTag);
                if (verbose_) {
                    if (!stream_started_) {
                        std::cerr << "----- stream -----\n";
                        stream_started_ = true;
                    }
                    std::cerr << kThinkingOpenTag << std::flush;
                }
            }
        }
        response_buffer_ += result->text;
        if (verbose_) {
            if (!stream_started_) {
                std::cerr << "----- stream -----\n";
                stream_started_ = true;
            }
            std::cerr << result->text << std::flush;
        }
    }
    return 0;
}

std::string LlmRuntime::expandImageTags(const std::string& prompt, std::size_t image_count)
{
    if (image_count <= 1 || prompt.find("<image>") == std::string::npos) {
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

std::string LlmRuntime::generateMultimodal(const std::string& prompt, const VisionEncoder& vision,
                                            int max_new_tokens, float temperature)
{
    if (!handle_ || vision.frameCount() == 0) {
        return {};
    }

    {
        std::lock_guard lock(response_mutex_);
        response_buffer_.clear();
        response_ready_ = false;
        thinking_open_printed_ = false;
        stream_started_ = false;
    }

    multimodal_prompt_ = expandImageTags(prompt, vision.frameCount());
    const auto& info = vision.modelInfo();

    input_ = {};
    input_.input_type = RKLLM_INPUT_MULTIMODAL;
    input_.role = "user";
    input_.enable_thinking = enable_thinking_;
    input_.multimodal_input.prompt = multimodal_prompt_.data();
    input_.multimodal_input.image.image_embed = const_cast<float*>(vision.embeddings().data());
    input_.multimodal_input.image.n_image_tokens = info.image_tokens;
    input_.multimodal_input.image.n_image = vision.frameCount();
    input_.multimodal_input.image.image_start = "<|vision_start|>";
    input_.multimodal_input.image.image_end = "<|vision_end|>";
    input_.multimodal_input.image.image_content = "<|image_pad|>";
    input_.multimodal_input.image.image_height = info.height;
    input_.multimodal_input.image.image_width = info.width;

    // <=0 keeps the value from rkllm_init (RKLLMInferParam docs).
    const int effective_max =
        max_new_tokens > 0 ? max_new_tokens : param_.max_new_tokens;
    infer_params_.max_new_tokens = max_new_tokens > 0 ? max_new_tokens : 0;
    last_max_new_tokens_ = effective_max;
    last_generate_tokens_ = 0;
    last_prefill_tokens_ = 0;

    // Official Qwen3.5-VL card: non-thinking vs thinking presets.
    const float base_temp =
        enable_thinking_ ? thinking_temperature_ : default_temperature_;
    const float base_top_p = enable_thinking_ ? thinking_top_p_ : default_top_p_;
    const float base_presence =
        enable_thinking_ ? thinking_presence_penalty_ : default_presence_penalty_;
    const float temp = temperature >= 0.0f ? temperature : base_temp;

    sampling_ = {};
    sampling_.temperature = temp;
    sampling_.top_p = base_top_p;
    // top_k=1 makes temperature irrelevant — bump when sampling.
    sampling_.top_k = (temp > 1e-6f && default_top_k_ <= 1) ? 20 : default_top_k_;
    sampling_.repeat_penalty = param_.repeat_penalty;
    sampling_.frequency_penalty = param_.frequency_penalty;
    sampling_.presence_penalty = base_presence;
    infer_params_.sampling_params = &sampling_;

    if (verbose_) {
        const int tokens = infer_params_.max_new_tokens > 0 ? infer_params_.max_new_tokens
                                                            : param_.max_new_tokens;
        std::cerr << "rkllm_run enable_thinking=" << (enable_thinking_ ? "true" : "false")
                  << " max_new_tokens=" << tokens
                  << (infer_params_.max_new_tokens > 0 ? " (override)" : " (init)")
                  << " temperature=" << sampling_.temperature << " top_k=" << sampling_.top_k
                  << " top_p=" << sampling_.top_p
                  << " presence_penalty=" << sampling_.presence_penalty
                  << " frames=" << vision.frameCount() << '\n';
        std::cerr << "----- prompt -----\n" << multimodal_prompt_ << "\n----- end prompt -----\n";
    }

    const int ret = rkllm_run(handle_, &input_, &infer_params_, nullptr);
    if (ret != 0) {
        std::cerr << "rkllm_run returned " << ret << '\n';
    }

    std::unique_lock lock(response_mutex_);
    response_cv_.wait(lock, [this] { return response_ready_; });

    const std::string raw = response_buffer_;
    const std::string stripped = stripThinkingTags(raw);
    if (verbose_) {
        std::cerr << "----- answer (API response, thinking stripped) -----\n"
                  << stripped << '\n'
                  << "----- end answer -----\n";
    }
    return stripped;
}

void LlmRuntime::clearKvCache()
{
    if (handle_) {
        rkllm_clear_kv_cache(handle_, 1, nullptr, nullptr);
    }
}

}  // namespace vlm
