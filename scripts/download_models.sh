#!/usr/bin/env bash
# Download Qengineering RK3588 model weights from Hugging Face into MODELS_DIR.
# Skips files that already exist. Exits 0 only when all four weights are present.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/models}"
HF_BASE="https://huggingface.co/Qengineering"

mkdir -p "${MODELS_DIR}"

download_if_missing() {
    local local_name="$1"
    local repo="$2"
    local hf_name="$3"
    local dest="${MODELS_DIR}/${local_name}"

    if [[ -f "${dest}" ]]; then
        echo "skip (exists): ${local_name}"
        return 0
    fi

    local url="${HF_BASE}/${repo}/resolve/main/${hf_name}"
    local tmp="${dest}.part"
    echo "download: ${local_name} <- ${repo}/${hf_name}"
    curl -fL --retry 3 --retry-delay 5 -o "${tmp}" "${url}"
    mv "${tmp}" "${dest}"
}

# local_name | HF repo | HF filename (vision files use hyphen on HF)
download_if_missing "qwen3.5-0.8b_w8a8_rk3588.rkllm" \
    "Qwen3.5-0.8B-rk3588" "qwen3.5-0.8b_w8a8_rk3588.rkllm"
download_if_missing "qwen3.5-0.8b_vision_rk3588.rknn" \
    "Qwen3.5-0.8B-rk3588" "qwen3.5-0.8b-vision_rk3588.rknn"
download_if_missing "qwen3.5-2b_w8a8_rk3588.rkllm" \
    "Qwen3.5-2B-rk3588" "qwen3.5-2b_w8a8_rk3588.rkllm"
download_if_missing "qwen3.5-2b_vision_rk3588.rknn" \
    "Qwen3.5-2B-rk3588" "qwen3.5-2b-vision_rk3588.rknn"

for f in \
    qwen3.5-0.8b_w8a8_rk3588.rkllm \
    qwen3.5-0.8b_vision_rk3588.rknn \
    qwen3.5-2b_w8a8_rk3588.rkllm \
    qwen3.5-2b_vision_rk3588.rknn; do
    if [[ ! -f "${MODELS_DIR}/${f}" ]]; then
        echo "error: missing ${MODELS_DIR}/${f}" >&2
        exit 1
    fi
done

echo "models ready in ${MODELS_DIR}"
