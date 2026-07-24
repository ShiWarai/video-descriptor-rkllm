#!/usr/bin/env bash
# Download RK3588 model weights into MODELS_DIR.
#
# Presets (VLM_DOWNLOAD_MODELS or first argument):
#   0.8b | 2b | all | 1  — built-in Qengineering HF repos
#   comma-separated: 0.8b,2b
#
# Custom URLs (VLM_MODEL_URLS), comma- or newline-separated:
#   local_filename=https://host/path/file.rknn
#
# Only requested files are downloaded and verified.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/models}"
HF_BASE="${VLM_HF_BASE:-https://huggingface.co/Qengineering}"

mkdir -p "${MODELS_DIR}"

declare -a WANT_LOCAL=()
declare -a WANT_REPO=()
declare -a WANT_HFNAME=()
declare -a WANT_URL=()

add_catalog() {
    local preset="$1"
    case "${preset}" in
        0.8b|0.8|qwen3.5-0.8b|qwen3.5-0.8b-video)
            WANT_LOCAL+=("qwen3.5-0.8b_w8a8_rk3588.rkllm" "qwen3.5-0.8b_vision_rk3588.rknn")
            WANT_REPO+=("Qwen3.5-0.8B-rk3588" "Qwen3.5-0.8B-rk3588")
            WANT_HFNAME+=("qwen3.5-0.8b_w8a8_rk3588.rkllm" "qwen3.5-0.8b_vision_rk3588.rknn")
            WANT_URL+=("" "")
            ;;
        2b|qwen3.5-2b|qwen3.5-2b-video)
            WANT_LOCAL+=("qwen3.5-2b_w8a8_rk3588.rkllm" "qwen3.5-2b_vision_rk3588.rknn")
            WANT_REPO+=("Qwen3.5-2B-rk3588" "Qwen3.5-2B-rk3588")
            WANT_HFNAME+=("qwen3.5-2b_w8a8_rk3588.rkllm" "qwen3.5-2b_vision_rk3588.rknn")
            WANT_URL+=("" "")
            ;;
        *)
            echo "error: unknown model preset '${preset}' (use 0.8b, 2b, all)" >&2
            exit 1
            ;;
    esac
}

parse_selection() {
    local raw="${1:-}"
    raw="${raw// /}"
    if [[ -z "${raw}" || "${raw}" == "0" ]]; then
        return 0
    fi
    if [[ "${raw}" == "urls" || "${raw}" == "custom" ]]; then
        return 0
    fi
    if [[ "${raw}" == "1" || "${raw}" == "all" ]]; then
        add_catalog "0.8b"
        add_catalog "2b"
        return
    fi
    local IFS=',' preset
    for preset in ${raw}; do
        add_catalog "${preset}"
    done
}

parse_custom_urls() {
    local raw="${VLM_MODEL_URLS:-}"
    [[ -z "${raw}" ]] && return 0
    raw="${raw//$'\n'/,}"
    local IFS=',' entry local_name url
    for entry in ${raw}; do
        entry="${entry#"${entry%%[![:space:]]*}"}"
        entry="${entry%"${entry##*[![:space:]]}"}"
        [[ -z "${entry}" ]] && continue
        if [[ "${entry}" != *"="* ]]; then
            echo "error: VLM_MODEL_URLS entry must be local_name=URL, got: ${entry}" >&2
            exit 1
        fi
        local_name="${entry%%=*}"
        url="${entry#*=}"
        local_name="${local_name#"${local_name%%[![:space:]]*}"}"
        url="${url#"${url%%[![:space:]]*}"}"
        if [[ -z "${local_name}" || -z "${url}" ]]; then
            echo "error: invalid VLM_MODEL_URLS entry: ${entry}" >&2
            exit 1
        fi
        WANT_LOCAL+=("${local_name}")
        WANT_REPO+=("")
        WANT_HFNAME+=("")
        WANT_URL+=("${url}")
    done
}

download_if_missing() {
    local local_name="$1"
    local repo="$2"
    local hf_name="$3"
    local direct_url="${4:-}"
    local dest="${MODELS_DIR}/${local_name}"

    if [[ -f "${dest}" ]]; then
        echo "skip (exists): ${local_name}"
        return 0
    fi

    local url="${direct_url}"
    if [[ -z "${url}" ]]; then
        url="${HF_BASE}/${repo}/resolve/main/${hf_name}"
    fi

    local tmp="${dest}.part"
    echo "download: ${local_name} <- ${url}"
    curl -fL --retry 3 --retry-delay 5 -o "${tmp}" "${url}"
    mv "${tmp}" "${dest}"
}

selection="${VLM_DOWNLOAD_MODELS:-}"
if [[ $# -gt 0 ]]; then
    selection="$1"
fi

parse_selection "${selection}"
parse_custom_urls

if [[ ${#WANT_LOCAL[@]} -eq 0 ]]; then
    echo "error: nothing to download (set VLM_DOWNLOAD_MODELS and/or VLM_MODEL_URLS)" >&2
    exit 1
fi

for i in "${!WANT_LOCAL[@]}"; do
    download_if_missing "${WANT_LOCAL[$i]}" "${WANT_REPO[$i]}" "${WANT_HFNAME[$i]}" "${WANT_URL[$i]:-}"
done

for f in "${WANT_LOCAL[@]}"; do
    if [[ ! -f "${MODELS_DIR}/${f}" ]]; then
        echo "error: missing ${MODELS_DIR}/${f}" >&2
        exit 1
    fi
done

echo "models ready in ${MODELS_DIR}: ${WANT_LOCAL[*]}"
