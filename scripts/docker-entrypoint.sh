#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
ROLE="${VLM_ROLE:-all}"
CONFIG="${APP_DIR}/config.json"

if [[ "${VLM_FIX_FREQ:-1}" == "1" || "${VLM_FIX_FREQ}" == "true" || "${VLM_FIX_FREQ}" == "yes" ]]; then
    if [[ -x "${APP_DIR}/scripts/fix_freq_rk3588.sh" ]]; then
        "${APP_DIR}/scripts/fix_freq_rk3588.sh" || echo "fix_freq: skipped or partial (non-fatal)" >&2
    fi
fi

case "${ROLE}" in
    all|local|"")
        exec "${APP_DIR}/build/vlm_api_server" --config "${CONFIG}" "$@"
        ;;
    gateway)
        export VLM_RUNTIME="${VLM_RUNTIME:-distributed}"
        exec "${APP_DIR}/build/vlm_api_server" --config "${CONFIG}" "$@"
        ;;
    vision)
        exec "${APP_DIR}/build/vlm_vision_worker" --config "${CONFIG}" "$@"
        ;;
    llm)
        exec "${APP_DIR}/build/vlm_llm_worker" --config "${CONFIG}" "$@"
        ;;
    *)
        echo "Unknown VLM_ROLE=${ROLE}" >&2
        exit 1
        ;;
esac
