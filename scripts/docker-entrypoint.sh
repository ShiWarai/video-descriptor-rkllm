#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"

if [[ "${VLM_FIX_FREQ:-1}" == "1" || "${VLM_FIX_FREQ}" == "true" || "${VLM_FIX_FREQ}" == "yes" ]]; then
    if [[ -x "${APP_DIR}/scripts/fix_freq_rk3588.sh" ]]; then
        "${APP_DIR}/scripts/fix_freq_rk3588.sh" || echo "fix_freq: skipped or partial (non-fatal)" >&2
    fi
fi

exec "${APP_DIR}/build/vlm_api_server" --config "${APP_DIR}/config.json" "$@"
