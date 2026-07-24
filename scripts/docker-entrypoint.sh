#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
MODELS_DIR="${MODELS_DIR:-/app/models}"

if [[ "${VLM_DOWNLOAD_MODELS:-0}" == "1" ]]; then
    echo "VLM_DOWNLOAD_MODELS=1: checking models in ${MODELS_DIR}"
    MODELS_DIR="${MODELS_DIR}" "${APP_DIR}/scripts/download_models.sh"
fi

exec "${APP_DIR}/build/vlm_api_server" --config "${APP_DIR}/config.json" "$@"
