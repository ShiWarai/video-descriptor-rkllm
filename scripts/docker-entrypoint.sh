#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
MODELS_DIR="${MODELS_DIR:-/app/models}"

if [[ -n "${VLM_DOWNLOAD_MODELS:-}" && "${VLM_DOWNLOAD_MODELS}" != "0" ]]; then
    echo "VLM_DOWNLOAD_MODELS=${VLM_DOWNLOAD_MODELS}: checking models in ${MODELS_DIR}"
    MODELS_DIR="${MODELS_DIR}" VLM_DOWNLOAD_MODELS="${VLM_DOWNLOAD_MODELS}" \
        VLM_MODEL_URLS="${VLM_MODEL_URLS:-}" \
        "${APP_DIR}/scripts/download_models.sh"
fi

exec "${APP_DIR}/build/vlm_api_server" --config "${APP_DIR}/config.json" "$@"
