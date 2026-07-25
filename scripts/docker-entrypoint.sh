#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"

exec "${APP_DIR}/build/vlm_api_server" --config "${APP_DIR}/config.json" "$@"
