#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API_URL="${VLM_API_URL:-http://127.0.0.1:8080}"
VIDEO="${ROOT}/test_video.mp4"

echo "== API smoke =="
curl -sf "${API_URL}/health" >/dev/null
for _ in $(seq 1 60); do
  if curl -sf "${API_URL}/ready" >/dev/null; then
    break
  fi
  sleep 5
done
curl -sf "${API_URL}/ready" >/dev/null || {
  echo "API not ready (${API_URL}/ready)"
  exit 1
}
models="$(curl -sf "${API_URL}/v1/models")"
echo "${models}" | grep -q '"qwen3.5-0.8b-video"' || {
  echo "missing 0.8b model in /v1/models"
  exit 1
}

if [[ ! -f "${VIDEO}" ]]; then
  echo "skip e2e analyze: ${VIDEO} not found"
  exit 0
fi

echo "== POST /v1/video/analyze =="
resp="$(curl -sf -X POST "${API_URL}/v1/video/analyze" \
  -F "file=@${VIDEO}" \
  -F "model=qwen3.5-0.8b-video" \
  -F "frames=4" \
  -F "lang=ru")"

echo "${resp}" | grep -q '"description"' || {
  echo "no description in response"
  echo "${resp}"
  exit 1
}

if echo "${resp}" | grep -q '<think>'; then
  echo "thinking tags leaked into description"
  exit 1
fi

# When enable_thinking=false, raw path must use /no_think (checked via server logs separately)
if echo "${resp}" | grep -qi 'thinking\|</think>'; then
  echo "thinking residue in response"
  exit 1
fi

budget="$(echo "${resp}" | sed -n 's/.*"frame_budget":\([0-9]*\).*/\1/p')"
used="$(echo "${resp}" | sed -n 's/.*"frames_used":\([0-9]*\).*/\1/p')"
if [[ -n "${budget}" && -n "${used}" && "${used}" -gt "${budget}" ]]; then
  echo "frames_used (${used}) > frame_budget (${budget})"
  exit 1
fi

status="$(curl -sf "${API_URL}/v1/status")"
if ! echo "${status}" | grep -q '"model_loaded":true'; then
  echo "model should stay loaded after request"
  echo "${status}"
  exit 1
fi

echo "e2e_analyze: ok"
