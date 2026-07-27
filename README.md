# video-descriptor-rkllm

[![Deploy](https://github.com/ShiWarai/video-descriptor-rkllm/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/video-descriptor-rkllm/actions/workflows/deploy.yml)
[![License: BSD-3-Clause](https://img.shields.io/github/license/ShiWarai/video-descriptor-rkllm)](LICENSE)
![Platform](https://img.shields.io/badge/platform-linux%2Farm64-blue)
![Docker](https://img.shields.io/badge/docker-GHCR-blue?logo=docker)

Описание видео/GIF на **RK3588 NPU**: Qwen3.5-VL (RKNN + RKLLM), декод кадров через **ffmpeg-rockchip** (libav), OpenAI-compatible API и Flask web UI.

## Быстрый старт

**Требования:** RK3588/S, **8 GiB RAM**, Ubuntu 22.04/24 arm64, RKNPU2 (`/dev/mpp_service`, `/dev/rga`, `/dev/dri`, `/dev/dma_heap`), веса в `./models` или `VLM_DOWNLOAD_MODELS` в `.env`.

```bash
git clone https://github.com/ShiWarai/video-descriptor-rkllm.git && cd video-descriptor-rkllm
cp .env.example .env   # whisper, автоподгрузка моделей
docker compose up -d --build
```

- **API:** http://localhost:8080 — `/health`, `/ready`, `/v1/models`, `/v1/video/analyze`
- **Web UI:** http://localhost:5000

GHCR (prod / prerelease):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull && docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml pull && docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

Образы: API `linux/arm64`, web `amd64+arm64` — `ghcr.io/shiwarai/video-descriptor-rkllm(:main|:prerelease)` и `-web`.

## Пайплайн

```mermaid
flowchart LR
  Video[Video/GIF] --> FrameExt[Frame extract]
  Video --> AudioExt[Audio libav]
  FrameExt --> RKNN[Vision RKNN 1-3x]
  AudioExt --> Whisper[whisper-rknn]
  RKNN --> RKLLM[RKLLM]
  RKLLM --> API[vlm_api_server]
  Whisper --> API
  API --> Web[Flask UI]
```

Параллельно: **кадры** (libav `h264_rkmpp` + `scale_rkrga` → RGB 448×448) + **аудио** (libav → WAV 16 kHz mono в RAM → whisper-rknn) + **vision encode** (до **3×** RKNN, адаптивно по RAM) → multimodal LLM.

| Вход | ASR | Кадры |
|------|-----|-------|
| Видео | да ([whisper-rknn](https://github.com/ShiWarai/whisper-rknn), `WHISPER_RKNN_URL`) | rkmpp |
| GIF | `transcript.status=skipped` | software libav |

LLM ждёт ASR (`ok`/`provided`), вставляет речь в начало prompt. Метрики: `image_prep_ms` — wall подготовки картинок (модель + extract + vision, параллельно); `vision_encode_ms` — только RKNN после готовности модели; `frame_extract_ms` / `transcript_ms` — wall отдельных потоков (∥). В UI % только у последовательных этапов.

**Thinking:** `enable_thinking` в config/запросе (без `/think` в prompt). В `description` только финальный ответ (`stripThinkingTags`). При thinking нужно **512–1024+** `max_tokens`; отдельный sampling: `thinking_*` в config. **Промпты:** `simple` | `detailed` (`include/core/vision_prompts.hpp`).

## HTTP API

```bash
curl -s localhost:8080/health    # liveness (loading|idle|busy)
curl -s localhost:8080/ready     # readiness (503 пока грузятся модели)
```

При заданном `VLM_API_KEY` (или `api_key` в config) все `/v1/*` требуют заголовок `Authorization: Bearer <key>`. `/health` и `/ready` остаются без auth.

`POST /v1/video/analyze` (multipart): `file`, `model` (`qwen3.5-{0.8b,2b,4b}-video`), `frames`, `frame_budget`, `max_tokens`, `lang` (`ru`|`eng`), `prompt_mode`, `enable_thinking`, `temperature`, `transcript` (override ASR).

Ответ: `job_id`, `description`, `transcript` (`status`: `ok`|`stub`|`error`|`skipped`|`provided`), `metrics` (`wall_ms`, `image_prep_ms`, `vision_encode_ms`, …), `frames_used`.

`GET /v1/jobs/{job_id}` — прогресс активного или только что завершённого job (`progress_percent` 0–100, `stage`, `stage_label`, `details`). Пока `POST /v1/video/analyze` синхронный — опрашивайте из другого клиента по `current_job_id` из `/v1/status` или `job_id` из ответа.

```bash
curl -s localhost:8080/v1/jobs/job-... -H "Authorization: Bearer $VLM_API_KEY" | jq
```

```bash
curl -s localhost:8080/v1/video/analyze \
  -H "Authorization: Bearer $VLM_API_KEY" \
  -F file=@test.mp4 -F model=qwen3.5-0.8b-video -F frames=8 -F lang=ru | jq
```

Также: `POST /v1/chat/completions` с `video_url` + `extra_body`, `GET /v1/models`.

**k8s:** liveness `/health`, readiness `/ready` (startupProbe `failureThreshold: 60` при `VLM_DOWNLOAD_MODELS=all`). Ресурсы пода `vlm_api_server` (RK3588 8 GiB, замер RSS):

`qwen3.5-2b-video` (рекомендуется):

```yaml
resources:
  requests:
    cpu: "1"
    memory: 4Gi
  limits:
    cpu: "4"
    memory: 5Gi
```

`qwen3.5-0.8b-video`:

```yaml
resources:
  requests:
    cpu: "1"
    memory: 2Gi
  limits:
    cpu: "4"
    memory: 3Gi
```

`qwen3.5-4b-video` — отдельная нода **≥12 GiB** RAM (`requests`/`limits` ~7–8Gi), на 8 GiB с k3s не ставить.

**Память:** перед загрузкой модели оценивается RSS (`estimateModelRamBytes`: LLM + N× vision RKNN + KV); при нехватке `MemAvailable` — отказ с `error` в JSON (HTTP 500), **текущая модель не выгружается**. Резерв под систему — через k8s `resources`, не в config. Vision: **3→2→1** RKNN-контекст(ов) по убыванию, пока оценка влезает; при `<3` последний воркер — `RKNN_NPU_CORE_AUTO` (runtime может задействовать несколько ядер NPU).

## Конфигурация

Переменные — [`.env.example`](.env.example) (подхватывается compose). Ключевые:

| Переменная | Назначение |
|------------|------------|
| `WHISPER_RKNN_URL` | ASR (whisper-rknn NPU или hwdsl2/whisper-server CPU); `http://` (LAN) или `https://` (интернет); пусто = stub |
| `WHISPER_API_KEY` | Bearer для `POST /v1/audio/transcriptions`; пусто = без auth |
| `VLM_DOWNLOAD_MODELS` | `0`, `0.8b`, `2b`, `4b`, `all` |
| `VLM_API_KEY` | Bearer для `/v1/*`; пусто = без auth |
| `VLM_FIX_FREQ` | NPU+CPU+DDR max (`scripts/fix_freq_rk3588.sh`, нужен `privileged`) |
| `VLM_FIX_GPU_FREQ` | `0` = не трогать Mali GPU |

**Секреты:** только в `.env` (файл в [`.gitignore`](.gitignore) и [`.dockerignore`](.dockerignore) — не коммитится и не образ не копируется). Шаблон — [`.env.example`](.env.example) без реальных ключей. `VLM_API_KEY` — наш API; `WHISPER_API_KEY` — исходящие запросы к whisper-rknn (web его не видит). Для whisper: `http://` только в доверенной сети, снаружи — `https://` (сборка с OpenSSL).

Серверный config — [`config.example.json`](config.example.json): модели, опционально `api_key` / `pipeline.whisper_api_key` (предпочтительнее env).

## Модели

Веса не в git → `./models`. Источник: [Qengineering на HF](https://huggingface.co/Qengineering).

| | Vision | LLM |
|--|--------|-----|
| 0.8B | `qwen3.5-0.8b_vision_rk3588.rknn` | `qwen3.5-0.8b_w8a8_rk3588.rkllm` |
| 2B | `qwen3.5-2b_vision_rk3588.rknn` | `qwen3.5-2b_w8a8_rk3588.rkllm` |
| 4B | `qwen3.5-4b_vision_rk3588.rknn` | `qwen3.5-4b_w8a8_rk3588.rkllm` |

```bash
MODELS_DIR=./models ./scripts/download_models.sh 0.8b   # или VLM_DOWNLOAD_MODELS в .env
```

## Сборка и тесты

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc) && ctest --output-on-failure
```

RPATH к `third_party/lib` и `ffmpeg-rockchip/lib` вшит в бинарники. Docker-тесты (как CI):

```bash
docker compose -f docker-compose.dev.yml build && docker compose -f docker-compose.dev.yml run --rm test
```

Артефакты: `build/vlm_api_server`, `build/VLM_VIDEO_NPU` (CLI не в runtime-образе).

```bash
./build/VLM_VIDEO_NPU models/*.rknn models/*.rkllm video.mp4 --frames 20 --verbose
./build/vlm_api_server --config config.json
```

## CI/CD

Push `main`/`dev` → `ctest` в Docker (`ubuntu-24.04-arm`). Prerelease: commit `[prerelease]` на `dev` или manual workflow. Telegram: secrets `TELEGRAM_TOKEN`, `TELEGRAM_TO` (опционально).

## Атрибуция

[Qengineering](https://github.com/Qengineering) · [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) · Rockchip RKLLM/RKNN

## Лицензия

BSD 3-Clause — [LICENSE](LICENSE)
