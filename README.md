# video-descriptor-rkllm

[![Deploy](https://github.com/ShiWarai/video-descriptor-rkllm/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShiWarai/video-descriptor-rkllm/actions/workflows/deploy.yml)
[![License: BSD-3-Clause](https://img.shields.io/github/license/ShiWarai/video-descriptor-rkllm)](LICENSE)
![Platform](https://img.shields.io/badge/platform-linux%2Farm64-blue)
![Docker](https://img.shields.io/badge/docker-GHCR-blue?logo=docker)

Описание видео на **Rockchip RK3588 NPU**: Qwen3.5-VL (RKNN + RKLLM), аппаратный декод через **ffmpeg-rockchip**, OpenAI-compatible HTTP API и Flask web UI.

## Оглавление

| Раздел | Содержание |
|--------|------------|
| [Быстрый старт](#быстрый-старт) | Docker Compose на RK3588 |
| [Требования](#требования) | Железо, драйверы, модели |
| [Архитектура](#архитектура) | Пайплайн video → API |
| [HTTP API](#http-api) | Эндпоинты |
| [CLI](#cli) | Локальный запуск |
| [Конфигурация](#конфигурация) | `.env`, `config.json` |
| [Модели](#модели) | HF, автоподгрузка |
| [Сборка из исходников](#сборка-из-исходников) | CMake, тесты |
| [CI/CD](#cicd) | GHCR `:main` / `:prerelease` |
| [Атрибуция](#атрибуция) | Upstream |
| [Лицензия](#лицензия) | BSD 3-Clause |

---

## Быстрый старт

На плате RK3588 (arm64):

```bash
git clone https://github.com/ShiWarai/video-descriptor-rkllm.git
cd video-descriptor-rkllm

cp .env.example .env
# Отредактируйте .env: whisper, автоподгрузка моделей и т.д.

# Положите веса в `./models` или включите автоподгрузку в .env:
# VLM_DOWNLOAD_MODELS=0.8b
docker compose up -d --build
```

- **API:** http://localhost:8080 (`/health`, `/v1/models`, `/v1/video/analyze`)
- **Web UI:** http://localhost:5000

Продакшен (образы из GHCR):

```bash
cp .env.example .env   # если ещё не создан
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Prerelease:

```bash
cp .env.example .env   # если ещё не создан
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml pull
docker compose -f docker-compose.yml -f docker-compose.prerelease.yml up -d
```

---

## Требования

| Компонент | Версия / примечание |
|-----------|---------------------|
| SoC | Rockchip **RK3588 / RK3588S** |
| OS | Ubuntu 22.04 / 24.04 arm64 |
| NPU | RKNPU2 driver, устройства `/dev/mpp_service`, `/dev/rga`, `/dev/dri`, `/dev/dma_heap` |
| Runtime | RKLLM 1.3.0+, RKNN (в `third_party/lib`) |
| OpenCV | host: `libopencv-dev` (core+imgproc); Docker: минимальная сборка в образе (~без Mesa/LLVM) |
| ffmpeg | vendored **ffmpeg-rockchip** (`third_party/ffmpeg-rockchip/`) |
| Модели | не в git — volume `./models` или `VLM_DOWNLOAD_MODELS` в `.env` |

---

## Архитектура

```mermaid
flowchart LR
  Video[Video file] --> FrameExt[Frame extract]
  Video --> AudioExt[Audio extract]
  Video --> ModelLoad[Model load]
  FrameExt --> RKNN[VisionEncoder RKNN]
  AudioExt --> Whisper[whisper-rknn ASR]
  ModelLoad --> RKNN
  RKNN --> RKLLM[LlmRuntime RKLLM]
  RKLLM --> API[vlm_api_server]
  Whisper --> API
  API --> Web[web_client Flask]
```

Сервис: видео → **параллельно** кадры (ffmpeg-rockchip) + аудио → ASR ([whisper-rknn](https://github.com/ShiWarai/whisper-rknn)) + vision encoder (RKNN) → multimodal LLM (RKLLM) → **description** + **transcript**.

ASR и vision encode идут параллельно. Перед генерацией описания LLM **дожидается** ASR (если `status=ok` / `provided`) и вставляет речь **в начало prompt** (кадры + речь, затем задание). В ответе API транскрипт также возвращается отдельно.

**Режимы промпта** (`prompt_mode`):

| Режим | Структура |
|-------|-----------|
| `simple` | «Тебе даны кадры… [с речью…]» → «Опиши кратко и по делу видео.» |
| `detailed` (по умолчанию в UI) | то же вступление → «Опиши это видео…» + пункты 1–3 (о чём / текст на экране / жанр) |

Тексты промптов — в `include/core/vision_prompts.hpp`; сборка через `LlmRuntime::buildUserVisionPrompt`.

---

## HTTP API

### Статус

```bash
curl -s http://localhost:8080/health   # liveness: процесс жив (status=loading|idle|busy)
curl -s http://localhost:8080/ready    # readiness: 200 когда модели скачаны и сервис готов
curl -s http://localhost:8080/v1/status | jq
```

При `VLM_DOWNLOAD_MODELS` сервер поднимает HTTP сразу (`/health` → 200, `status: loading`), а скачивание весов идёт в фоне startup. `web` и k8s readiness ждут `/ready` (503 до окончания загрузки).

**k3s / Kubernetes probes:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  periodSeconds: 15
  failureThreshold: 40   # до ~10 мин на первую загрузку all
startupProbe:
  httpGet:
    path: /ready
    port: 8080
  periodSeconds: 15
  failureThreshold: 60   # до ~15 мин на первую загрузку all
```

### `POST /v1/video/analyze` (multipart)

```bash
curl -s http://localhost:8080/v1/video/analyze \
  -F "file=@test_video.mp4" \
  -F "model=qwen3.5-0.8b-video" \
  -F "frames=8" \
  -F "frame_budget=28" \
  -F "lang=ru" | jq
```

### `POST /v1/chat/completions` (OpenAI-compatible + `video_url`)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-0.8b-video",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "file:///app/test_video.mp4"}}
      ]
    }],
    "extra_body": {"frames": 8, "frame_budget": 28, "lang": "ru"}
  }' | jq
```

### `GET /v1/models`

Список моделей из `config.json` (без alias в ответе).

---

## CLI

```bash
LD_LIBRARY_PATH=./third_party/lib:./third_party/ffmpeg-rockchip/lib \
  ./build/VLM_VIDEO_NPU \
  models/qwen3.5-0.8b_vision_rk3588.rknn \
  models/qwen3.5-0.8b_w8a8_rk3588.rkllm \
  test_video.mp4 --context 8192 --frames 20 --verbose
```

---

## Конфигурация

### Docker (`.env`)

Скопируйте шаблон и отредактируйте под свою среду:

```bash
cp .env.example .env
```

Docker Compose автоматически подхватывает `.env` из корня проекта. После изменений перезапустите сервисы:

```bash
docker compose up -d
```

| Переменная | Описание |
|------------|----------|
| `WHISPER_RKNN_URL` | URL whisper-rknn. Пусто = stub ASR без транскрипции |
| `VLM_DOWNLOAD_MODELS` | Автоподгрузка моделей VLM: `0`, `0.8b`, `2b`, `all` |
| `VLM_MODEL_URLS` | Свои URL весов: `имя_файла=https://...` |
| `VLM_VERBOSE` | Подробные логи API (`1` / `0`) |

`WHISPER_RKNN_URL` переопределяет `pipeline.whisper_url` из `config.json`.

Пример для k3s-кластера — в `.env`:

```bash
WHISPER_RKNN_URL=http://whisper-rknn.whisper-rknn.svc.cluster.local:8080
```

Или разово без файла:

```bash
WHISPER_RKNN_URL=http://whisper-rknn.whisper-rknn.svc.cluster.local:8080 docker compose up -d api
```

Whisper health: `GET {WHISPER_RKNN_URL}/health` → `{"status":"ok"}`.  
Transcribe: `POST {WHISPER_RKNN_URL}/transcribe` (multipart, поле `file`).

### `config.json`

Пример: [`config.example.json`](config.example.json). Основные поля:

| Поле | Описание |
|------|----------|
| `models[]` | id, пути к `.rknn` / `.rkllm`, sampling overrides |
| `default_model` | модель по умолчанию |
| `preload_model` | загрузить на старте (опционально) |
| `pipeline.frame_budget` | потолок кадров (28 по умолчанию) |
| `pipeline.ffmpeg_bin_path` | `third_party/ffmpeg-rockchip/bin` |
| `pipeline.whisper_url` | URL whisper-rknn (пусто = stub; в Docker обычно задаётся через `.env`) |

---

## Модели

Веса **не** в git. Каталог `./models` на хосте создаётся при `docker compose up` (volume `./models` → `/app/models`).

| Модель | Vision (RKNN) | LLM (RKLLM) | Hugging Face |
|--------|---------------|-------------|--------------|
| Qwen3.5-0.8B | `qwen3.5-0.8b_vision_rk3588.rknn` | `qwen3.5-0.8b_w8a8_rk3588.rkllm` | [HF](https://huggingface.co/Qengineering/Qwen3.5-0.8B-rk3588) |
| Qwen3.5-2B | `qwen3.5-2b_vision_rk3588.rknn` | `qwen3.5-2b_w8a8_rk3588.rkllm` | [HF](https://huggingface.co/Qengineering/Qwen3.5-2B-rk3588) |

**Автоподгрузка (opt-in):** в `.env` задайте `VLM_DOWNLOAD_MODELS=0.8b` (или `2b`, `all`, `0.8b,2b`). Свои URL: `VLM_MODEL_URLS=имя_файла=https://...`

| `VLM_DOWNLOAD_MODELS` | Что качается |
|-----------------------|--------------|
| `0` / не задано | ничего |
| `0.8b` | только 0.8B |
| `2b` | только 2B |
| `all` / `1` | обе |

```bash
# в .env: VLM_DOWNLOAD_MODELS=0.8b
docker compose up -d api
```

Ручная загрузка:

```bash
MODELS_DIR=./models ./scripts/download_models.sh 0.8b
```

---

## Сборка из исходников

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

Или unit-тесты в Docker (как в CI, без NPU/моделей):

```bash
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml run --rm test
```

Артефакты: `build/vlm_api_server`, `build/VLM_VIDEO_NPU`.

```bash
LD_LIBRARY_PATH=./third_party/lib:./third_party/ffmpeg-rockchip/lib \
  ./build/vlm_api_server --config config.json
```

### Структура проекта

```
include/              публичные заголовки
src/                  реализация C++
third_party/
  lib/                librkllmrt.so, librknnrt.so
  ffmpeg-rockchip/    vendored ffmpeg + Rockchip libs
  rkllm/              C API headers
scripts/              download_models.sh, docker-entrypoint.sh
web_client/           Flask UI
models/               веса (volume mount)
.env.example          шаблон переменных окружения для Docker
```

---

## CI/CD

| Workflow | Триггер | Результат |
|----------|---------|-----------|
| **Deploy** | push `main` / `dev` | `docker-compose.dev.yml` → build + `ctest` (`ubuntu-24.04-arm`) |
| **Deploy → prerelease** | push `dev` с `[prerelease]` или manual `publish_prerelease` | `ghcr.io/shiwarai/video-descriptor-rkllm:prerelease` (+ web) |
| **Publish** | успешный Deploy на `main` | `:main` и `:${sha}` |

Образы: **linux/arm64 only**.

Telegram (опционально): secrets `TELEGRAM_TOKEN`, `TELEGRAM_TO` — уведомления после Deploy (тесты / prerelease / ошибки).

```bash
# Prerelease из dev
git commit -m "feat: ... [prerelease]"
git push origin dev
```

GHCR:
- `ghcr.io/shiwarai/video-descriptor-rkllm:main`
- `ghcr.io/shiwarai/video-descriptor-rkllm-web:main`

---

## Атрибуция

- [Qengineering](https://github.com/Qengineering) — Qwen3.5 RK3588 models and reference NPU demos
- [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) — hardware video decode (rkmpp/rkrga)
- Rockchip RKLLM / RKNN runtimes

---

## Лицензия

BSD 3-Clause — см. [LICENSE](LICENSE)

_Проект создан с использованием нейросетей._