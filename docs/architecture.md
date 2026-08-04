# Архитектура video-descriptor-rkllm

Проект поддерживает два режима развёртывания:

| Режим | Compose | `VLM_RUNTIME` | Описание |
|-------|---------|---------------|----------|
| **monolith** | `docker-compose.yml` | `local` | Один контейнер: HTTP API + vision RKNN + LLM RKLLM |
| **distributed** | `docker-compose.distributed.yml` / k3s | `distributed` | Gateway (HTTP API) + vision/LLM gRPC workers |

Внешний контракт в обоих режимах — **один** OpenAI-compatible REST entrypoint: `vlm_api_server` (`POST /v1/video/analyze`, `/v1/chat/completions`).

## Единый пайплайн

Local и distributed отличаются только транспортом vision→LLM. Вся логика extract/ASR/prompt — в gateway/monolith:

| Модуль | Назначение |
|--------|------------|
| `pipeline/video_context_pipeline.cpp` | Оркестрация analyze |
| `pipeline/stage_transport.hpp` | `StageTransport` интерфейс |
| `pipeline/local_stage_transport.cpp` | In-process vision + LLM |
| `pipeline/grpc_stage_transport.cpp` | gRPC `EncodeThenGenerate` |
| `pipeline/audio_transcriber.cpp` | HTTP Whisper ASR |

## Распределённый пайплайн

```
Client (HTTP)
    │
    ▼
┌─────────────┐   RGB 448²     ┌──────────────────────────────┐
│  gateway    │ ─────────────► │ frame extract + ASR (CPU)    │
│  (api)      │                └──────────────┬───────────────┘
└──────┬──────┘                               │ prompt + frames
       │ gRPC EncodeThenGenerate               ▼
       │                          ┌───────────────────────┐
       └────────────────────────► │ vision worker × N     │  RKNN NPU
                                  └───────────┬───────────┘
                                              │ embeddings (~1.5 MiB/frame)
                                              │ напрямую в llm worker
                                              ▼
                                  ┌───────────────────────┐
                                  │ llm worker × M        │  RKLLM NPU
                                  └───────────┬───────────┘
                                              │ text
                                              ▼
                                    HTTP response
```

Gateway **не** десериализует embeddings: vision-воркер получает `llm_target` и сам вызывает `LlmService.Generate`.

### Сервисы и роли

| `VLM_ROLE` | Бинарь | Что грузится |
|------------|--------|--------------|
| `all` / `local` | `vlm_api_server` | vision.rknn + llm.rkllm |
| `gateway` | `vlm_api_server` (`VLM_RUNTIME=distributed`) | без моделей; extract + ASR + gRPC |
| `vision` | `vlm_vision_worker` | `.rknn` — один load + `rknn_dup_context` на NPU cores |
| `llm` | `vlm_llm_worker` | `.rkllm` multimodal |

Роль задаётся в `scripts/docker-entrypoint.sh`.

### gRPC контракт

Protobuf: [`proto/vlm/v1/worker.proto`](../proto/vlm/v1/worker.proto)

- `VisionService.Encode`: frames → embeddings (отладка)
- `VisionService.EncodeThenGenerate`: frames + prompt + `llm_target` → text (основной путь)
- `LlmService.Generate`: embeddings + prompt → text
- `Health`: inflight, MemAvailable, model_id

Сгенерированные stubs: `src/grpc_gen/`. Регенерация — через CMake при сборке (`cmake -B build && cmake --build build`).

Балансировка воркеров: least-inflight в `src/grpc/grpc_client.cpp`. Headless k8s Services резолвятся через `expand_targets()`.

## Рекомендация: одна машина (RK3588)

На **одной плате** используйте **monolith** — `docker compose up -d`. Один контейнер, vision pool на все NPU cores, LLM in-process. Distributed на той же машине добавляет gRPC-оверхед и конкуренцию за NPU.

Distributed имеет смысл при **горизонтальном масштабировании** (несколько RK3588-узлов).

## Monolith (один узел)

```bash
docker compose up -d --build
```

## Распределённый Compose (smoke)

```bash
docker network create vlm_rkllm_default
docker compose -f docker-compose.distributed.yml up -d --build
curl -sf http://localhost:8080/ready
```

| Сервис | Назначение |
|--------|------------|
| `gateway` | HTTP API, `VLM_RUNTIME=distributed` |
| `vision-0` | vision gRPC worker (NPU) |
| `llm-0` | LLM gRPC worker (NPU) |
| `web` | Flask UI → gateway |

Переменные gateway: `VISION_TARGETS`, `LLM_TARGETS`, `VLM_RUNTIME=distributed`.

## k3s

Те же роли и gRPC-контракты:

| Компонент | Масштабирование |
|-----------|-----------------|
| gateway | 1 HTTP Service |
| vision-* | **по одному на NPU-узел** |
| llm-* | **M реплик** на NPU-узлах |

```text
VISION_TARGETS=vision-0:50051,vision-1:50051
LLM_TARGETS=llm-0:50052,llm-1:50052
```
