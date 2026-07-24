# Model weights (not in git)

Place RK3588 NPU binaries here or mount this directory into the container at `/app/models`.

## Expected files

| Model | Vision (RKNN) | LLM (RKLLM) |
|-------|---------------|-------------|
| Qwen3.5-0.8B | `qwen3.5-0.8b_vision_rk3588.rknn` | `qwen3.5-0.8b_w8a8_rk3588.rkllm` |
| Qwen3.5-2B | `qwen3.5-2b_vision_rk3588.rknn` | `qwen3.5-2b_w8a8_rk3588.rkllm` |

## Download from Hugging Face

- [Qwen3.5-0.8B-rk3588](https://huggingface.co/Qengineering/Qwen3.5-0.8B-rk3588)
- [Qwen3.5-2B-rk3588](https://huggingface.co/Qengineering/Qwen3.5-2B-rk3588)

On Hugging Face, filenames match `config.json` (e.g. `qwen3.5-0.8b_vision_rk3588.rknn`).

## Automatic download (opt-in)

Set `VLM_DOWNLOAD_MODELS` when starting the API container. The entrypoint downloads **only** the selected weights (skips files that already exist).

| `VLM_DOWNLOAD_MODELS` | Что качается |
|-----------------------|--------------|
| `0` / не задано | ничего |
| `0.8b` | только 0.8B (vision + llm) |
| `2b` | только 2B |
| `0.8b,2b` или `all` / `1` | обе модели |

```bash
VLM_DOWNLOAD_MODELS=0.8b docker compose up -d api
```

### Свой URL (произвольный адрес файла)

`VLM_MODEL_URLS` — список через запятую: `имя_файла_в_models=URL`

```bash
VLM_DOWNLOAD_MODELS=0.8b \
VLM_MODEL_URLS="qwen3.5-0.8b_w8a8_rk3588.rkllm=https://huggingface.co/Qengineering/Qwen3.5-0.8B-rk3588/resolve/main/qwen3.5-0.8b_w8a8_rk3588.rkllm" \
docker compose up -d api
```

Можно комбинировать: пресет `0.8b` + дополнительные пары в `VLM_MODEL_URLS`. Для каталога HF по умолчанию: `VLM_HF_BASE=https://huggingface.co/Qengineering`.

Ручной запуск:

```bash
MODELS_DIR=./models ./scripts/download_models.sh 0.8b
```

## Requirements

- Rockchip RK3588 / RK3588S with RKNPU2 driver
- RKLLM runtime 1.3.0+ and RKNN runtime (bundled in `third_party/lib` for Docker)
