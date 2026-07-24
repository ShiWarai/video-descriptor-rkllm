# Model weights (not in git)

Place RK3588 NPU binaries here or mount this directory into the container at `/app/models`.

## Expected files

| Model | Vision (RKNN) | LLM (RKLLM) |
|-------|---------------|-------------|
| Qwen3.5-0.8B | `qwen3.5-0.8b_vision_rk3588.rknn` | `qwen3.5-0.8b_w8a8_rk3588.rkllm` |
| Qwen3.5-2B | `qwen3.5-2b_vision_rk3588.rknn` | `qwen3.5-2b_w8a8_rk3588.rkllm` |

## Download from Hugging Face

- [Qengineering/Qwen3.5-0.8B-rk3588](https://huggingface.co/Qengineering/Qwen3.5-0.8B-rk3588)
- [Qengineering/Qwen3.5-2B-rk3588](https://huggingface.co/Qengineering/Qwen3.5-2B-rk3588)

On Hugging Face, vision files use a hyphen (`…-vision_rk3588.rknn`). This project expects an underscore (`…_vision_rk3588.rknn`) — rename after download or use the script below.

## Automatic download (opt-in)

Set `VLM_DOWNLOAD_MODELS=1` when starting the API container. The entrypoint checks for missing files and downloads only what is absent:

```bash
VLM_DOWNLOAD_MODELS=1 docker compose up -d api
```

Or run the script manually:

```bash
MODELS_DIR=./models ./scripts/download_models.sh
```

Without `VLM_DOWNLOAD_MODELS=1`, the container starts immediately and expects weights to already be present in `./models`.

## Requirements

- Rockchip RK3588 / RK3588S with RKNPU2 driver
- RKLLM runtime 1.3.0+ and RKNN runtime (bundled in `third_party/lib` for Docker)
