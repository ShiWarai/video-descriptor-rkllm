# ffmpeg-rockchip (vendored)

Prebuilt **libav** libraries and headers for RK3588 (rkmpp/rkrga decode in-process).

| Field | Value |
|-------|-------|
| Upstream | [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) |
| Platform | linux/arm64 (RK3588) |

## Contents

- `include/` — vendored libav* headers (pinned `.so` versions)
- `lib/libav*.so*`, `lib/libsw*.so*`, `lib/librga.so*`, `lib/librockchip_mpp.so*` — runtime for linked binaries

CLI `ffmpeg`/`ffprobe` are not vendored (not used by this project).

## In this project

| Path | Use |
|------|-----|
| `src/core/frame_extractor.cpp` | Video/GIF frames: libav + rkmpp/rkrga |
| `src/core/audio_extractor.cpp` | ASR audio: libav demux/decode/swr → WAV in RAM |

No CLI `ffmpeg`/`ffprobe` in the API pipeline or Docker runtime image.

Host/container apt deps: `libdrm2`, `zlib1g`, `libopus0`, `libmp3lame0`, `libvorbis0a`, `libvorbisenc2`, `libssl3t64`.

Builder image additionally needs `-dev` packages for linking: `libdrm-dev`, `libopus-dev`, `libmp3lame-dev`, `libvorbis-dev`, `libssl-dev`.

## Rebuild (host)

See upstream [ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip). After `make install`, copy `.so` and headers into `lib/` and `include/` as documented in upstream build notes.
