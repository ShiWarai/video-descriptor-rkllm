# ffmpeg-rockchip (vendored)

Prebuilt binaries for RK3588 hardware decode (rkmpp/rkrga).

| Field | Value |
|-------|-------|
| Upstream | [nyanmisaka/ffmpeg-rockchip](https://github.com/nyanmisaka/ffmpeg-rockchip) |
| Commit | `388741a` (host build in `/usr/local`) |
| Platform | linux/arm64 (RK3588) |

## Contents

- `bin/ffmpeg`, `bin/ffprobe` — from `/usr/local/bin`
- `lib/libav*.so*`, `lib/libsw*.so*` — FFmpeg shared libraries
- `lib/librga.so*`, `lib/librockchip_mpp.so*` — Rockchip MPP/RGA runtime

`libdrm` and `zlib` are expected from the host/container via apt.
