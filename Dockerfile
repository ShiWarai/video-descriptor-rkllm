# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        g++ \
        pkg-config \
        libopencv-dev \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libdrm2 \
        libgomp1 \
        libopus0 \
        libmp3lame0 \
        libvorbis0a \
        libvorbisenc2 \
        libssl3t64 \
        libopencv-core406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-imgproc406t64 \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/vlm_api_server /app/build/vlm_api_server
COPY --from=builder /app/build/VLM_VIDEO_NPU /app/build/VLM_VIDEO_NPU
COPY third_party/lib /app/third_party/lib
COPY third_party/ffmpeg-rockchip /app/third_party/ffmpeg-rockchip
COPY config.json /app/config.json
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
COPY scripts/download_models.sh /app/scripts/download_models.sh

RUN chmod +x /app/scripts/docker-entrypoint.sh /app/scripts/download_models.sh \
    && mkdir -p /app/models

ENV LD_LIBRARY_PATH=/app/third_party/lib:/app/third_party/ffmpeg-rockchip/lib
ENV PATH=/app/third_party/ffmpeg-rockchip/bin:/app/build:${PATH}

EXPOSE 8080

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
