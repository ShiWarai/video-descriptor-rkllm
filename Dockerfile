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
        libdrm-dev \
        libmp3lame-dev \
        libopus-dev \
        libssl-dev \
        libvorbis-dev \
        zlib1g-dev \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt config.json ./
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY third_party ./third_party

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" --target vlm_api_server \
    && strip --strip-unneeded \
        build/vlm_api_server \
        third_party/lib/*.so \
        third_party/ffmpeg-rockchip/lib/*.so*

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libdrm2 \
        libgomp1 \
        liblzma5 \
        libopus0 \
        libmp3lame0 \
        libvorbis0a \
        libvorbisenc2 \
        libssl3t64 \
        zlib1g \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/vlm_api_server /app/build/vlm_api_server
COPY --from=builder /app/third_party/lib /app/third_party/lib
COPY --from=builder /app/third_party/ffmpeg-rockchip/lib /app/third_party/ffmpeg-rockchip/lib
COPY config.json /app/config.json
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
COPY scripts/download_models.sh /app/scripts/download_models.sh
COPY scripts/fix_freq_rk3588.sh /app/scripts/fix_freq_rk3588.sh

RUN chmod +x /app/scripts/docker-entrypoint.sh /app/scripts/download_models.sh \
    /app/scripts/fix_freq_rk3588.sh \
    && mkdir -p /app/models \
    && cd /app/third_party/ffmpeg-rockchip/lib \
    && if [ -f librockchip_mpp.so.1 ]; then \
         rm -f librockchip_mpp.so librockchip_mpp.so.0; \
         ln -sf librockchip_mpp.so.1 librockchip_mpp.so.0; \
         ln -sf librockchip_mpp.so.1 librockchip_mpp.so; \
       fi

ENV LD_LIBRARY_PATH=/app/third_party/lib:/app/third_party/ffmpeg-rockchip/lib
ENV PATH=/app/build:${PATH}

EXPOSE 8080

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
