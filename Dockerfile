# syntax=docker/dockerfile:1

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        g++ \
        libdrm-dev \
        libgrpc++-dev \
        libmp3lame-dev \
        libopus-dev \
        libprotobuf-dev \
        libssl-dev \
        libvorbis-dev \
        pkg-config \
        protobuf-compiler \
        protobuf-compiler-grpc \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt config.json ./
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY third_party ./third_party
COPY proto ./proto

# grpc_gen is gitignored; always regenerate with container protoc (3.21+ on Ubuntu 24.04).
RUN rm -rf src/grpc_gen \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" --target vlm_api_server vlm_vision_worker vlm_llm_worker \
    && strip --strip-unneeded \
        build/vlm_api_server \
        build/vlm_vision_worker \
        build/vlm_llm_worker \
        third_party/lib/*.so \
        third_party/ffmpeg-rockchip/lib/*.so*

FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        util-linux \
        libdrm2 \
        libgomp1 \
        libgrpc++1.51t64 \
        liblzma5 \
        libopus0 \
        libmp3lame0 \
        libprotobuf32t64 \
        libvorbis0a \
        libvorbisenc2 \
        libssl3t64 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/vlm_api_server /app/build/vlm_api_server
COPY --from=builder /app/build/vlm_vision_worker /app/build/vlm_vision_worker
COPY --from=builder /app/build/vlm_llm_worker /app/build/vlm_llm_worker
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

EXPOSE 8080 50051 50052

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
