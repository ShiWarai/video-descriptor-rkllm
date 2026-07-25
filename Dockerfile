# syntax=docker/dockerfile:1

# Minimal OpenCV (core+imgproc only) — Ubuntu packages pull Mesa/LLVM (~300MB+).
FROM ubuntu:24.04 AS opencv

ENV DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.6.0

RUN apt-get update && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        g++ \
        pkg-config \
        wget \
        zlib1g-dev \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN wget -q "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
        -O opencv.tar.gz \
    && tar -xzf opencv.tar.gz \
    && cmake -S "opencv-${OPENCV_VERSION}" -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv \
        -DBUILD_LIST=core,imgproc \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python3=OFF \
        -DWITH_FFMPEG=OFF \
        -DWITH_GSTREAMER=OFF \
        -DWITH_GTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_OPENGL=OFF \
        -DWITH_IPP=OFF \
        -DWITH_TBB=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_OPENCL=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_V4L=OFF \
        -DWITH_JPEG=OFF \
        -DWITH_PNG=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_OPENJPEG=OFF \
        -DWITH_JASPER=OFF \
        -DWITH_1394=OFF \
        -DCPU_BASELINE="" \
    && cmake --build build -j"$(nproc)" \
    && cmake --install build \
    && strip --strip-unneeded /opt/opencv/lib/libopencv_*.so* \
    && rm -rf /tmp/opencv.tar.gz /tmp/opencv-${OPENCV_VERSION} /tmp/build

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_PREFIX_PATH=/opt/opencv
ENV LD_LIBRARY_PATH=/opt/opencv/lib

RUN apt-get update && \
    for attempt in 1 2 3 4 5; do \
      apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        g++ \
        pkg-config \
        zlib1g-dev \
      && break; \
      echo "apt install failed (attempt ${attempt}), retrying..."; \
      sleep 15; \
      apt-get update; \
    done \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv /opt/opencv /opt/opencv

WORKDIR /app
COPY CMakeLists.txt config.json ./
COPY include ./include
COPY src ./src
COPY tests ./tests
COPY third_party ./third_party

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/opencv \
    && cmake --build build -j"$(nproc)" \
    && strip --strip-unneeded \
        build/vlm_api_server \
        build/VLM_VIDEO_NPU \
        third_party/lib/*.so \
        third_party/ffmpeg-rockchip/bin/ffmpeg \
        third_party/ffmpeg-rockchip/bin/ffprobe \
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
COPY --from=opencv /opt/opencv/lib /app/third_party/opencv/lib
COPY --from=builder /app/third_party/lib /app/third_party/lib
COPY --from=builder /app/third_party/ffmpeg-rockchip/bin /app/third_party/ffmpeg-rockchip/bin
COPY --from=builder /app/third_party/ffmpeg-rockchip/lib /app/third_party/ffmpeg-rockchip/lib
COPY config.json /app/config.json
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
COPY scripts/download_models.sh /app/scripts/download_models.sh

RUN chmod +x /app/scripts/docker-entrypoint.sh /app/scripts/download_models.sh \
    && mkdir -p /app/models \
    && cd /app/third_party/ffmpeg-rockchip/lib \
    && if [ -f librockchip_mpp.so.1 ]; then \
         rm -f librockchip_mpp.so librockchip_mpp.so.0; \
         ln -sf librockchip_mpp.so.1 librockchip_mpp.so.0; \
         ln -sf librockchip_mpp.so.1 librockchip_mpp.so; \
       fi

ENV LD_LIBRARY_PATH=/app/third_party/lib:/app/third_party/opencv/lib:/app/third_party/ffmpeg-rockchip/lib
ENV PATH=/app/third_party/ffmpeg-rockchip/bin:/app/build:${PATH}

EXPOSE 8080

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
