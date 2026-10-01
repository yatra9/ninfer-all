FROM ubuntu:24.04 AS dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev \
    libavfilter-dev libswscale-dev libpng-dev nlohmann-json3-dev ffmpeg python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/ninfer-all
COPY src/media/local_video src/media/local_video
COPY tools/video-lab tools/video-lab
RUN cmake -S tools/video-lab -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build /build --parallel \
    && ctest --test-dir /build --output-on-failure
ENTRYPOINT ["/build/video-lab"]
