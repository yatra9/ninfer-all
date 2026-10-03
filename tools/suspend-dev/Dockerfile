# syntax=docker/dockerfile:1
FROM ninfer-all:build

# The product build keeps its Ninja tree in this BuildKit cache, not in the image.
# Import only the current configuration and keep the original absolute paths.
RUN --mount=type=cache,id=ninfer-build,target=/baseline-cache,sharing=locked \
    test -s /baseline-cache/configuration \
    && baseline_key="$(sha256sum /baseline-cache/configuration | cut -d ' ' -f 1)" \
    && test -s "/baseline-cache/$baseline_key/CMakeCache.txt" \
    && mkdir -p /build \
    && cp -a /baseline-cache/src /baseline-cache/configuration /build/ \
    && cp -a "/baseline-cache/$baseline_key" "/build/$baseline_key" \
    && printf '%s\n' "/build/$baseline_key" > /build/suspend-build-dir

# BUILD_TESTING requires the Python interpreter for CMake test registration.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G
WORKDIR /build/src
CMD ["sleep", "infinity"]
