# syntax=docker/dockerfile:1
FROM ninfer-all:build

# Ninja objects live in BuildKit's cache, not in the parent image's layers.
# Preserve the original absolute paths and source mtimes for incremental builds.
RUN --mount=type=cache,id=ninfer-build,target=/baseline-cache,sharing=locked \
    test -s /baseline-cache/configuration \
    && baseline_key="$(sha256sum /baseline-cache/configuration | cut -d ' ' -f 1)" \
    && test -s "/baseline-cache/$baseline_key/CMakeCache.txt" \
    && mkdir -p /build \
    && cp -a /baseline-cache/src /baseline-cache/configuration /build/ \
    && cp -a "/baseline-cache/$baseline_key" "/build/$baseline_key" \
    && printf '%s\n' "/build/$baseline_key" > /build/video-mcp-build-dir

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G
WORKDIR /build/src
CMD ["sleep", "infinity"]
