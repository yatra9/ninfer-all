#!/bin/sh
set -eu

build_dir=$(cat /build/suspend-build-dir)
case "$build_dir" in
    /build/*) key=${build_dir#/build/} ;;
    *) echo "unexpected baseline build path: $build_dir" >&2; exit 1 ;;
esac
case "$key" in
    ''|*[!0-9a-f]*) echo "invalid baseline configuration key" >&2; exit 1 ;;
esac
test "${#key}" -eq 64
test -s "$build_dir/CMakeCache.txt"
test -f /workspace/CMakeLists.txt

# Unchanged files keep the baseline mtimes so Ninja reuses their object files.
# /build/src is the imported container source tree, never a host bind mount.
rsync --recursive --links --checksum --delete \
    --exclude=/.git --exclude=/.cache --exclude=/.codex --exclude=/.claude \
    --exclude=/build --exclude='/build-*' --exclude='/cmake-build-*' \
    --exclude=/out --exclude=/models --exclude=/profiles --exclude=/logs \
    --exclude='**/__pycache__' --exclude='**/.venv' --exclude='**/venv' \
    /workspace/ /build/src/

cmake -S /build/src -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF
printf 'Prepared incremental build: %s\n' "$build_dir"
