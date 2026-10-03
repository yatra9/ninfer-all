#!/bin/sh
set -eu
build_dir=$(cat /build/suspend-build-dir)
test -s "$build_dir/CMakeCache.txt"
log=/tmp/ninfer-suspend-build.log
if cmake --build "$build_dir" -j8 \
    --target ninfer_model_suspend_test ninfer_qwen3_5_loading_test > "$log" 2>&1; then
    printf 'Suspend test build completed. Log: %s\n' "$log"
else
    tail -n 80 "$log"
    exit 1
fi
