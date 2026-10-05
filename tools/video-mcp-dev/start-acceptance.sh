#!/bin/sh
set -eu
exec /acceptance/ninfer-serve \
  /models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer \
  --model-id qwen3.8-27b --host 0.0.0.0 --port 8080 --api-key video-mcp-test \
  --vision --vision-residency overlay --vision-max-merged 256 \
  --local-media-root /videos --local-video-max-tokens 2048 \
  --max-context 8192 --no-thinking --default-max-tokens 1024 --device-profile off \
  --enable-model-suspend --suspend-snapshot-memory pageable \
  --log-level debug --request-log-jsonl /acceptance/requests.jsonl --log-stats-interval-ms 0
