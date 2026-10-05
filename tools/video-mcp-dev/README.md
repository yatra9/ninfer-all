# Video MCP development with WSLC

Run from the repository root in an elevated/authorized WSLC execution context:

```powershell
wslc build -f tools/video-mcp-dev/Dockerfile -t ninfer-all:dev .
wslc run -d --name ninfer-all-dev --gpus all `
  -v "${PWD}:/workspace:ro" -v ninfer-all-dev-ccache:/ccache `
  -p 127.0.0.1:18080:8080 ninfer-all:dev
wslc exec ninfer-all-dev sh /workspace/tools/video-mcp-dev/prepare-build.sh
```

The Dockerfile imports the current `ninfer-build` BuildKit cache into `/build`,
preserving the original Ninja tree's paths. It fails if the baseline cache is
missing; it does not silently rebuild CUDA. The host checkout stays read-only.
The preparation script uses checksum synchronization to preserve unchanged source
mtimes. Build targets incrementally in the directory recorded by
`/build/video-mcp-build-dir`; never replace that directory with a fresh build tree.

FFmpeg's CLI is used only to generate/check test fixtures. Production tools use
the shared libav implementation. The image's Python interpreter is for CMake test
registration and fixture scripts; inspect its version before using it.

Model/video mounts and a free GPU are required only for inference E2E. Pass local
video paths visible inside the container, and configure `--local-media-root` to
their authorized parent directory. The published port is restricted to host
localhost for the Windows OpenCode client.

## Real-model acceptance

The selected model is copied into the native `ninfer-video-mcp-models` volume and
checked against its distributed `SHA256SUMS`. Keep the original Windows artifact.
The acceptance container uses the same `ninfer-all:dev` image with the model volume
read-only at `/models`, the repository read-only at `/workspace`, and localhost port
18081 mapped to 8080. `start-acceptance.sh` records the exact model and serve options.
It enables model suspend so `acceptance.py` can verify CPU-only tools while suspended.

The development container can temporarily expose its built executable with Python:

```sh
nohup python3 -m http.server 8080 --bind 0.0.0.0 \
  --directory "$(cat /build/video-mcp-build-dir)/apps" >/tmp/video-mcp-build-http.log 2>&1 </dev/null &
```

Run `prepare-acceptance.sh http://<dev-container-ip>:8080/ninfer-serve` inside the
acceptance container, then start the server using:

```sh
nohup sh /workspace/tools/video-mcp-dev/start-acceptance.sh \
  >/acceptance/server.log 2>&1 </dev/null & echo $! >/acceptance/server.pid
python3 /workspace/tools/video-mcp-dev/acceptance.py
```

`acceptance.py` verifies API-key protection, initialize/discovery, two resolves,
inspect → native video inference, shared source/index reuse in debug logs, all three
tools while suspended, resume and the existing Responses endpoint. Its report is
`/acceptance/http-report.json`. Debug paths and test credentials are fixture-only.

It also verifies `frame=0/2/3` as image inputs (with crop and scale), red/blue answers,
API rejection of missing frames and combined range controls, and single-frame MCP calls
while suspended. For OpenCode single-frame acceptance, pass a prompt requiring
`inspect_video` with `frame=2` and verify its unchanged URI reaches the next model request.

## OpenCode acceptance

Use `run-opencode.ps1` with the user-selected executable and `opencode.example.json`.
The script isolates OpenCode data/config/state/cache beneath `.cache/video-mcp` and
restricts the acceptance agent to these MCP tools. The neutral-name fixture has red
frames 0–1 and blue frames 2–3. Check the exact selected range, not the filename.

For wire evidence, `recording-proxy.py` is an acceptance-only observer. Run it in a
separate WSLC container with `--upstream http://<acceptance-container-ip>:8080`, publish
its port 8080 at host localhost:18082, and change both URLs in the example config to
18082. It forwards auth/session headers and records `/tmp/video-mcp-wire.jsonl` without
recording prompts or credentials. The production MCP remains inside NInfer's listener.

Run `check-opencode.py --wire <wire.jsonl> --events <opencode-events.jsonl> --report
<report.json>` with a checked Python interpreter. It fails unless a returned resource
URI appears as `video_url` in a later model request from the same agent session and the
answer contains the expected color. A plausible answer without media does not pass.

The initial executable lacked the applied patch and discarded resource links. After the user applied the patch and rebuilt it, both red and blue frame selections passed the same-agent wire/session checks. Normal execution selected native without the opt-in flag. See ACCEPTANCE.md for executable fingerprints and results.
