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
