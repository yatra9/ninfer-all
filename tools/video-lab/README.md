# video-lab

`video-lab` is a CPU-only FFmpeg CLI for validating NInfer's local-video
pipeline. Its canonical reusable source lives in
[`src/media/local_video`](../../src/media/local_video); the CLI and NInfer build the same file. It
opens a local file directly; it does not load the compressed file into memory or invoke `ffmpeg` as
a subprocess.

The reusable `video_pipeline` library owns probing, display-frame indexing,
keyframe seek, exact frame selection, PTS timestamps, `bwdif` deinterlacing,
crop, resize and RGB conversion. The CLI adds JSON, SHA-256 and PNG output.
`VideoSource` retains a completed timing index for reuse, while each
`VideoReader` retains its decoder and filter state across chunk reads.
`VideoSource::plan()` derives selected source indices, exact PTS timestamps and
output geometry from that index without retaining RGB frames or decoding them a second time.

## Build and test with WSLC

```powershell
cd E:\koji\work\20260813\NInfer\ninfer-all
wslc build -f .\tools\video-lab\Dockerfile -t video-lab:dev .
```

The image build runs the integration test suite. It uses generated H.264 CFR,
H.264 VFR and interlaced fixtures and checks sequential decode against indexed
seek, including B-frames.

## Use

Mount a directory containing the input as read-only and a separate output
directory as writable. Paths passed to the command are container paths.

```powershell
wslc run --rm `
  -v C:\videos:/videos:ro `
  -v C:\video-lab-output:/out `
  video-lab:dev probe /videos/example.mp4

wslc run --rm `
  -v C:\videos:/videos:ro `
  -v C:\video-lab-output:/out `
  video-lab:dev extract /videos/example.mp4 `
  --output /out/run-001 `
  --start-frame 100 --end-frame 1000 --skip-frame 2 `
  --bbox 320,180,640,360 --scale 1.5 --align 32 `
  --deinterlace auto

wslc run --rm `
  -v C:\videos:/videos:ro `
  video-lab:dev verify /videos/example.mp4 `
  --start-frame 100 --end-frame 1000 --skip-frame 2 `
  --bbox 320,180,640,360 --scale 1.5 --align 32
```

`extract` requires a new output directory and writes `manifest.json` plus one
PNG per selected frame. `--no-images` still runs the complete pixel pipeline and
records the RGB SHA-256 without writing PNGs. `verify` processes the request once
from frame zero and once through the index/seek path, then compares source index,
PTS, timestamp, geometry and RGB hash exactly.

Run `video-lab:dev --help` for all options.

## Processing contract

- Frame indices refer to decoded display order and start at zero.
- `end_frame` is inclusive. `skip_frame=N` selects every `N+1` source frames,
  anchored at `start_frame`.
- Timestamps come from each frame's FFmpeg best-effort PTS and stream time base,
  relative to the first source frame's PTS. A selected range retains that source origin.
  The library refuses to synthesize missing timestamps from average FPS.
- Indexed mode first builds a lightweight PTS/keyframe index. Before publishing
  output, it verifies that decode from the selected random-access point reaches
  the requested frame exactly. H.264 open GOPs and other unsafe seek points fall
  back to sequential decode. The index contains no image data. Indexing and
  processing durations are reported separately.
- A `VideoSource` builds its full index lazily on the first indexed reader and
  reuses it for later readers. A cancelled, failed or file-invalidated scan is
  never published. `SourceStats` exposes build, scanned-frame and reuse counts.
- Deinterlacing uses FFmpeg `bwdif` in `send_frame` mode, so one source frame
  remains one output frame. `auto` follows decoded frame metadata; progressive
  frames in a mixed stream pass through unchanged, and progressive-only inputs
  do not instantiate the filter.
- Crop coordinates are exact source-pixel coordinates. The current reference
  implementation converts the full decoded frame to RGB before crop so odd
  chroma-aligned coordinates remain exact. YUV ROI conversion is a later measured
  optimization.
- Resize uses bicubic interpolation. Requested dimensions are
  `round(source_dimension * scale)` and then rounded to the nearest alignment
  multiple, with ties rounded upward. The default alignment is 1; use 32 for the
  current Qwen3.8 configuration.
- `VideoReader::read_chunk(N)` returns at most `N` selected frames. Decoder,
  seek, selection and deinterlacer state continue across calls. EOF is stable:
  later calls return an empty EOF chunk. The implementation applies backpressure
  and retains decoder/filter history plus at most the current chunk and one
  produced frame, rather than all selected RGB frames.
- Resource limits, cancellation checkpoints, midstream format changes, corrupt
  frames and input-file changes are explicit errors.
- Display rotation metadata is reported but is not applied. This must be decided
  before NInfer integration because `bbox` coordinates can refer either to coded
  pixels or to display-oriented pixels.

`autotone` is reserved. The CLI accepts only `--autotone 0` or `false`; enabling
it fails explicitly.

## Reuse from C++

Link the `video_pipeline` CMake target and include:

```cpp
#include <media/local_video/video_pipeline.h>

ninfer::media::local_video::Options options;
options.start = 100;
options.end = 1000;
options.skip = 2;

ninfer::media::local_video::VideoSource source("/videos/example.mp4");
auto plan = source.plan(options); // timing and geometry only; reuses the source index
auto reader = source.create_reader(options);
while (true) {
    auto chunk = reader.read_chunk(16);
    for (auto& frame : chunk.frames) {
        // Consume or move frame.rgb.
    }
    if (chunk.eof) break;
}
```

The legacy `process()` callback wrapper remains available. The public library
header has no FFmpeg, JSON, PNG or NInfer dependency.
