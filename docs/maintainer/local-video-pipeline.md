# Local video pipeline

This reference owns the design boundary for the `ninfer-video://` serving extension. The feature
accepts a server-local video selected by an explicit root policy, preserves source-frame timing,
and supplies Qwen Vision in bounded temporal chunks. The external behavior and parameter contract
are specified in the workspace `../SPEC.md`; this file records how that contract fits NInfer's
current frontend, Program, and Vision execution ownership.

The canonical CPU decoder/filter implementation is `src/media/local_video/video_pipeline.*` and
the Linux/WSLC build exposes it as the `ninfer_local_video` static-library target. The workspace
`video-lab` CLI links those same sources from the workspace-root Docker context, so its regression
suite exercises the product implementation rather than a copy. Native Windows support remains
outside the initial WSLC scope.

## Existing Vision semantics

For the supported Qwen3.5/3.8 Vision configuration, one raw patch contains two temporal frames and
one 16x16 spatial patch. Four raw patches (a 2x2 spatial merge) produce one prompt Vision token.
The ordinary-media frontend prepares one immutable BF16 patch payload per media item and retains
each payload until its Vision item has encoded. Local video uses the bounded chunk path below.

One video item has a grid `[temporal_groups, patch_rows, patch_columns]`. `build_vision_control`
sets:

```text
segment_length = patch_rows * patch_columns
segment_count  = temporal_groups
```

Every Vision transformer layer calls packed attention with that segment length. Attention is
therefore confined to one temporal group; different two-frame groups do not attend to one another.
Spatial position IDs also restart with the same two-dimensional grid for each group. Patch
projection, layer norms, MLPs, and the spatial merger are row-local outside that segmented
attention. These facts make a contiguous range of complete temporal groups an exact execution
unit, subject to the same BF16 operations and group order.

Ordinary eager media is still encoded as one item before its columns are scattered into text
prefill chunks. Local video takes the chunk path below, so its BF16 payload and Vision workspace
are bounded by `--vision-max-merged` even when the logical video is much larger.

## Logical item and execution chunk

A local video remains one logical `VisionItem`. Its prompt placeholders, timestamps, token spans,
MRoPE positions, content identity, and request budget describe the complete selected video.
Execution divides that item into `LocalVideoChunkPlan` values containing contiguous, nonempty
temporal-group ranges. It must not expose those ranges as separate chat media items: doing so would
add item markers, change prompt identity, and alter cache frontiers.

For aligned output width `W`, height `H`, spatial merge `M`, and patch size `P`:

```text
tokens_per_group = (W / (P * M)) * (H / (P * M))
groups_per_chunk = floor(execution_merged_token_limit / tokens_per_group)
```

`groups_per_chunk` must be at least one. The final chunk may be smaller. An odd final selected
frame is duplicated once at the end of the logical video, matching the current processor. A chunk
boundary never creates padding. The dimensions and chunk capacity are derived for each video from
its crop and scale result; 1024x768 is only one example.

## Ownership and lifetime

Product/serving owns URL parsing, canonical path/root authorization, source-cache lookup, request
deadline, and cancellation. It creates an owning, typed local-video input before ordinary byte
acquisition. Model code never parses the URL or opens an unvalidated arbitrary path.

`product/local_video/local_video_url.*` implements the first part of that boundary. It recognizes
only the exact `ninfer-video://` prefix, parses every supported query field once into
`LocalVideoSpec`, rejects unknown and duplicate fields, and rejects malformed encoding and numeric
values. `authorize_local_path` requires an explicit media root and uses canonical component paths
to reject traversal, sibling-prefix matches, and symlink escapes before a decoder opens the file.
`--local-media-root` now supplies that explicit process policy as an absolute container path; its
empty default keeps `ninfer-video` disabled.

Chat Completions `video_url` and Responses `input_video.video_url` now classify the custom scheme
as `SourceKind::LocalVideo`. Generation acquisition parses and authorizes it before byte
acquisition, then stores the canonical path and resolved selection controls in
`OwnedMedia::local_video`. Ordinary HTTP(S) and data inputs retain the byte-backed member. The
frontend materializer consumes this alternative without converting it back to encoded bytes.

Frontend owns Qwen geometry, temporal pairing, timestamps, placeholder layout, total token/context
budget, and the immutable logical video plan. It computes all prompt-visible metadata before
execution. The plan contains no decoder and no decoded pixel or patch history.

Program owns the request's mutable materializer/reader and its current chunk. It requests a chunk
only when the Vision session can consume it. The materializer owns decoder, seek, deinterlacer,
resize, RGB conversion, normalize, and patchify state. The next chunk is not decoded speculatively
in the first implementation. A reader is never shared by requests, even when they share an index.

Vision execution owns device workspace, overlay windows, encode streams, and the current chunk's
merged embedding handoff. It consumes the same complete temporal groups described by the logical
item and scatters their columns to the original global prompt spans. Global prompt positions and
scatter indices are not rebased at chunk boundaries.

The host lifetime is bounded by decoder/filter history, one reader lookahead frame, selected frames
for the current chunk, and its BF16 patches. A chunk's BF16 patches may be released after encoding.
Its merged embeddings remain until every text prefill chunk that references them has completed.

## Implemented typed boundary

The implementation uses an explicit alternative to an eager `PreparedMediaPayload`:

```cpp
struct PreparedLocalVideoInput; // immutable whole-item metadata and authorized source handle
struct LocalVideoChunkPlan;     // temporal begin/count and corresponding global spans

class LocalVideoPayloadReader {
public:
    std::shared_ptr<PreparedMediaPayload> read_chunk(std::size_t chunk_index);
};
```

`PreparedPromptData::local_videos` is indexed one-to-one with `vision_items`; an ordinary eager
item has a null local-video slot, while a local video has a typed plan and a null eager payload.
Request planning validates total item metadata and each chunk extent. `VisionPrefillSession`
creates one request-owned reader, obtains each payload in plan order, encodes it, and releases it
with the existing request cleanup and exception guarantees. Each payload reserves bytes from the
shared media live-memory account until its final reference is released. Cancellation and the
request deadline are checked during indexing, decode, live-memory waits, and patchification row
boundaries. Source identity is checked before and after every chunk returned to execution.

The current `VisionItemControl` is item-wide. Chunk execution needs a control view containing the
chunk's group count, patch range, position arrays, merged count, and the original global scatter
indices. `slice_vision_control` provides this checked, contiguous temporal-group view. It narrows
patch and position storage while retaining global scatter indices and does not change the logical
`VisionItem` stored for prompt identity.

## Timing and identity

Every selected frame carries its original display index, PTS, time base, and seconds timestamp.
For each two-frame group, the prompt timestamp is the mean of the two exact source timestamps. CFR
may produce those timestamps from exact frame timing; VFR always uses decoded PTS. Timestamps do not
restart at the selected range or at a chunk boundary.

`VideoSource::plan` exposes this selection metadata and the crop/scale/alignment output geometry
directly from the reusable complete index. It enforces the same selection and resource limits as
the reader and performs no RGB conversion. The frontend transforms this result into model-specific
temporal groups and token counts before execution.

The frontend keeps up to eight `VideoSource` entries in an LRU cache keyed by canonical path,
file size, and modification time. An unchanged path reuses its completed frame index across
requests. A changed file publishes a new source; active requests retain the old source and its
unchanged checks reject mutation safely. Eviction drops only the cache reference, so it cannot
invalidate an active reader. A failed or cancelled scan is not published by `VideoSource`.
This cache is independent of prompt prefix reuse. The first implementation marks local-video
prompt identity non-reusable, because path and modification metadata are not a content digest and
pixels are materialized after prompt preparation. A later strong content identity requires a
separate contract and validation.

## Limits and failure

The existing `vision_max_merged_tokens` remains an execution-chunk envelope. A separate local-video
total-token limit bounds the complete logical item/request. Total Vision tokens, text tokens,
timestamp serialization, requested output allowance, frame/pixel limits, and context capacity are
validated before admitting execution whenever the index makes them known.

Errors from indexing, decode, filtering, materialization, Vision encode, deadline, cancellation, or
file mutation poison the request. No partial prompt is published as success. Reader and provider
destruction stops background work before releasing request-owned state. Overlay restoration and
the existing Program transaction cleanup remain authoritative for GPU failure recovery.

## Qualification evidence

The CPU payload regression concatenates every chunk from a deterministic H.264 video and compares
all BF16 patch elements exactly with an independently decoded full payload. Runtime control tests
compare prompt token IDs, token types, positions, timestamps, patch ranges, and global scatter
columns exactly. A shared live-memory regression holds one request's chunk while a second request
materializes the same chunk under a one-chunk capacity: the second waits, proceeds after release,
and returns the account to zero rather than exceeding the configured capacity.

On the RTX 3090 target, a real Qwen3.8-27B Vision tower encoded the same deterministic four-group
patch payload once as a complete item and once as 1+2+1 temporal-group slices. The direct 81,920
BF16-element embedding comparison measured overall cosine similarity 0.999908 and RMSE 0.0133091;
the worst individual token measured cosine 0.999886 and normalized RMSE 0.028944. The standalone
regression requires both overall and per-token cosine to remain at least 0.999, overall absolute
RMSE at most 0.03, and per-token RMSE normalized by the reference token RMS at most 0.05. It rejects non-finite
outputs or metrics. Its temporal groups use distinct hashed inputs with the same `[-2,2]`
distribution, so repeated or reordered groups cannot pass through periodic input. Exact BF16
equality is not required; the measured difference is retained as empirical qualification rather
than attributed to a particular CUDA route without route-level evidence.

The same 256-token video was also executed through overlay Vision once with a 256-token envelope
and once as four 64-token chunks. The greedy first token matched, its logprob differed by 0.07572,
and 19 of the top 20 alternatives were shared. A 1024x768, 256-frame input completed as 98,304
Vision tokens in seven chunks. With the 163,840-token RTX 3090 launch profile, process RSS was
12,897,168 kB before that request and reached a 13,259,916 kB high-water mark: a 362,748 kB
(354.25 MiB) request-time increase. RSS after completion was 12,970,668 kB, 73,500 kB (71.78 MiB)
above the baseline. The HTTP 200 response completed in 100.41 seconds with a 99,403-token prompt.
A client disconnect during the same workload cancelled the request and the next media request
completed, demonstrating request cleanup and overlay restoration.
