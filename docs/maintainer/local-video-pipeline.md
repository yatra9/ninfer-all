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
The frontend currently prepares one immutable BF16 patch payload per media item and retains every
payload until its Vision item has encoded.

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

The current `VisionPrefillSession` nevertheless encodes the whole media item before scattering
columns into text prefill chunks. Text prefill chunking does not bound the item's BF16 patch payload
or Vision encode workspace. Raising the current one-item limit would retain the whole prepared
video and size the encode workspace for it.

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

## Proposed typed boundary

The implementation should introduce an explicit alternative to an eager `PreparedMediaPayload`:

```cpp
struct LocalVideoPlan;          // immutable whole-item metadata and authorized source handle
struct LocalVideoChunkPlan;     // temporal begin/count and corresponding global spans

class LocalVideoMaterializer {
public:
    PreparedMediaPayload materialize(const LocalVideoChunkPlan&,
                                     const PreparationControl&);
};
```

Exact names may follow the surrounding code, but the distinction is required. `PreparedPromptData`
must be able to describe an eager payload or a local-video plan without null pointers standing for
an undocumented state. Request planning validates total item metadata and each chunk extent.
`VisionPrefillSession` obtains a materialized payload through the typed provider, encodes it, and
releases it with the existing request cleanup and exception guarantees.

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

The lightweight frame index may be cached across requests using canonical path plus file identity
and modification facts. A failed or cancelled scan is not published. This cache is independent of
prompt prefix reuse. The first implementation marks local-video prompt identity non-reusable,
because path and modification metadata are not a content digest and pixels are materialized after
prompt preparation. A later strong content identity requires a separate contract and validation.

## Limits and failure

The existing `vision_max_merged_tokens` remains an execution-chunk envelope. A separate local-video
total-token limit bounds the complete logical item/request. Total Vision tokens, text tokens,
timestamp serialization, requested output allowance, frame/pixel limits, and context capacity are
validated before admitting execution whenever the index makes them known.

Errors from indexing, decode, filtering, materialization, Vision encode, deadline, cancellation, or
file mutation poison the request. No partial prompt is published as success. Reader and provider
destruction stops background work before releasing request-owned state. Overlay restoration and
the existing Program transaction cleanup remain authoritative for GPU failure recovery.

## Qualification boundary

The first production change must prove the group-slicing premise before raising limits:

1. Generate one deterministic multi-group BF16 patch payload.
2. Encode it once as one item and again as consecutive complete-group chunks.
3. Concatenate chunk outputs in group order and compare with the one-item output under the existing
   Vision numerical qualification tolerance, for resident and overlay routes used by this fork.
4. Compare prompt token IDs, token types, positions, timestamps, and global scatter columns exactly.

CPU decoder tests prove media selection and preprocessing, but do not prove this Vision equivalence.
The 96Ki target is accepted only after the real RTX 3090 model path also demonstrates bounded host
patch memory, bounded Vision workspace, correct cleanup, and a successful response.
