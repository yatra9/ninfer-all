# Video MCP acceptance, 2026-10-06

Status: all v1 acceptance conditions passed after the user rebuilt OpenCode with the patch applied.

Environment: WSLC `ninfer-all:dev` from `ninfer-all:build`, CUDA 13.1, RTX 3090.
The original baseline build directory and compiled model kernels were reused.
Python 3.12.3 is used only for fixtures and acceptance observers.

Selected model: `Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer`, copied from the
user-selected Windows directory into `ninfer-video-mcp-models`. Both distributed
SHA256SUMS entries (model and conversion report) passed.

Passed:

- Ten relevant CPU CTests: source service, URI, MCP HTTP, local-video payload,
  OpenAI/Anthropic schemas, Responses store, MCP proxy, HTTP transport/error handling.
- CFR/VFR/B-frame/audio/interlaced/mixed/nonzero PTS and coarse-time-base MKV fixtures.
- Real HTTP cancellation/retry, scan-time health response and 10/20-second resolves.
- Real NInfer listener: API-key protection for MCP POST/GET/DELETE and `/v1/models`,
  authenticated initialize/discovery, metadata, two resolves, inspect and session DELETE.
- Native video inference through the inspect URI returned `red` for red frames.
- Debug evidence after metadata → resolve → resolve → inspect → native reader:
  metadata probes remained 1; index builds remained 1; index reuses increased to 3;
  decoder opens increased to 3. Discovery and decoder reopening are counted separately.
- All three tools succeeded while the model was suspended, without resuming it.
  Explicit resume and the existing `/v1/responses` endpoint then succeeded.

Initial OpenCode run before patch application (resolved by the retest below):

- Executable: `E:\koji\work\20260813\opencode\opencode.exe`.
- Reported version: `0.0.0--202610051423`.
- SHA256: `4BDF1B83DC9030030D3575DCA643160D7787F72C6A4892284F809E798C60D90D`.
- It connected to MCP and received the correct inspect `resource_link` with
  `video/mp4` and `ninfer-video:///videos/mystery.mp4?start_frame=2&end_frame=3`.
- Default selection logged `llm.runtime=ai-sdk`. Explicit
  `OPENCODE_EXPERIMENTAL_NATIVE_LLM=true` selected `llm.runtime=native`.
- Both paths sent no `video_url` in the next model request. NInfer's request log
  reported no media, and the source index stayed unbuilt for this OpenCode fixture.
- The native run answered blue, but that is insufficient: `check-opencode.py`
  returned `complete=false`, one agent session, one inspect resource link and zero
  matching native video URLs. No visual E2E success is claimed.

Raw wire summaries and the latest OpenCode events/runtime log are in
`.cache/video-mcp/`; HTTP and OpenCode reports remain under `/acceptance` in the
stopped acceptance container. The observer and GPU server were stopped after the
checks. The development container and model volume are retained for continuation.

Specification §19 is complete after the rebuilt-executable retest below.

## Rebuilt executable retest: passed

The user applied the patch and rebuilt the same executable path. Version:
`0.0.0--202610051714`; SHA256:
`52C335CDB867B11523CD144083EEDE0268AA8E2ED4C59E4554E3328C5A8C026F`.

Normal execution selected the native runtime without an explicit opt-in flag.
Each run used one agent session and one inspect call. Its completed tool state
contained the file attachment; the next model request contained the exact returned
URI as `video_url`, and NInfer recorded `media 1`.

| Selected source frames | Returned native URI query | Visual answer | Wire/session check |
| --- | --- | --- | --- |
| 2–3 | `start_frame=2&end_frame=3` | blue | passed |
| 0–1 | `end_frame=1` | red | passed |

The neutral-name fixture `/videos/mystery.mp4` contains red frames 0–1 and blue
frames 2–3. The second run reused the first run's source/index (`index_builds=1`,
`index_reuses=1` observed before its reader). The real-model HTTP acceptance also
passed again, including authentication, native inference and tools while suspended.
Reports for both color runs and HTTP acceptance are saved under `.cache/video-mcp`.
All v1 completion conditions now pass. Test servers were stopped after verification.
