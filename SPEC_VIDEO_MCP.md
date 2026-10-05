# NInfer Video MCP 実装仕様

## 1. 目的

`ninfer-all` の既存 HTTP サーバーに MCP (Model Context Protocol) endpoint を追加し、AI エージェントから NInfer の動画入力機能を直接利用できるようにする。

本仕様の対象は以下の 3 tool である。

- `get_video_metadata`
- `resolve_video_time`
- `inspect_video`

各 tool の公開 name / description / inputSchema / outputSchema の正本は、同梱の `mcp/tool-schema.json` とする。

3 tools の `path` は NInfer から見える local file path のみを受け付ける。URI は入力として受け付けず、query によるオプション指定も行わない。`inspect_video` の詳細な絶対パス要件は §9.2 に従う。生成した `ninfer-video://` URI は tool result と後続モデル入力に使用する。

本ファイルは、`tool-schema.json` を実際に `ninfer-all` に実装するための内部処理、共有キャッシュ、エラー処理、MCP transport、`ninfer-video://` との連携方法を定義する。

この 2 ファイルだけを読めば、実装担当者が追加確認なしで実装に着手できる状態を目標とする。

---

## 2. 全体構成

既存 OpenAI-compatible API が以下で公開されているとする。

```text
http://127.0.0.1:8080/v1/...
```

MCP は同じ HTTP server / port 上の以下に公開する。

```text
http://127.0.0.1:8080/mcp
```

つまり `/v1` と `/mcp` は同一プロセス・同一 port で共存する。

```text
ninfer-all
├─ /v1/...     OpenAI-compatible API
└─ /mcp        MCP Streamable HTTP
```

MCP transport は **Streamable HTTP** を使用する。

`/mcp` は MCP の initialize / tools/list / tools/call 等を処理する。MCP クライアントが必要とする `POST /mcp` を実装し、使用する MCP library / protocol version が GET stream / DELETE session を要求する場合は同一 endpoint 上でそれらも実装する。

初期版は localhost 利用を主用途とし、独自認証機構の追加は必須としない。既存 server の bind/address/security policy に従う。

---

## 3. 最重要設計: VideoSourceService を共有する

### 3.1 背景

以下の呼び出しは連続して行われる可能性が高い。

```text
get_video_metadata(path)
resolve_video_time(path, 10.0)
resolve_video_time(path, 20.0)
inspect_video(path, start_frame=..., end_frame=...)
```

また `inspect_video` が生成する `ninfer-video://` は、その後 NInfer 本体の動画 decoder に入力される。

各 tool と `ninfer-video://` 本体が個別に ffmpeg / container probe / timestamp scan を実行すると無駄が大きい。

したがって、**MCP 専用 decoder を別に作ってはならない。**

既存 `ninfer-video://` 実装と MCP tools の両方が使用する共通層を作る。

仮称:

```cpp
VideoSourceService
```

概念構造:

```text
MCP get_video_metadata ─┐
MCP resolve_video_time ─┼──> VideoSourceService ──> existing FFmpeg/libav path
MCP inspect_video ──────┤          │
                        │          ├─ metadata cache
ninfer-video:// decoder ┘          ├─ frame timestamp/index cache
                                   ├─ seek/keyframe information
                                   └─ optional decoder-session cache
```

### 3.2 共通キャッシュの必須対象

最低限、以下を共有する。

1. **Probe / metadata cache**
   - video stream index
   - width / height
   - duration
   - nominal / average FPS
   - time base
   - audio stream presence/count
   - container-reported frame count
   - interlace / field-order information
   - CFR/VFR 判定に使った情報

2. **Frame timestamp index**
   - 0-based source frame number
   - 実際の presentation timestamp
   - 必要なら keyframe / seek anchor 情報

3. **ファイル identity**
   - canonical path
   - size
   - modification time
   - OS/file-system から安価に得られる stable identity があればそれも使用可

### 3.3 任意の共有対象

以下は有効なら共有してよいが、v1 必須ではない。

- open済み `AVFormatContext`
- decoder context
- demux seek state
- 小規模な decoded-frame cache

ただし decoder context を長時間保持するために file handle / RAM が増える場合は、無理に常駐させない。

**重要なのは metadata と frame index の共有であり、decoder instance の共有そのものは必須ではない。**

### 3.4 キャッシュ key と invalidation

cache key は少なくとも canonicalized local path を含む。

同じ path のファイルが置換された場合に古い情報を使わないよう、cache entry には file identity を保存する。

最低限:

```text
canonical path
file size
mtime
```

のいずれかが変化したら entry を invalid とし、再解析する。

`ninfer-video://` に crop / scale / skip_frame 等が付いていても、**source-level metadata/frame index の cache key は元動画ファイル単位**とする。

crop / scale / sampling は source index とは別レイヤーである。

### 3.5 キャッシュ lifetime

process-wide cache とする。

初期版は単純な LRU でよい。

推奨:

```text
max cached video sources: 8～32
```

または timestamp index の総 bytes に上限を設ける。

frame timestamp は `int64` の source time-base tick または microseconds/nanoseconds の整数で保持することを推奨する。`double seconds` のみを正本にしない。

例として数十万 frame の timestamp index は数 MB 程度なので、通常は十分小さい。

---

## 4. Source frame の定義

全 tool および `ninfer-video://` の frame number は、以下で統一する。

```text
0-based decoded video frame number in presentation order
```

つまり最初に表示される decoded video frame が frame 0。

packet number、decode order、DTS order ではない。

B-frame を含む動画でも **presentation order** を基準とする。

frame timestamp は decoder が得る presentation timestamp を使用する。FFmpeg/libav を使う場合は、適切な PTS / best-effort presentation timestamp を stream time base で解釈する。

公開する秒数の原点は **source frame 0 の presentation timestamp** とする。

```text
timestamp_seconds = (frame_pts - first_frame_pts) * stream_time_base
```

`resolve_video_time.time_seconds`、返却する各 timestamp、およびモデルへ渡す動画 timestamp はこの定義に統一する。元の整数 PTS は index/seek 用に保持する。内部 PTS が 100 秒から始まる動画では、`time_seconds=10` は内部時刻 110 秒に対応する。

途中の frame range を選択しても原点は元動画の source frame 0 のままとし、選択範囲の先頭へリセットしない。duration は時間の長さであり、生の終端 PTS をそのまま duration として返さない。

この定義を `get_video_metadata`、`resolve_video_time`、`inspect_video`、既存 `ninfer-video://` decoder の全てで共通化する。

---

## 5. CFR / VFR の扱い

### 5.1 CFR

動画が確実に CFR で、固定 FPS が既知の場合は、時刻から frame number を計算できる。

概念:

```text
frame ~= time_seconds * fps
```

計算に使う時刻は §4 の source frame 0 を原点とする秒数。開始 PTS の offset を除去し、MCP と既存 decoder の公開 timestamp を同じ定義に揃える。

AI agent には Skill 側で、**confirmed CFR なら `resolve_video_time` を呼ばず FPS から直接 frame を計算してよい**ことを伝える。

### 5.2 VFR

VFR では `time * average_fps` による frame number 推定は禁止する。

`resolve_video_time` は実際の frame presentation timestamp から解決する。

### 5.3 CFR/VFR 判定

container の nominal/average rate の差だけで断定しない。

既存 video decoder で信頼できる判定手段があれば利用する。

確実に判定できない場合は:

```json
"variable_frame_rate": null
```

とする。

`null` の場合、正確な時刻→frame変換が必要なら `resolve_video_time` を使う。

---

## 6. Frame timestamp index

### 6.1 目的

VFR で以下を高速化する。

```text
resolve_video_time(path, 10)
resolve_video_time(path, 20)
inspect_video(start_frame=A, end_frame=B)
```

### 6.2 index 内容

最低限:

```cpp
struct VideoFrameIndexEntry {
    int64_t pts;       // source stream time base
};
```

array index 自体が 0-based source frame number なので、frame number を各 entry に重複保存する必要はない。

必要なら追加:

```text
keyframe flag
seek anchor / packet position
```

### 6.3 lazy build

全動画について起動時に index を作らない。

必要になったときだけ構築する。

推奨ルール:

- `get_video_metadata`
  - 安価な probe だけで回答できる範囲は index を作らない
  - 既に index があるなら exact frame count / VFR判定等に再利用する

- `resolve_video_time`
  - VFR または exact mapping が必要なら index を構築 / 補完する
  - 同一 source の次回呼び出しでは再利用する

- `ninfer-video://` / `inspect_video`
  - frame range seek に index が有用なら既存 index を再利用する
  - decode 中に新しい正確な frame timestamp が得られたら index cache を補完してよい

### 6.4 一度の解析を再利用する

最重要ケース:

```text
get_video_metadata(foo.mp4)
resolve_video_time(foo.mp4, 10)
resolve_video_time(foo.mp4, 20)
inspect_video(foo.mp4, start_frame=300, end_frame=600)
```

期待動作:

1. metadata で probe cache 作成
2. 10秒 resolve 時に必要なら frame index 作成
3. 20秒 resolve は同じ index を binary search
4. inspect_video / ninfer-video decoder は同じ metadata/indexを使用して seek

**2回目の resolve で動画全体を再scanしてはならない。**

---

## 7. Tool: get_video_metadata

公開 schema は `tool-schema.json` に従う。

### 7.1 入力

```json
{
  "path": "/path/to/video.mp4"
}
```

NInfer から見える local file path のみを受け取る。`ninfer-video://` 等の URI は受け付けない。

返す metadata は **元 source video** の情報とする。

### 7.2 処理

1. path normalize / canonicalize
2. file existence / readability check
3. `VideoSourceService` から source entry 取得
4. cache hit なら既存 probe を使用
5. cache miss なら既存 `ninfer-video` と同じ FFmpeg/libav path で probe
6. metadata を構造化して返す

### 7.3 width / height

source video stream の decoded dimensions。

rotation metadata を既存 `ninfer-video` decoder が表示方向に適用する場合、ここで返す width/height の定義もその実装と統一すること。

どちらの定義にするかをコード上で明示し、tool と decoder で食い違わせない。

推奨は **実際に model へ渡す前の source display dimensions**。

### 7.4 duration_seconds

信頼できる stream/container duration を使用する。

より正確な duration が frame index から得られる場合、cache 内では更新してよい。

### 7.5 frame_count

exact に分かる場合のみ値を返す。

```json
"frame_count": 12345,
"frame_count_exact": true
```

container metadata が不正確 / 不明で、index scan をまだ実施していない場合:

```json
"frame_count": null,
"frame_count_exact": false
```

**metadata tool のためだけに常に動画全体を decode して exact frame count を求めることは v1 では必須ではない。**

すでに exact frame index cache があれば、その length を返す。

### 7.6 FPS

- `average_fps`: descriptive average
- `nominal_fps`: stream/container reported rate
- `variable_frame_rate`: true / false / null

VFR時の time→frame 解決に `average_fps` を使用してはならない。

### 7.7 audio

最低限:

```text
has_audio
audio_stream_count
```

audio decode は不要。stream existence を見るだけでよい。

### 7.8 interlace

返却:

```text
progressive
interlaced
mixed
unknown
```

および `field_order`。

container/codec metadata だけで確定できない場合、既存 decoder が実フレームを観測する仕組みを持っていれば共有する。

重い全編解析を metadata tool のたびに実行しない。

判定不能なら `unknown`。

---

## 8. Tool: resolve_video_time

公開 schema は `tool-schema.json` に従う。

### 8.1 用途

1つの時刻を実 source frame 番号へ解決する。

典型例として、AI agent が 10～20 秒を詳しく見たい場合:

```text
resolve_video_time(path, 10.0)
resolve_video_time(path, 20.0)
```

を別々に呼び、得られた境界 frame を使って:

```text
inspect_video(path, start_frame=..., end_frame=...)
```

とする。

### 8.2 入力

```json
{
  "path": "/path/to/video.mp4",
  "time_seconds": 10.0,
  "radius_frames": 2
}
```

### 8.3 実装

CFR/VFRにかかわらず、tool が呼ばれた場合は **実 timestamp に基づく結果を返すことを正確性の基準**とする。

ただし confirmed CFR で exact mapping が trivial な場合は高速計算してよい。

VFRの場合:

1. `VideoSourceService` の frame index を取得
2. index 未構築なら構築または必要範囲まで確実に構築
3. target time を stream time base に対応づけ、source frame 0 の PTS を加えて index 上の時刻と比較する（最近傍比較前の整数丸めで精度を失わないこと）
4. sorted PTS index を binary search
5. 最も近い presentation frame を選択
6. 前後 `radius_frames` を返す

### 8.4 tie-break

指定時刻から同距離の frame が前後にある場合は、**前側の frame を nearest とする**。

このルールを固定し、テストする。

### 8.5 video 範囲外

`time_seconds > duration` の場合は tool error とする。

duration が完全に信頼できない場合でも、index構築後に最終 frame timestamp を超えることが判明したら error。

負値は schema validation で拒否。

### 8.6 返却例

```json
{
  "requested_time_seconds": 10.0,
  "nearest_frame_number": 299,
  "nearest_timestamp_seconds": 9.987,
  "frames": [
    {
      "frame_number": 297,
      "timestamp_seconds": 9.920,
      "delta_seconds": -0.080,
      "is_nearest": false
    },
    {
      "frame_number": 298,
      "timestamp_seconds": 9.953,
      "delta_seconds": -0.047,
      "is_nearest": false
    },
    {
      "frame_number": 299,
      "timestamp_seconds": 9.987,
      "delta_seconds": -0.013,
      "is_nearest": true
    },
    {
      "frame_number": 300,
      "timestamp_seconds": 10.021,
      "delta_seconds": 0.021,
      "is_nearest": false
    },
    {
      "frame_number": 301,
      "timestamp_seconds": 10.055,
      "delta_seconds": 0.055,
      "is_nearest": false
    }
  ]
}
```

---

## 9. Tool: inspect_video

公開 schema は `tool-schema.json` に従う。

### 9.1 重要

`inspect_video` 自身が Qwen を別途呼び出して動画を解析してはならない。

この tool の役割は:

1. 引数 validation
2. `ninfer-video://` URI の安全な組み立て
3. MCP tool result として、その URI を **resource link** で OpenCode/agent に返す

ことである。

その後、OpenCode 側の patch により resource link が同じ agent の次の model call の `video_url` content に変換される。

したがって、「同じ Qwen/NInfer agent 自身が動画を見る」という構造を維持する。

### 9.2 URI 生成

`path` 入力は **NInfer から見える動画ファイルの絶対パス** のみとする。Linux/WSLC では例として `/videos/demo.mp4` を指定する。`ninfer-video://`、`file://` 等の URI および相対パスは受け付けない。

`path` の `?` 以降に query オプションを付ける方式はサポートしない。frame range、sampling、crop、scale、deinterlace は各 tool 引数で指定する。入力 path を URI として解析したり、query を取り出したりしない。ファイル名そのものに含まれる予約文字は出力 URI の生成時に escape する。

省略時は先頭から末尾まで、`skip_frame=0`、`scale=1`、`deinterlace=auto`、bbox は全体とする。`ninfer-video://` URI はこの tool の **出力** として生成する。

入力例:

```json
{
  "path": "/work/video.mp4",
  "instruction": "Read the small text in the upper-right corner.",
  "start_frame": 100,
  "end_frame": 200,
  "skip_frame": 2,
  "scale": 2.0,
  "bbox": {
    "x": 1200,
    "y": 0,
    "width": 720,
    "height": 400
  },
  "deinterlace": "auto"
}
```

内部で既存 `ninfer-video://` parser が受け付ける canonical URI を生成する。

URI syntax の正本は **既存 `ninfer-video://` implementation** とし、MCP 側で別 parser / 別 semantics を作らない。

MCP 側は既存 URI builder/helper を呼ぶこと。

例（実際の parser syntax に合わせる）:

```text
ninfer-video:///work/video.mp4?start_frame=100&end_frame=200&skip_frame=2&scale=2.0&bbox=1200,0,720,400&deinterlace=auto
```

### 9.3 validation

最低限:

- path が絶対パスであること（URI/相対パスを拒否）、path exists / readable、および既存 local-media-root 認可
- `start_frame <= end_frame`
- start/end が exact frame count 既知なら範囲内
- bbox が source dimensions 内
- scale > 0
- skip_frame >= 0
- deinterlace enum

bbox が一部 source 範囲外の場合は silent clamp せず error とする。

### 9.4 resource_link result

MCP result は少なくとも:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Inspect the selected video frames visually."
    },
    {
      "type": "resource_link",
      "name": "selected video",
      "uri": "ninfer-video:///work/video.mp4?...",
      "mimeType": "video/mp4"
    }
  ]
}
```

とする。

`instruction` は tool call の意図として text result に含めてもよい。

推奨:

```text
Inspect the selected video frames visually and answer this instruction: <instruction>
```

### 9.5 inspect_video と cache

`inspect_video` は URI を返すだけでも、validation のため `VideoSourceService` の metadata cache を利用する。

さらに後段で NInfer がその `ninfer-video://` URI を decode するときも **同じ process-wide `VideoSourceService` entry を使う**。

したがって:

```text
get_video_metadata
  -> probe cache
resolve_video_time
  -> timestamp index cache
inspect_video
  -> URI validation uses same cache
next model request / ninfer-video decoder
  -> same cache/index reused
```

となること。

---

## 10. 既存 ninfer-video decoder との統合

### 10.1 重複実装禁止

以下を MCP 専用に再実装しない。

- path parsing
- video stream selection
- FFmpeg/libav initialization
- frame decoding
- timestamp extraction
- crop
- scale
- skip_frame sampling
- deinterlace
- VFR timestamp generation

既存 `ninfer-video://` 処理を共通 library/service に寄せ、MCP tools からも呼べるようにする。

### 10.2 推奨リファクタ

概念:

```cpp
class VideoSourceService {
public:
    Result<VideoSourceHandle> open(const VideoSourceId& source);
    Result<VideoMetadata> metadata(const VideoSourceHandle& source);
    Result<FrameResolution> resolve_time(
        const VideoSourceHandle& source,
        double seconds,
        int radius_frames);
    Result<VideoFrameReader> create_reader(
        const VideoSourceHandle& source,
        const VideoSelection& selection);
};
```

既存 URI decoder は:

```text
parse ninfer-video URI
        ↓
VideoSelection
        ↓
VideoSourceService::create_reader()
```

MCP tools は:

```text
get_video_metadata -> metadata()
resolve_video_time -> resolve_time()
inspect_video      -> metadata()/validation + URI builder
```

### 10.3 URI parser / builder

parser と builder は同じ module に置く。

```cpp
parse_ninfer_video_uri(...)
build_ninfer_video_uri(...)
```

MCP server が query string を独自連結しないこと。

escaping / Windows path / WSL path / special characters を1箇所で処理する。

---

## 11. MCP protocol 実装

### 11.1 endpoint

標準 endpoint:

```text
http://127.0.0.1:8080/mcp
```

port は既存 ninfer-all HTTP server の port 設定に従う。

例:

```text
--port 8080
```

なら:

```text
OpenAI API: http://127.0.0.1:8080/v1
MCP:        http://127.0.0.1:8080/mcp
```

### 11.2 transport

MCP Streamable HTTP を使用する。

MCP implementation/library が protocol negotiation、session ID、SSE response 等を提供する場合はそれを使い、独自 JSON-RPC transport を一から実装しない。

### 11.3 tools/list

`tool-schema.json` の3 toolを公開する。

```text
inspect_video
get_video_metadata
resolve_video_time
```

schema は可能な限り `tool-schema.json` と同一に保つ。

C++ struct / schema code generation を使う場合でも、公開 schema に drift が生じないテストを追加する。

### 11.4 tools/call

各 call は同期的に完了して結果を返す。

長時間の background job API は v1 では不要。

`resolve_video_time` 初回だけ timestamp index scan に時間がかかる可能性は許容する。その結果は cache し、次回以降を高速化する。

---

## 12. Error handling

MCP protocol-level invalid request と tool execution error を区別する。

推奨 tool error codes / messages:

```text
video_not_found
video_not_readable
video_open_failed
video_stream_not_found
invalid_frame_range
frame_out_of_range
invalid_bbox
invalid_video_time
video_time_out_of_range
video_index_failed
video_decode_failed
unsupported_video_format
```

MCP library が structured tool error をサポートする場合は structured data を返す。

最低限、人間と agent が原因を特定できる message を返す。

内部 FFmpeg error code だけをそのまま返さない。

---

## 13. Thread safety / concurrency

OpenAI API と MCP が同じ process で並行呼び出しされる可能性がある。

`VideoSourceService` cache は thread-safe にする。

同一動画の frame index が同時に2回構築されないようにする。

推奨:

```text
per-source entry lock / once-state
```

異なる動画は可能なら並列解析可能にする。

既存 `ninfer-video` decoder が FFmpeg context を request-local に持つ場合、その構造を無理に共有しない。

metadata/index cache は immutable snapshot として共有すると実装が単純になる。

---

## 14. Memory / resource policy

v1 では decoded video frames 全体を cache しない。

保持対象は主に:

```text
metadata
frame timestamp index
small seek/index metadata
```

のみ。

これにより長時間動画でも RAM 使用量を抑える。

必要なら小さな decoded-frame LRU を将来追加してよいが、v1 非必須。

cache entry eviction 時は open decoder/file handle 等を確実に解放する。

---

## 15. Logging / observability

debug 時に以下を確認できるようにする。

```text
video source canonical path
metadata cache hit/miss
frame index cache hit/miss
frame index build duration
indexed frame count
resolve target time
resolved nearest frame/timestamp
inspect_video generated ninfer-video URI
cache eviction
```

通常 log level では URI に機密パスが含まれる可能性があるため、full path logging は既存 ninfer-all policy に従う。

---

## 16. テスト

### 16.1 MCP transport

- `/mcp` に接続できる
- initialize / discovery が成功する
- tools/list に3 toolsが出る
- input schema が期待どおり

### 16.2 get_video_metadata

少なくとも:

- CFR MP4
- VFR MP4/MKV
- audioあり
- audioなし
- progressive
- interlaced source

を用意する。

### 16.3 resolve_video_time

- CFR
- VFR
- fractional seconds
- time=0
- 最終 frame 近辺
- range外
- radius=0
- radius=2
- tie時は前frame

を検証する。

VFRテストでは `time * average_fps` では得られない frame が正しく返るケースを必ず含める。

### 16.4 cache reuse

同一動画で:

```text
get_video_metadata
resolve_video_time(10)
resolve_video_time(20)
inspect_video(...)
```

を連続実行し、以下を assertion / instrumentation で確認する。

- probe は1回
- frame timestamp index 全体 scan は最大1回
- 2回目 resolve は index reuse
- `ninfer-video://` decode が同じ source metadata/index entry を参照

### 16.5 inspect_video

MCP result の `resource_link.uri` が正しい `ninfer-video://` URI であること。

入力は絶対パスのみ受理し、URI/相対パスを拒否すること。query オプションを解釈しないこと、選択条件は個別引数とその既定値だけで決まることも検証する。

OpenCode patch と組み合わせた E2E で最終的に NInfer OpenAI request に:

```json
{
  "type": "video_url",
  "video_url": {
    "url": "ninfer-video://..."
  }
}
```

が入り、同じ agent が動画内容を回答できること。

### 16.6 URI edge cases

- spaces
- non-ASCII path
- Windows drive path
- WSL/Linux path
- query escaping

をテストする。

---

## 17. 実装順序

推奨順序:

### Phase 1: 共通 VideoSourceService

1. 既存 `ninfer-video://` 実装から probe / decoder helper を抽出
2. metadata cache を追加
3. frame timestamp index を追加
4. existing decoder を共通 service 利用へ切替

### Phase 2: MCP endpoint

1. `/mcp` Streamable HTTP endpoint を既存 HTTP server に追加
2. tools/list
3. schema registration

### Phase 3: metadata / resolve

1. `get_video_metadata`
2. `resolve_video_time`
3. cache reuse test

### Phase 4: inspect_video

1. shared URI builder
2. validation
3. MCP resource_link return
4. OpenCode patch との E2E test

---

## 18. v1で実装しないもの

以下は不要。

- MCP server を別 process として起動
- MCP専用 ffmpeg subprocess
- toolごとの個別 video cache
- 全 decoded frame のRAM cache
- audio内容解析
- speech-to-text
- 動画の要約を tool 内部の別LLMで実施
- VFR timestamp を average FPS で近似
- background indexing daemon

必要になった場合のみ将来追加する。

---

## 19. 完了条件

以下をすべて満たしたら v1 完了。

- [x] `http://127.0.0.1:<port>/mcp` で MCP Streamable HTTP が利用可能
- [x] `tool-schema.json` の3 toolsが公開される
- [x] `get_video_metadata` が実動画 metadata を返す
- [x] CFR/VFR/audio/interlace 情報を schemaどおり返す
- [x] `resolve_video_time` が実 presentation timestamp から0-based source frameを返す
- [x] VFRで `average_fps` を frame mapping に使用しない
- [x] `inspect_video` が canonical `ninfer-video://` resource_link を返す
- [x] `inspect_video` 自身は別LLM inferenceを実行しない
- [x] MCP tools と `ninfer-video://` decoder が同じ `VideoSourceService` を使用する
- [x] metadata cache が共有される
- [x] frame timestamp index が共有される
- [x] `resolve_video_time(path, 10)` の後の `resolve_video_time(path, 20)` で再全scanしない
- [x] 後続 `ninfer-video://` decode でも既存 metadata/index が再利用される
- [x] source file変更時に cache が invalidationされる
- [x] OpenCode + MCP + NInfer のE2Eで同じQwen agentが動画を視覚入力として受け取れる

---

## 20. 実装上の最終原則

この機能は「MCP server」「metadata probe」「timestamp resolver」「ninfer-video decoder」を別々の動画実装として作るものではない。

**動画 source の理解と index は1つ、入口だけが複数**という設計にする。

```text
                 ┌─ get_video_metadata
                 ├─ resolve_video_time
VideoSourceService
                 ├─ inspect_video validation/URI build
                 └─ ninfer-video:// actual decode
```

これにより、agent が同一動画を段階的に調査するときの repeated probe / repeated timestamp scan / redundant file open を最小化する。

### 確定した実装契約（2026-10-06）

- 指定 OpenCode の native-video lowering に合わせ、`inspect_video` の custom resource link は、元 container に関係なく互換用 MIME `video/mp4` を付ける。ユーザー承認済み。URI は native video input の識別子で、MP4 bytes を配信する HTTP resource ではない。実ファイル形式は既存 FFmpeg pipeline が検出する。
- Linux の `/mcp` は既存 listener に登録する。fastmcpp pinned transport の MCP 2025-11-25 / 2025-06-18 profile を使用し、POST は JSON、GET は 405、DELETE は session 終了。initialize と initialized 通知を完了してから tools を呼ぶ。sampling 等の server-initiated request は提供しない。
- 3 tools の path は絶対 local path に統一する。query の分解は行わず、`?` 等はファイル名の一部として扱う。root 認可と symlink 解決は全 call で実施する。
- tool call は最大4並行、scan 上限2,000,000 frames、geometry 上限64 Mi pixels、120秒 deadline。MCP cancelled 通知と shutdown は probe/scan/wait の checkpoint へ伝播する。OS filesystem 操作自体は同期的で、割り込み不能な filesystem syscall の即時停止までは保証しない。
- 未設定 local-media-root でも discovery は可能。実 tool は local_media_disabled の tool error を返す。既存 API key を全 MCP request に適用し、OPTIONS は既存 CORS policy に従う。Host は loopback または明示 bind host、Origin は同一 HTTP origin を検証する。
