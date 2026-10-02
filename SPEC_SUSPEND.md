# NInfer GPU Suspend / Resume 仕様

## 1. 目的

NInfer 上で動作する大規模モデルを、**プロセス・CUDA context・CUDA GraphExec を維持したまま、一時的に GPU VRAM から退避し、後で高速に復帰**できるようにする。

主用途は、RTX 3090 など VRAM 24 GiB 級の単一 GPU 上で、以下のように複数モデルを交互に利用する構成である。

- テキスト / マルチモーダル生成: Qwen3.8-27B
- 画像生成 / 編集: Qwen-Image-2.1
- 上位ハーネス: OpenCode 等

想定フロー:

```text
Qwen3.8-27B
  ↓ tool call
Qwen-Image-2.1
  ↓ generated image
Qwen3.8-27B が画像を評価
  ↓ 必要なら再度 tool call
Qwen-Image-2.1
  ...
```

Qwen3.8-27B は通常時に RTX 3090 の VRAM をほぼ使い切るため、画像生成モデル利用中は Qwen3.8-27B の主要 GPU resident memory を解放する必要がある。

本機能では、単なる model unload / reload ではなく、**CUDA Graph の再 capture / instantiate を避ける**ことを重要要件とする。

---

## 2. 非目的

初期実装では以下を対象外とする。

- 実行中 generation の途中 suspend / preemption
- 複数 active request を強制停止して suspend
- GPU 全体を一括保存する VRAM snapshot（`Program::persistent` の whole-arena snapshot のみ行う）
- CUDA context の破棄 / 再生成
- CUDA GraphExec の破棄 / 再 capture
- model process の停止 / 再起動
- `auto_resume:false`での停止中にinferenceをqueueして待機する機能
- arbitrary な memory class を API caller が細かく指定する機能

初期版は **request 間の安全な境界でのみ suspend** する。

---

## 3. 基本設計

### 3.1 v1 の最小構成

20 GiB 程度の VRAM 解放が目的であり、v1 では細粒度な residency 管理を行わない。

扱う GPU memory は以下の3領域だけとする。

```text
Model / Program
├─ WeightArena          fixed-VA VMM / SSDから復帰
├─ Program::persistent fixed-VA VMM / RAM raw snapshotから復帰
└─ workspace_storage    fixed-VA VMM / 内容保存せず fresh backing
```

加えて、以下は suspend 中も保持する。

```text
CUDA context
CUDA modules / kernel handles
CUDA streams / events
cudaGraph_t / cudaGraphExec_t
各 arena の VA reservation
pinned host ingress / egress
CUDA driver / graph 内部 allocation
CPU-side Program / allocator metadata
```

### 3.2 WeightArena

model weights は約17 GiB規模で RAM 常駐コピーを持たない。

- suspend: physical VRAM backing を release
- resume: same VA に backing を再 map
- `.ninfer` artifact から SSD -> pinned staging -> VRAM で再 upload

### 3.3 Program::persistent

現行 `Program::persistent` は KV、StateImage、block table、RoundState、sampling/config、replay buffer 等を含む単一 `DeviceArena` である。

v1 では **この arena を分割しない**。

suspend 時に arena 全体を byte-for-byte で Host RAM に raw snapshot し、その後 physical VRAM backing を release する。

resume 時は same VA に backing を再 map し、raw snapshot を arena 全体へそのまま H2D restore する。

これにより以下の実装は不要になる。

- `persistent_control` / `persistent_suspendable` 分割
- KV page payload と execution table の backing 分離
- suspend 専用の KV/State demote transaction
- Device KV page lease / State slot lease の解放・再構築
- resume 時の Host continuation materialization

CPU-side allocator / sequence metadata はプロセス内にそのまま残るため、persistent arena を同一 VA・同一 bytes で復元すれば suspend 前の device state と整合する。

### 3.4 persistent raw snapshot の Host memory

snapshot は model weights のような17 GiB級ではなく、Program persistent arena の容量分だけ必要である。

48 GiB RAM 環境を前提に、まず実機で `persistent.capacity()` を計測し、RAM予算内であることを確認する。

v1 は性能より単純性を優先し、snapshot buffer は通常の Host RAM でよい。必要なら後で chunked pinned staging を追加する。

推奨実装:

```text
D2H:
  cudaStreamSynchronize(all relevant streams)
  cudaMemcpy(host_persistent_snapshot,
             persistent.data(),
             persistent.capacity(),
             cudaMemcpyDeviceToHost)

H2D:
  cudaMemcpy(persistent.data(),
             host_persistent_snapshot,
             persistent.capacity(),
             cudaMemcpyHostToDevice)
  cudaStreamSynchronize(nullptr) // pageable H2Dのdefault-stream DMA完了・エラー確認
```

snapshot buffer は suspend/resume 機能有効時に lazy allocate してよい。snapshot は `SUSPENDED` の間だけ保持し、resume 成功後は解放してよい。

### 3.5 Workspace

`workspace_storage` は request 間で内容を保持する必要がない scratch として扱う。

- suspend: physical backing を release
- resume: same VA に fresh backing を map
- zero-init は既存コードが必要とする範囲のみ行う

実装時に、quiescent な request 境界を越えて workspace 内容へ correctness 上依存するコードがないことを確認する。依存が見つかった場合、その領域だけを `Program::persistent` 側へ移すか snapshot 対象にする。workspace 全体の snapshot 化は v1 の既定方針にはしない。

### 3.6 20 GiB 解放目標

v1 の VRAM 回収対象は以下だけでよい。

```text
WeightArena
+ Program::persistent
+ workspace_storage
```

この3領域の合計解放量が20 GiB以上なら、他の小領域を追加で suspendable にしてはならない。

不足する場合のみ、実測に基づいて追加領域の分離を検討する。

## 4. CUDA VMM による fixed virtual address

### 4.1 必須要件

CUDA Graph は capture / instantiate 時の device pointer を保持するため、resume 後も GPU pointer 値を変えてはならない。

そのため、suspend 対象の大規模 GPU allocation は `cudaMalloc()` ではなく CUDA Driver API の Virtual Memory Management を用いる。

概念:

```text
初期化
  cuMemAddressReserve()  -> VA を確保
  cuMemCreate()          -> physical allocation
  cuMemMap()             -> VA に map
  cuMemSetAccess()

suspend
  cuMemUnmap()
  cuMemRelease()
  ※ cuMemAddressFree() は呼ばない

resume
  cuMemCreate()
  cuMemMap()             -> 同一 VA に再 map
  cuMemSetAccess()
```

`data()` / device pointer は Engine lifetime 中不変でなければならない。GraphExec が直接参照する pointer だけでなく、`Program::persistent` 内に device pointer 値が保存される場合、その参照先も resume 後に同一 VA でなければならない。v1 では suspend/release 対象を WeightArena、`Program::persistent`、`workspace_storage` に限定し、これらはすべて fixed-VA とする。

### 4.2 VMM capability check

起動時に VMM 対応を確認する。

```cpp
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
```

`--enable-model-suspend` 指定時に未対応 GPU / driver だった場合は起動時エラーとする。未指定時は従来 path で起動する。

### 4.3 GraphExec 保持

以下を守ること。

- graph replay 中に unmap しない
- capture 中に map/unmap しない
- suspend 前に全 relevant CUDA work を drain する
- resume 完了後にのみ graph replay を許可する

同一 VA に backing を remap した後、既存 `cudaGraphExec_t` を再利用する。

---

## 5. NInfer 内部メモリの扱い

### 5.1 v1 では persistent を分割しない

現行 `Program::persistent` をそのまま1個の fixed-VA VMM arena に変更する。

概念的な ownership:

```cpp
struct ProgramImplCore {
    DeviceArena persistent;        // fixed-VA VMM, raw snapshot対象
    DeviceArena workspace_storage; // fixed-VA VMM, 内容保存不要
    // graph family / pinned host buffers / CPU metadata は従来どおり
};
```

`persistent` 内部の layout、offset、constructor contract は変更しない。

したがって以下は v1 では変更不要である。

- `PagedKVCache` の constructor / backing 構造
- KV pages と execution tables の layout planner
- `StateImageDevicePool` の layout
- `RoundState` / sampling / token count 配置
- GDN replay / DFlash persistent layout の arena 分割

### 5.2 Persistent arena を raw snapshot できる理由

suspend は request 間の quiescent point でのみ実行する。

その時点で GPU work を完全に drain し、`Program::persistent` の内容を丸ごと Host RAM にコピーする。

以下は process 内に保持される。

- `SequenceState` 等の CPU-side metadata
- page allocator の CPU-side bookkeeping
- slot / lease / generation metadata
- graph objects
- fixed VA reservation

resume では persistent arena の VA を変えず、全 bytes を元通りに戻すため、GPU-side block table、KV payload、StateImage、RoundState、replay data 等を個別に再構築する必要はない。

### 5.3 VMM backing

`persistent` と `workspace_storage` は、`--enable-model-suspend` 有効時だけ remappable VMM backing を使用してよい。

`data()` / base pointer は Engine lifetime 中不変とする。

### 5.4 Weight arena

weight arena も同じく fixed-VA VMM 化するが、内容の復帰元は RAM snapshot ではなく `.ninfer` artifact とする。

### 5.5 Arena 外 allocation の確認

v1 実装前に、CUDA Graph または `Program::persistent` から参照され、かつ suspend 時に内容・address の維持が必要な NInfer-owned device allocation が上記3 arena 外にないことを確認する。小さな arena 外 allocation が存在する場合は VRAM に残してよい。release する場合のみ fixed-VA 化が必要である。

### 5.6 suspend 完了時の invariant

v1 の必須 invariant は以下だけとする。

```text
- active request == 0
- pending decode / prefill == 0
- pending context/materialization transaction == 0
- relevant CUDA streams are fully drained
- Program::persistent raw snapshot is complete in Host RAM
- WeightArena physical backing is detached
- Program::persistent physical backing is detached
- workspace_storage physical backing is detached
- all three VA reservations remain allocated
- cudaGraphExec_t / CUDA context remain alive
```

KV page lease、State slot lease、execution table row lease は **解放しなくてよい**。persistent arena を完全復元するため、suspend 前の状態をそのまま維持する。

## 6. Continuation の扱い

v1 では suspend 専用の continuation 処理を実装しない。

`Program::persistent` 全体を raw snapshot / restore するため、KV、StateImage、block table、RoundState、replay buffer 等は suspend 前の device state のまま復元される。既存 Host KV / Host State / continuation cache は通常用途のままとし、本機能から `DemoteToHost` / rematerialize を呼ばない。

Host RAM に追加で必要なのは主に以下である。

```text
Program::persistent raw snapshot = persistent.capacity()
weight loader staging           = 現行 materializer 相当（数百 MiB 規模）
```

weight full mirror は保持しない。実装前に max context 160K の実設定で `persistent.capacity()` をログ出力し、48 GiB RAM 環境で余裕があることを確認する。RAM 予算を満たせない場合のみ、v1 完了後の最適化として arena 分割または既存 continuation cache への demote を検討する。

---

## 7. Weight restore

### 7.1 RAM mirror は作らない

48 GiB RAM 環境を想定し、17 GiB 級 weight の Host mirror は保持しない。

### 7.2 `.ninfer` artifact を restore source にする

復帰元のentryと使用したmultipart continuationはEngineの存続中に変更・置換しない。
Readerとdevice placementをModel側で保持し、resume前にファイルサイズと更新時刻を確認する。
既存のframing・object geometry・range検証を再利用し、source変更、missing file、read errorを
復帰失敗として扱う。任意のpayload破損を検出するための全量hashや追加の全量readはv1では行わない。

`.ninfer` は実行用 layout に量子化 / packing 済み tensor payload を保持している。

現行 materializer は以下を既に実装している。

- SSD / file read
- source-range coalescing
- pinned staging buffers
- 最大2本の並列direct read（LinuxはO_DIRECT、Windowsは非buffered positional read）
- async H2D
- deterministic DeviceArena placement

この loader を resume 用にも再利用する。

read-aheadは既存の最大64 MiB×4 staging slot内に限定する。CUDA uploadとevent待ちは
呼出し側のdevice/rankで管理し、GPUが読むslotを上書きしない。読込結果は元のsource順で
uploadし、読込/転送失敗時は未着手readを取消し、実行中readをjoinしてからGPU workをdrainし、
stagingを解放する。Readerの続巻とdirect handleの初回openも並列読込に対応する。

### 7.3 Materializer のリファクタ

現行の

```text
Reader + MaterializationPlan
  -> DeviceArena 新規作成
  -> upload
```

を以下の2段階に分離する。

```text
1. arena allocation / ownership
2. upload into existing arena
```

想定 API:

```cpp
Status upload_materialization(
    const ArtifactReader& reader,
    const MaterializationPlan& plan,
    DeviceSpan destination,
    DeviceContext& device);
```

resume 時:

```cpp
weight_arena.resume_backing();
upload_materialization(
    artifact_reader,
    retained_plan,
    weight_arena.span(),
    device);
```

### 7.4 Separate SSD snapshot file

初期版では作らない。

まず `.ninfer` からの direct restore 性能を測る。

将来、artifact parsing / scattered reads が bottleneck になる場合のみ、GPU arena layout と同じ compact residency cache file を追加してよい。

例:

```text
qwen3.8.residency-cache
```

これは最適化であり v1 の必須要件ではない。

---

## 8. Suspend / Resume state machine

状態:

```text
READY
  │ suspend
  ▼
SUSPENDING
  │
  ▼
SUSPENDED
  │ resume
  ▼
RESUMING
  │
  ▼
READY
```

異常時:

```text
ERROR
```

### 8.1 READY

通常 inference を受け付ける。
ゼロ出力のCPU-only submissionもEngineCoreのqueue lockの下でREADY確認と受付を一体化し、residency遷移と同じadmission境界を使用する。

### 8.2 SUSPENDING

新規 inference admission を停止する。

### 8.3 SUSPENDED

通常 inference request は拒否する。

CUDA context / GraphExec / VA reservation 等は生存している。

### 8.4 RESUMING

weight / device backing を復帰中。

通常 inference request は拒否する。

### 8.5 ERROR

suspend / resume 中の fatal failure。

自動 recovery は初期版では必須としない。

---

## 9. Suspend の前提条件

suspend は request 間でのみ許可する。

最低限、以下を全て満たす必要がある。

```text
- active request == 0
- pending decode == 0
- prefill in progress == 0
- pending context transaction == 0
- materialization transaction == 0
- relevant CUDA streams can be fully drained
```

busy の場合、suspend は待機せず 409 を返す。

初期版では generation を interrupt しない。

## 10. Suspend 手順

推奨順序:

```text
1. state READY -> SUSPENDING
2. 新規 inference admission を閉じる
3. busy condition を検査
4. 全 relevant CUDA work を drain
5. Program::persistent 全体を Host RAM へ raw D2H snapshot
6. WeightArena の physical backing を unmap / release
7. Program::persistent の physical backing を unmap / release
8. workspace_storage の physical backing を unmap / release
9. CUDA GraphExec / CUDA context / 各 VA reservation は保持
10. memory accounting を更新
11. state -> SUSPENDED
```

途中失敗時は、可能なら backing を維持したまま READY に戻す。

persistent snapshot 完了後に一部 backing を release してから失敗した場合は、安全に再 map / restore できるなら rollback し、できなければ ERROR とする。

## 11. Resume 手順

推奨順序:

```text
1. state SUSPENDED -> RESUMING
2. WeightArena に physical backing を同一 VA で再 map
3. `.ninfer` から retained MaterializationPlan に従い weight を再 upload
4. Program::persistent に physical backing を同一 VA で再 map
5. Host raw snapshot を Program::persistent 全体へ H2D restore
6. workspace_storage に physical backing を同一 VA で再 map
7. 必要な workspace 初期化のみ実施
8. CUDA work 完了を synchronize
9. GraphExec が有効なままであることを確認
10. state -> READY
```

KV / State / block table / RoundState 等を個別に再構築してはならない。raw snapshot から persistent arena を丸ごと復元する。

## 12. OpenAI互換 HTTP server への独自 API

標準 OpenAI API には model suspend / resume の標準仕様はないため、NInfer 独自 extension とする。

### 12.1 Suspend

```http
POST /v1/models/{model}/suspend
Content-Type: application/json
```

v1 request body:

```json
{}
```

初期版では caller に memory class を選ばせない。

suspend bodyは`{}`または`{"auto_resume":true|false}`のみ許可する。省略時はtrue。
resume bodyは`{}`のみ。空body、array、null、未知field、boolean以外のauto_resumeは400。
管理routeは既存API-key認証と公開model alias（`--model-id`を含む）を使用し、未知modelは404。
公開aliasはslashや末尾`/residency`を含められる。`GET /v1/models/{alias}`の完全一致は
常に既存model detailを返し、状態照会はそのURLへさらに`/residency`を追加する。
この区別はsuspend有効・無効の双方で維持する。
flag未指定のsuspend/resumeは400 `model_suspend_disabled`、residency照会はenabled=falseを返す。

成功:

```json
{
  "object": "model.residency",
  "model": "qwen3.8-27b",
  "state": "suspended",
  "auto_resume": true,
  "released_device_bytes": 21438267392,
  "retained_device_bytes": 183500800,
  "persistent_snapshot_bytes": 3824132096
}
```

### 12.2 Resume

```http
POST /v1/models/{model}/resume
Content-Type: application/json
```

body:

```json
{}
```

成功:

```json
{
  "object": "model.residency",
  "model": "qwen3.8-27b",
  "state": "ready",
  "restored_device_bytes": 21438267392,
  "weight_restore_source": "artifact",
  "restore_seconds": 3.18
}
```

### 12.3 Residency status

```http
GET /v1/models/{model}/residency
```

例:

```json
{
  "object": "model.residency",
  "model": "qwen3.8-27b",
  "state": "ready",
  "device_bytes": 21983756288,
  "persistent_snapshot_bytes": 0
}
```

### 12.4 READY 以外での inference

HTTPの生成API `/v1/responses`、`/v1/chat/completions`、`/v1/messages` は、
SUSPENDEDかつauto_resume=trueなら認証・model ID・基本的な入力検証後にresumeを1回実行し、
完了後に元の要求を実行する。auto_resumeは今回の停止のHTTP受付方針であり、Engine自体は明示管理のまま。
同時要求は同じresume完了を待ち、待機中も既存のservice受付上限とpending deadlineに含める。
待機数超過は429、待機中のtimeoutは503 request_queue_timeout、切断は499。
resumeを開始した要求の切断・timeoutでも共有restoreは完了させ、他要求を巻き込まない。
復帰失敗は待機要求に500 model_residency_errorを返し、ERRORとsnapshotを保持する。
後続要求は503 model_errorで拒否し、自動再試行しない。SSEは復帰完了後に開始する。

auto_resume=falseのSUSPENDED、SUSPENDING、ERRORは生成を503で拒否する。
RESUMINGはauto_resume=trueのとき既存のservice resume完了を待ち、falseでは503を返す。
モデル一覧・residency・health・診断・token counting・Responses compactでは自動復帰しない。
状態JSONにauto_resumeを公開する。次のsuspendで省略すればtrueへ戻り、resumeはどちらの方針でも使用できる。

```text
503 Service Unavailable
```

例 (`SUSPENDED`):

```json
{
  "error": {
    "message": "Model is suspended",
    "type": "model_unavailable",
    "code": "model_suspended",
    "param": null
  }
}
```

自動復帰対象外の`SUSPENDING` / `RESUMING`では現在状態に対応するmessage/codeを返す。

### 12.5 suspend busy

active request / transfer / transaction が存在する場合:

```text
409 Conflict
```

例:

```json
{
  "error": {
    "message": "Model cannot be suspended while requests or context transfers are active",
    "type": "invalid_state",
    "code": "model_busy",
    "param": null
  }
}
```

---

## 13. API idempotency

推奨仕様:

- `suspend` on `SUSPENDED`: GPU操作なしでauto_resumeを更新し、200で現在状態を返す
- `resume` on `READY`: 200 を返して現在状態を返す
- `suspend` on `SUSPENDING` / `RESUMING`: 409
- `resume` on `SUSPENDING` / `RESUMING`: 409
- `ERROR`: 500 を返し、error detail を返す

OpenCode / orchestration 側で retry しやすくするため、同一安定状態への API は idempotent とする。

---

## 14. Concurrency policy

初期版は model 単位の exclusive residency transition lock を持つ。

```text
residency_mutex / state_machine lock
```

以下を同時実行してはならない。

- suspend と resume
- resume と inference admission
- backing map/unmap と graph replay

HTTP request thread 上で内部 state transition を開始する場合でも、Engine 側で明示的に serialization すること。

---

## 15. Memory accounting / observability

VMM map時間はweight/persistent/workspaceの新規backing create/map/accessのみを合算する。
persistent H2D時間は全capacityの同期raw copyのみ。weight restore時間はmapを除く既存の
artifact read・transcode・staging・H2D pipeline全体で、読込と転送bytesをそれぞれ報告する。
readとH2Dはoverlapするため、weight restoreを単純なread時間とcopy時間の和として扱わない。
transcode-only経路も計測対象とし、transcodeの読込・変換・H2D完了を計測に含める。

以下を計測する。

### 15.1 Suspend metrics

- `suspend_total_seconds`
- `persistent_snapshot_d2h_seconds`
- `persistent_snapshot_bytes`
- `released_device_bytes`
- `retained_device_bytes`
- `vmm_unmap_release_seconds`

### 15.2 Resume metrics

- `resume_total_seconds`
- `weight_artifact_read_bytes`
- `weight_h2d_bytes`
- `weight_restore_seconds`
- `persistent_snapshot_h2d_seconds`
- `persistent_snapshot_bytes`
- `vmm_map_seconds`
- `mapped_device_bytes`

### 15.3 Residency status

少なくとも以下を API / logs から確認可能にする。`persistent_snapshot_bytes` は snapshot buffer を保持している間のみ非 0 とし、resume 成功後に buffer を解放した場合は 0 とする。

```text
state
weight_device_bytes
persistent_device_bytes
workspace_device_bytes
retained_device_bytes
persistent_snapshot_bytes
last_suspend_seconds
last_resume_seconds
last_error
```

## 16. Failure handling

### 16.1 Persistent snapshot failure

D2H snapshot に失敗した場合は backing を解放せず suspend を中断し、可能なら READY に戻す。

### 16.2 Weight restore failure

SSD read error / corrupt artifact / H2D error の場合:

- state = ERROR
- inference は受け付けない
- GraphExec / VA reservation を破棄せず diagnostics を残す

### 16.3 Persistent restore failure

same-VA remap または H2D restore に失敗した場合は state = ERROR とする。
pageable H2Dは呼出しのreturnだけではDMA完了を保証しないため、転送streamの完了同期と
エラー確認を必須とする。同期成功後にのみsnapshotを解放しREADYを公開する。

### 16.4 VMM remap failure

partial map 状態を安全に cleanup し、state = ERROR とする。poolの複数pieceの途中でcreate/map/accessに失敗した場合、成功済みのmapping/handleを即時rollbackする。cleanup自体に失敗した資源のみ所有権情報を保持し、destructorで再試行する。

### 16.5 Graph replay validation failure

PoC / debug build では resume 後に小さな validation inference / graph replay test を行えるようにする。production では optional とする。

### 16.6 Suspended状態での正常終了

正常にSUSPENDEDへ遷移したEngineの終了でも、設定済みHybrid prefix cacheのHost tierを
既存の永続化ファイルへ保存する。suspend時点でHost書込み完了とtransfer idleを確認済みのため、
保存はHost slabsとCPU indexのみを読み、GPU backingの復元・GPU reset/spill・暗黙resumeを行わない。
detachまたはresumeに失敗してERRORになった場合は、この正常終了保存を行わない。

## 17. 起動オプション案

v1 で必須なのは以下だけでよい。

```text
--enable-model-suspend
```

必要なら debug / validation 用に `--suspend-require-vmm` を追加してよい。

既存 `--host-kv-mib` / `--host-state-slots` は suspend 機能専用には使用しない。

## 18. 互換性

### 18.1 suspend 無効時

現行 behavior を完全に維持する。

- existing `cudaMalloc` path を残してよい
- existing model loading behavior を変更しない

### 18.2 VMM allocator abstraction

推奨:

```cpp
enum class DeviceResidency {
    Permanent,
    Suspendable,
};
```

または backing abstraction:

```cpp
class RemappableDeviceAllocation {
 public:
    void* data() const;
    size_t size() const;

    Status attach_physical();
    Status detach_physical();

 private:
    CUdeviceptr va_;
    size_t bytes_;
    CUmemGenericAllocationHandle backing_;
};
```

`data()` は Engine lifetime 中変わらない。

---

## 19. 実装順序

### Phase 0: standalone PoC

NInfer 本体を変更する前に RTX 3090 上で以下を確認する。

1. VA reserve
2. physical A map
3. CUDA Graph capture / instantiate
4. graph replay 成功
5. stream synchronize
6. device data を Host snapshot
7. physical A unmap / release
8. physical B を same VA に map
9. Host snapshot を restore
10. **同一 GraphExec** で replay
11. 1000 回以上反復

### Phase 1: WeightArena の fixed-VA VMM 化

- weight arena を remappable にする
- `.ninfer` materializer を existing arena upload 対応にする
- GraphExec を保持したまま release / restore を検証

### Phase 2: Program::persistent の fixed-VA VMM + raw snapshot

- arena layout は変更しない
- `persistent.capacity()` 分の Host snapshot buffer を追加
- suspend: whole-arena D2H snapshot -> unmap/release
- resume: same-VA remap -> whole-arena H2D restore

### Phase 3: Workspace の VMM 化

- content preservation 不要
- same VA remap のみ

### Phase 4: HTTP API / state machine

- `/suspend`
- `/resume`
- `/residency`
- 503 / 409 handling

### Phase 5: 必要な場合だけ performance tuning

実測で切替が遅い場合のみ行う。

- persistent snapshot の pinned/chunked copy
- artifact read span optimization
- larger / tuned pinned staging
- overlapped SSD read + H2D
- optional compact residency-cache file

**v1 で persistent arena 分割は行わない。**

## 20. 検証項目

### 20.1 Correctness

- suspend 前後で同じ prompt continuation から正常生成できる
- persistent raw restore 後も suspend 前の continuation から正常に生成できる
- Vision request も resume 後に動作する
- MTP / speculative backend 利用時も正しく復帰する

### 20.2 Graph

- Graph capture count が resume 前後で増えない
- Graph instantiate count が増えない
- existing GraphExec handle が維持される

### 20.3 Memory

RTX 3090 / Qwen3.8-27B / max context 160K で:

```text
suspend 後に >= 20 GiB の VRAM を解放
```

を目標とする。

GraphExec / CUDA context / CUDA driver 内部 allocation 等による残留 VRAM は許容する。

### 20.4 RAM

48 GiB RAM 環境で安定動作すること。

- weight full mirror を作らない
- persistent raw snapshot は `persistent.capacity()` 分のみ
- loader staging: 数百 MiB 規模
- 160K設定で OS / OpenCode / snapshot を含め48 GiB RAM内に収まる

### 20.5 Latency

計測:

- suspend total
- Program::persistent raw D2H snapshot
- `.ninfer` read
- H2D upload
- resume total
- resume 後最初の request TTFT

目標は、追加した suspend/resume 機構の overhead を小さくし、model switching の主な律速を `.ninfer` からの weight reload にすることである。

---

## 21. 将来拡張

v1 完了後、実測で必要性が確認された場合にのみ以下を検討する。

- `.ninfer` からの復帰が遅い場合: compact GPU-layout residency cache file
- `Program::persistent` の RAM snapshot が大きすぎる場合: arena 分割または既存 continuation cache への demote
- SSD 読み込みが律速になる場合: read / H2D pipeline の調整

v1 では、複数モデル管理、自動切替、resource class 指定 API、L3 continuation cache 等の汎用機能は実装しない。

---

## 22. 重要な設計上の禁止事項

以下は禁止する。

1. suspend 時に `cudaDeviceReset()` する
2. CUDA context を破棄する
3. GraphExec を破棄して resume 時に再 capture する
4. suspendable allocation の VA reservation を解放する
5. active request / GPU work 実行中に backing を unmap する
6. auto_resume=falseの停止、ERROR、または状態照会/token countingで自動resumeする
7. 17 GiB 級 weight full mirror を RAM に常駐させる
8. v1 で `persistent` を control / KV / state / transient 等へ分割する
9. v1 で KV page payload と execution tables の backing を分離する
10. v1 で suspend 専用 continuation demote / rematerialize path を新設する

20 GiB 解放目標を満たす限り、追加の細粒度 residency 機能を実装しない。

## 23. 完了条件

画像MCPは別リポジトリ `qwen-image-runtime` の
[SPEC_QWEN_IMAGE_RUNTIME.md](../../qwen-image-runtime/qwen-image-runtime/SPEC_QWEN_IMAGE_RUNTIME.md) と
[PLAN_QWEN_IMAGE_RUNTIME.md](../../qwen-image-runtime/qwen-image-runtime/PLAN_QWEN_IMAGE_RUNTIME.md)
に従う別作業である。2026-10-02の最新ユーザー指示により、画像MCPとの連携検証はNInfer suspendの完了条件から除外する。

初期実装の Done 条件は以下。

- [x] RTX 3090 で VMM + existing GraphExec remap PoC が安定動作
- [x] Qwen3.8-27B weight arena を fixed-VA VMM 化
- [x] `.ninfer` から same VA への weight restore が可能
- [x] `Program::persistent` を分割せず fixed-VA VMM 化
- [x] `Program::persistent` whole-arena raw D2H/H2D snapshot/restore が可能
- [x] `workspace_storage` を fixed-VA VMM 化し、内容保存なしで復帰可能
- [x] suspend 後 20 GiB 以上 VRAM が利用可能
- [x] resume 後 CUDA Graph recapture / reinstantiate なし
- [x] 160K設定で persistent snapshot を含め48 GiB RAM内で安定動作
- [x] `/v1/models/{model}/suspend` 実装
- [x] `/v1/models/{model}/resume` 実装
- [x] `/v1/models/{model}/residency` 実装
- [x] auto_resume=falseのsuspended中inferenceが503
- [x] HTTP suspendのauto_resume省略時true、3生成APIの共有resume、受付上限/切断/timeout/ERROR処理
- [x] busy suspend が 409
- [x] suspend/resume の繰り返し試験で leak / corruption がない
- 対象外: 画像MCPとの連携検証（最新ユーザー指示）。

NInfer単体項目は2026-10-02にWSLC/RTX3090の指定160K/MTP3/Vision overlay構成で受入済み。容量・RAM・100cycle・raw token・Graph・overlay両貸出経路・故障結果と未検証組合せは[PLAN_SUSPEND.md](PLAN_SUSPEND.md)のP6単体受入結果を参照する。画像MCP実装は別作業であり、その連携検証は本機能の完了条件に含めない。OpenCode等ハーネスの実行はユーザー指示により後日扱う。
