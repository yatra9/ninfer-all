# GPU Suspend / Resume 実装計画

作成日: 2026-10-02

## 0. 現在地と再開地点

- 要件の正本は [SPEC_SUSPEND.md](SPEC_SUSPEND.md)。本書は実装順序、対象コード、検証、未確定事項を管理する。
- 実装開始（2026-10-02）。P0専用PoC `tests/test_vmm_suspend.cu` を追加し、WSLCの既存
  `ninfer-syntax:dev`、CUDA 13.1.115、RTX 3090でcompile/run成功。1000回のphysical backing
  解放・新規create・same-VA mapを、単一capture/instantiateのGraphExecで検証した。
  persistent capacity全bytesの復元一致、内部pointer、継続round counter、毎回異なる値で汚した
  fresh workspaceからの正しい出力も確認済み。
- P1 Core実装完了: `RemappableDeviceAllocation` と通常DeviceArenaのopt-in fixed-VA経路、
  weight/KV poolのidle detach/attach、logical capacityとphysical bytesの分離を実装。
  VA・offset・mirror・貸出metadataは維持し、再attachは新規handleを利用する。
  各pieceのhome/overlay mappingと有効handleを追跡し、constructor/partial attach失敗と
  detached状態のdestructorで二重releaseを避ける。別CUDA contextでの操作を拒否する。
- P1検証: WSLC、CUDA 13.1.115、GCC 13、RTX 3090で `test_suspend_backing`、既存arena、
  weight pool、KV poolをcompile/runして成功。新規backingの両pool20回反復、arenaのmove/assignment、
  detached破棄、create/map/access失敗注入、復帰後overlay貸出とweight mirror復元を確認した。
  Python入り既存 `ninfer-vision-audit` image（Python 3.12.3）で正式CTest configure成功。
  Pythonテストは実行していない。Windows nativeでのbuild/runは未検証。
- 合意済み: suspendはidle時のみ。既存のVision overlayも対応対象とし、対応を別機能として切り離さない。
- 合意済み（2026-10-02）: 実機検証はWSLC上のLinuxを優先する。P0 PoCと実モデル受入はこの環境から進める。
- 現在の作業: P4は02e2be4dにcommit済み。P5 HTTP・CLI flag・サービス受付gate・timing細分化を実装し、対象buildと4関連テストが成功。P5は8e3cdd7cにcommit済み。P6の指定Qwen3.8-27B/MTP3/Vision overlay単体受入は完了。P0/P3の残項目も下記証拠で完了した。画像MCP連携は最新ユーザー指示により対象外。監査指摘2件の修正・回帰検証も完了（末尾参照）。新開発imageは ninfer-suspend:dev（FROM ninfer-all:build）、コンテナ ninfer-suspend-dev。既存BuildKit cacheを /build へ取り込んだ差分buildを使用する。正確なbuild pathと再開手順は末尾の最新検証欄参照。
  P0の各arena容量・physical量とproduction workspace監査は完了（末尾のP6結果参照）。予備実測用の
  旧 `ninfer-suspend-budget` コンテナ（host port 18082）はユーザーのディスク整理で削除済み。
- 各実装段階の完了時に、本節へ変更内容、実施した検証、未検証事項、次の作業を記録する。
- 未チェック項目は省略・不要と判断した項目ではなく、完了に必須の残作業。P0のarena所有権・workspace依存監査と容量/RAM実測、P3のcontinuation/Vision overlay復帰はP6単体受入で完了した。P4/P5接続を先に進めたのは実Program受入の入口を作るためであり、P0/P3の出口条件を免除したものではない。これらの証拠が揃ったためNInfer単体の完了を確定し、画像MCP実装へ進める。画像MCP連携項目は最新ユーザー指示により完了条件から除外する。
- 2026-10-02に実装開始と作業単位ごとのcommitを承認済み。AGENTS.mdのsuspend checkpoint規約に従い、
  検証済み単位をcommitして問題がなければ次段階へ継続する。

ユーザーの最新進行指示（2026-10-02）: 自力で解決できるエラーは修正して止まらず進める。ユーザーに聞かなければ解決できない問題に限り停止して確認する。非同期Askは使わない。以前のbuild後停止指示より、この最新の継続指示を優先する。

## 1. 実装方針

対象は単一GPUのGeneration Engine。起動時に `--enable-model-suspend` を指定した場合のみ有効化する。
既存のconcurrency設定は維持し、すべての受付済み要求とGPU処理が完了した境界でのみsuspendする。

| 対象 | suspend | resume | 保持するもの |
|---|---|---|---|
| WeightArena | physical backingを解放 | 同一VAへ再確保しartifactからupload | binding、復元用plan、artifact reader |
| `Program::persistent` | capacity全量を通常RAMへコピー後にbackingを解放 | 同一VAへ再確保し全bytesを復元 | CPU metadata、KV/State lease、snapshot |
| `workspace_storage` | 内容を保存せずbackingを解放 | 同一VAへ再確保し必要箇所のみ初期化 | layoutとVA |

CUDA context、Graph/GraphExec、streams/events、VA reservation、pinned ingress/egressは維持する。
Model/Programの再生成、graph recapture、persistentの分割、continuation demote、weight全量のRAM mirrorは行わない。
arenaの論理capacityとVMM granularityへ切り上げた物理確保量を区別する。

### 1.1 Vision overlayとの統合

idle時にはVision windowが閉じ、weight/KVの貸出が返却済みであることを確認する。
Vision実行途中のsuspendや、貸出途中の状態保存は実装しない。

- 通常のowning `DeviceArena` はremappable backingを所有する。
- overlayのweightは `EvictableWeightPool` が物理メモリを所有し、arenaはborrowed view。
- `EvictableKVPool` を使う構成ではpersistentもborrowed view。poolがpersistent全体のbackingを管理する。
- detach/attachは実際の所有者で行う。borrowed arenaから二重解放しない。
- poolのhome VAとoverlay用VA、貸出metadata、既存の限定的なweight window mirrorは維持する。
  snapshot対象はpersistent全体であり、KVのlendable prefixだけに限定しない。
- resume後のoverlay貸出は新しいphysical handleを使う。古いhandleをmetadataに残さない。
- overlay用mirrorとpinned Vision weightsもRAM計測へ含める。suspendのためのweight全量mirrorは追加しない。

## 2. 現行コードで確認した接続点

| 領域 | ファイル | 計画に反映する事実 |
|---|---|---|
| arena | `src/core/arena.h`, `arena.cu` | owning/borrowedの両方が存在。通常allocationに加えWindowsのresidency経路もある |
| overlay所有者 | `src/core/evictable_weight_pool.*`, `evictable_kv_pool.*` | 既存VMMと貸出transactionを保持したまま、idle時の全backing detach/attachを追加する |
| artifact upload | `src/artifact/materializer.h`, `materializer.cpp` | allocation、view生成、Host/Pinned materialization、device uploadが一体化している |
| load | `src/models/qwen3_5/load.cpp`, `src/runtime/engine/model_instance.cpp` | Readerは現行ではload時のローカル変数。planの `source` はReaderへの生ポインタ |
| Program | `src/models/qwen3_5/program/program_impl.h`, `program_impl.cpp` | persistentはKV pool経由の場合がある。workspaceは独立。追加rank用arenaもある |
| scheduling | `src/runtime/engine/engine_core.h` | `queue_mutex_`、`execution_mutex_`、worker、pending queue、outstanding予約がある |
| public API | `include/ninfer/engine.h`, `include/ninfer/types.h`, `src/runtime/engine/engine.cpp` | Engineを管理操作の入口とし、HTTPだけの状態管理にしない |
| serving | `src/serve/http_server.*`, `generation_service.*`, `serve_options.*` | availabilityとprotocol errorへ接続。既存GET `/v1/models/(.+)`と新routeの競合を避ける |
| 既存PoC | `tests/test_vmm_graph_remap.cu` | 同じphysical allocationのoverlay/home間remapを検証。解放後の新規backingで1000回反復する要件は未検証 |

materializerにはdevice objectのtranscode経路もある。resumeで単純なfile bytesのコピーだけを実装せず、
起動時と同じplacement・transcode・scale等の解釈を再現する。Host/Pinned資源は再生成しない。

### 2.1 P0所有権監査の記録（2026-10-02、進行中）

| 領域 | 実所有者と復元時に保持する契約 |
|---|---|
| weight | `MaterializedArtifact::arenas_`、overlayでは `pool_` が所有。rank 0のlogical capacityは `MaterializationPlan::device_capacity(0)`。pool physical量はchunk切上げを別途集計する |
| persistent | `ProgramImpl::persistent`、overlayで条件を満たす場合は `kv_arena` が所有。`plan.persistent.bytes` 全体を保存。Decoder、StateImage、RoundState、replay、DFlashのviewは再bindしない |
| workspace | `ProgramImpl::workspace_storage` が所有し `work` はgeneral領域のborrowed view。Vision bridgeはその末尾の1列で、request内のbridge転送に用いる。requestをまたぐ保持対象にはしない |
| streams | DeviceContextのcompute/transfer/Visionに加え `HybridPrefixCache::restore_stream_` が独立して存在。context writeのevent、restore/landing batchもquiescence対象。通常のdevice.synchronizeだけでは不足 |
| arena外 | 単一GPUProgramの宣言上の追加バッファはround/score/logit/speculativeのPinnedHostBuffer、Host KV/State、Vision結果pool。stage linksと追加rank arenaはmulti-GPU専用でv1対象外。Op-owned補助allocationの調査は継続中 |

Graph preparationはpersistent上のKV/State/round controlを初期化し、general workspaceはcapture時に
`work.reset()`して再利用する。MTP/DFlashのrequest間stateはpersistentにbindされる。
workspaceの全read-before-write保証は各production実行経路と実モデルのfresh-backing比較で継続確認する。

## 3. Engineの状態と排他制御

### 3.1 管理API

public Engineに `suspend()`、`resume()`、`residency()` 相当の型付きAPIを追加する。
名称・戻り型は既存APIの流儀に合わせる。EngineCoreが状態の唯一の所有者になり、Modelはweight復元、
Programはpersistent snapshot/workspace復元と物理的quiescence確認を担当する。

状態は `READY → SUSPENDING → SUSPENDED → RESUMING → READY` と `ERROR`。
HTTP層には独立したstate machineを置かない。起動完了フラグとモデルのresidencyを混同しない。

### 3.2 idleの判定とadmission

以下をbusyに含め、待機・キャンセル・queue破棄をせず409を返す。

- active request、pending queue、submit途中のoutstanding予約。
- prefill/decode/capture/commit/terminal処理とcontext/materialization transaction。
- Vision window、KV loan、weight eviction transaction、未完了の関連GPU work。

実装は次の順序とする。

1. residency操作同士はtry-lock相当で競合を即時409にする。
2. admissionと同じ短いcritical sectionでstate確認、受付閉鎖、outstanding確認を行う。
   busyならREADYへ戻し、GPU処理の完了を待たない。
3. workerの実行権を確保し、Program側のquiescenceを再確認する。
   長時間実行中の `execution_mutex_` を単純に待ってからbusy判定する実装にはしない。
4. compute/transfer/Vision/load等、監査で列挙した全関連streamをdrainする。
5. snapshotとbacking操作を行う。I/O中はqueue/status用mutexを保持せず、状態照会を継続可能にする。

workerはREADY以外ではGPU処理を開始しない。submitの予約・queue投入の両段階と、
public APIの即時完了経路にもavailability検査を適用する。
HTTPではmedia acquisition前に早期拒否し、最終的なEngine submitでも再検査する。
CPUでprepare中にsuspendが成立した場合も、後続submitは503となりGPUへ到達しない。
prepareがGPUへ触れる経路が見つかれば同じ実行権の管理対象に含める。

### 3.3 失敗・終了時

- Host snapshot確保失敗等、device stateが健全でbacking未変更ならREADYへ戻す。
- CUDAエラーでは健全性を確認できない限りREADYへ戻さない。
- partial detach後は全領域の復元完了を証明できる場合だけrollbackし、それ以外はERRORとする。
  v1では複雑な自動recoveryは必須にしない。
- resumeのread/upload/remap/restore失敗はERROR。snapshot、VA、GraphExecとdiagnosticsは保持する。
- partial mapの後始末は各所有者が行い、map済み範囲と有効handleを記録して二重releaseを防ぐ。
- shutdown/destructionはresidency操作と直列化し、READY/SUSPENDED/ERRORのいずれからも安全に終了する。
  suspended状態で存在しないbackingを読むcleanupや、不要な暗黙resumeは行わない。

## 4. 実装段階

### P0: 所有権監査、メモリ予算、Graph PoC

- [x] weight/persistent/workspaceの所有者、容量、物理確保量、stream、arena外allocationを一覧化する。
- [x] workspace内のrequest境界を越える依存を監査する。依存があれば該当領域だけpersistentへ移す等で解消する。
  decode graph、MTP/DFlash、Vision bridge、context transfer scratchを対象とする。
- [x] 既存VMMテストを拡張または専用テストを追加し、capture/instantiate後にD2H、unmap、release、
  新規create、same-VA map、H2D、同一GraphExec replayを1000回以上反復する。
- [x] pointerを含むpersistent相当データとfresh workspaceを含め、値・アドレス・GraphExec同一性を検証する。
- [x] RTX 3090・Qwen3.8-27B・context/KV 163840を暫定160K設定として容量を計測する。
  実際のartifact、量子化、spec backend、Vision設定、concurrency、Host cache設定を明記する。

出口条件: 対象環境でPoC成功、3領域で20 GiB回収の見込みと48 GiB RAM予算を実測から判断できること。
不足時は原因を提示して仕様を再検討する。persistent分割や追加領域の退避へ勝手に広げない。

### P1: Coreのfixed-VA backingとpool統合

- [x] granularity、overflow、device/context、map/access/handleの寿命を管理するVMM backingを実装する。
- [x] VA予約とphysical backingの寿命を分離し、detach中もbase/capacity/offsetを保持する。
- [x] 通常arenaにopt-inで接続し、無効時の既存allocation経路を維持する。
- [x] weight/KV poolにidle時の全backing detach/attachを追加する。
  open transactionやpoisoned状態を拒否し、resume後の貸出も検証する。
- [x] allocate/map/accessの途中失敗、move/destructor、反復detach/attachのテストを追加する。

出口条件: 同一VAで新規backingへ復帰し、通常arenaと両poolで二重解放・leakがないこと。

### P2: artifactから既存weight arenaへの再upload

- [x] materializerを「初回storage/view構築」と「既存device destinationへのupload」に分離する。
  既存のread coalescing、pinned staging、async H2D、transcodeを共通経路として維持する。
- [x] suspend有効時のみ、復元に必要なReaderとdevice placement情報をModel側の所有物として保持する。
  Readerのアドレスを安定させ、dangling `plan.source` やHostPlacement payloadの重複保持を避ける。
- [x] multipart artifactも起動時と同じsourceから読み、destination容量・offset・alignmentを検査する。
- [x] artifactはEngine存続中に変更しない契約を明記する。復元元不整合、破損、read errorはERRORにする。
  既存artifact検証を再利用し、毎回17 GiBの余分なmirrorや全量二重readを追加しない。
- [x] weight parent/Parameters/既存GraphExecを作り直さず、起動時と同じdevice bytesを復元する。

出口条件: 通常・overlay双方のweight所有者で、同一VAへの復元と既存GraphExec replayが成功すること。

P2検証（2026-10-02）: WSLC CUDA 13.1 / GCC 13 / RTX 3090でartifact materialization targetを正式buildし、CTest成功。通常・overlayのmultipart fixtureで10回の同一VA復帰、全weight bytes、既存GraphExec replay、Host資源の保持、復元元変更の拒否を確認。Q8→Q4/Q6 transcodeの復帰とdestination容量拒否も成功。Model/load/ModelInstanceの構文検査成功。任意payload改変の全量hash検出は行わず、immutable source契約と既存framing検証、size/mtime検査を使用する。実モデルのweight復帰はP6で確認する。次はP3のwhole-persistent snapshot、fresh workspace、idle/stream監査とsuspended終了処理。

### P3: Programのpersistent snapshotとworkspace復帰

- [x] persistentのlayout/constructor contractを保ったまま、通常arenaまたはKV poolへP1を接続する。
- [x] `persistent.capacity()` 分の通常Host RAMをlazy allocateし、whole-arena raw D2H/H2Dを実装する。
  used bytes、live KV、lendable prefixだけのコピーに縮めない。
- [x] workspaceをfixed-VA化し、fresh backingで必要な初期化のみ実行する。
- [x] snapshotはSUSPENDED中と復旧判断に必要な失敗時に保持し、resume成功後に解放する。
- [x] CPU側sequence/cache/lease metadataを維持し、既存continuationからの生成とoverlay再実行を検証する。

出口条件: persistent全bytesの復元とfresh workspaceでの推論が成立し、graph再構築が不要なこと。

P3実装（2026-10-02）: opt-inのpersistent/workspaceをVMM化。通常RAMをlazy確保しcapacity全体をraw D2H/H2Dする。CPU metadataとview/graphを再構築しない。request/context transaction/seal/Vision/pool loanと全stream（hybrid独立restoreを含む）の非blocking idle検査を追加。partial detach以降のcleanupとdestructor disk spillを抑止。WSLC GCC 13でProgramImpl/residency/cleanup/ModelInstanceの構文検査成功。artifact回帰CTest成功。専用コンテナ ninfer-suspend-dev の /tmp/ninfer-suspend-build で製品runtime全buildを進行中。Programの実モデルsnapshot・continuation・fresh-workspace検証はEngine接続後に実施するためP3の出口条件はまだ未確認。次はP4のEngine state/admission/try-lock、typed public APIと状態遷移テスト。

### P4: Engine state machineとpublic契約

- [x] EngineOptionsからModel/Program構築へenable flagを伝搬し、VMM capabilityをallocation前に検査する。
- [x] §3の状態遷移、即時busy判定、worker排他、availability、終了処理を実装する。
- [x] single-GPU Generation以外の非対応組合せは起動時に明示拒否する。flagなしの既存機能は維持する。
- [x] queue投入との競合、同時suspend/resume、状態照会、失敗注入をモデルなしでも検証できる範囲でテストする。

出口条件: 非READY状態からGPU実行へ到達せず、busy操作がrequest完了待ちにならないこと。

P4作業中（2026-10-02）: EngineCoreへ唯一のresidency state、admission両段階の非READY拒否、管理操作try-lock、execution try-lock、worker再確認、shutdownとの直列化、typed public APIを実装。snapshot allocation以外の失敗はERRORとし、Program cleanupをGPU非参照へ切り替える。hybridの完了済みtransferは待たずpublicationし、独立streamと未完了transferを確認する。suspend有効時のEngine終了ではabortするCUDA_CHECK同期を使用しない。小型BF16 artifactの独立GPU testを追加中（busy active/pending、反復生成、並行管理、非READYのzero-output拒否、source変更ERROR、suspended/error終了）。初回runtime buildは ninfer-suspend-dev /tmp/ninfer-suspend-build で約473/639 targetまで進めたが、下記RAM圧力により中断した（exec session 90506は終了）。Engine/Programの構文検査は成功、hybrid/終了補正を含む再構文検査も成功。新behavior testはbuild完了後にtarget ninfer_model_suspend_testをbuildしCTestで実行する。P4はまだcommit前。timingのmap/D2H/H2D細分化とHTTPはP5。

停止状況（2026-10-02）: 後半のptxasが各約2 GiBへ増加し、WSLC MemAvailable約201 MiB / swap使用12784 MiBとなった。GPU実行の問題ではなく、初回CUDA buildの並列コンパイルによるRAM不足。ユーザーの「問題時は止まって確認」に従い、自分のコンテナ内のNinjaへSIGINTを送り中断した。既存build成果物とsource変更は保存済み、GPU behavior testはまだ未実行。次の具体案は cmake --build /tmp/ninfer-suspend-build -j4 --target ninfer_model_suspend_test による既存treeからの再開、その後 ctest --test-dir /tmp/ninfer-suspend-build -R ^ninfer_model_suspend_test$ --output-on-failure。ユーザーが並列数4での再開を承認したため、既存treeから ninfer_model_suspend_test のbuildを再開した（WSLC exec session 15614）。既存attention CUDAソースのcompile/ptxasが長時間かかっているが進行中。並列数4ではMemAvailable約14～20 GiB、swap約22 MiBで安定しており、RAM圧迫は再発していない。behavior testが通るまではcommitしない。

### P5: HTTP、起動option、計測

- [x] `--enable-model-suspend` とhelp/options testを追加する。
- [x] POST `/v1/models/{model}/suspend`、POST `/resume`、GET `/residency` を追加する。
  model alias、認証、body検証、例外変換は既存serverの規約へ合わせる。
- [x] SPEC §13のidempotencyを実装する。遷移中は409、ERRORの管理操作は500。
- [x] Chat Completions、Responses、Anthropic Messages等の推論入口を503へ統一する。
  SSE開始前に拒否し、background requestを含む受付済み要求の存在もidle判定へ接続する。
- [x] 管理APIと診断APIはSUSPENDED中も使用可能にし、health/readinessの意味を既存契約と整合させる。
- [x] SPEC §15のbytes・時間・last_errorを公開する。読込、H2D、snapshot、map/unmap、totalを区別する。
  statusは一貫したCPU snapshotから返し、unmapped device memoryを読み取らない。

出口条件: HTTP schema/状態遷移テストが通り、resumeを明示的に呼ぶまで推論を再開しないこと。

### P6: 実機受入とドキュメント

- [x] 下記検証表を実施し、利用した環境・設定・測定値・未実施項目を本書へ記録する。
- [x] `docs/serving.md`、`docs/cli.md`、`docs/maintainer/engine-architecture.md`、関連するmemory/overlay説明を更新する。
- 対象外: 画像MCPとの連携検証（2026-10-02の最新ユーザー指示）。
  ユーザー指示（2026-10-02）により、NInferの実装・単体受入後に別リポジトリ E:\koji\work\20260813\qwen-image-runtime\qwen-image-runtime のSPEC/PLANに従って画像MCPを実装し、連携受入を予定していたが、最新ユーザー指示によりNInfer完了条件から除外した。汎用model managerは実装しない。OpenCode等のハーネス経由の検証は後日別途判断する。

## 5. 検証と受入基準

| 観点 | 検証 | 合格条件 |
|---|---|---|
| VMM/Graph | P0 PoCを1000回以上反復 | 新規backingで同じGraphExecが正しい結果を返す |
| exact restore | 小規模fixtureでweight/persistentのbyte比較 | 全対象bytesが一致。base pointerも不変 |
| workspace | fresh backingをdebugで異なる値に汚して実行 | 保存内容への隠れた依存がない |
| continuation | suspendなしの対照と、同じprefix/continuationから生成 | deterministic設定でtoken一致。KV/State metadataの整合性維持 |
| graph再利用 | 同じ事前実行済みworkloadの前後でhandleとcountを比較 | capture/instantiate countが増えない。未実行shapeによる通常captureと区別 |
| Vision | resident/overlay/cpuのうち有効な既存構成で復帰後に画像処理 | 正常生成。overlayのKV loan/weight fallback双方が復帰後も動く |
| speculative | none/MTP、対応artifactのDFlash/DFlash2 | 復帰後の生成・commitが正常。未用意artifactは未検証と記録 |
| concurrency | 複数active、待機queue、submit予約中、context転送、同時管理操作 | busyは409、非READYの推論は503、deadlockなし |
| failure | Host確保、read、map/access、D2H/H2D、partial release | 不完全な状態をREADYとして公開しない。安全に終了できる |
| VRAM | 3領域の実physical bytesとdevice free量を前後計測 | 目標構成で20 GiB以上解放。別プロセスの利用可能量も確認 |
| RAM | processとsystemのpeak、Host cache、overlay mirror、snapshot、stagingを計測 | OS/OpenCodeを含む48 GiB環境で安定。weight全量mirrorなし |
| latency | suspend/resume内訳と復帰直後TTFT | 実測を記録。SPECの例示3.18秒をSLAとして扱わない |
| endurance | 実モデルでまず100回、異常時は原因に応じて追加 | memory増加傾向、corruption、GraphExec再生成なし |
| disabled | flagなしの既存load/推論/overlayと公開APIテスト | 既存動作を維持 |

Coreは `tests/cmake/CoreTests.cmake` の既存arena/VMM/pool test、artifactは
`tests/artifact/tests.cmake`、runtimeは `tests/models/qwen3_5/`、serveは既存HTTP/options testへ接続する。
変更対象のbuildとbehavior testを段階ごとに実施し、未検証の実機条件をCPU test成功で代替しない。

GPU実行前に利用状況を確認する。Windows nativeで検証する場合は既存 `build-ninja/` と
AGENTS.mdのVS 2022環境を使用し、再configureしない。優先環境のWSLCでは既存Dockerfileの製品経路を使用する。
WSLCでのPoC・受入成功をWindows nativeでの動作保証にはしない。

## 6. 計測値の定義と残る確認事項

### 6.1 計画上の既定案

- APIのdevice bytesはNInferが所有・把握する実physical backing量として定義し、VA予約量を含めない。
  CUDA context/driver/Graph内部の未知の確保量は別のdevice-wide観測値として報告し、正確な内訳を装わない。
  logical capacity、mapped bytes、今回release/restoreしたbytesを区別する。
- 管理APIは処理完了まで同期応答する。client切断後も開始済みtransitionを安全な終点まで進め、
  retry時のidempotencyとGET statusで結果を確認できるようにする。
- suspend無効時もstatusで無効であることを識別可能にし、管理操作は明示的な非対応エラーとする。
  最終的なHTTP code/bodyはP5着手時にSPECへ追記し、schema testで固定する。
- multi-GPUとCausalScoringはv1対象外とする案。enable flagとの組合せを起動時拒否する。
- latency tuning、pinned snapshot、compact cache fileは実測後の別判断とし、v1の必須作業へ入れない。

### 6.2 実機着手前に確認すること

P0予備実測（2026-10-02）: artifactは
`C:\AI\ninfer-rtx3090-windows-x64-0.11.0-rtx3090\models\huihui-Qwen3.8-27B-abliterated-NInfer-v3\Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer`。
混合Q4/Q5/Q6/Q8、context/KV 163840、rk8v4、FP16 GDN state、MTP3、draft LM head、
Vision overlay、vision-max-merged 16384、concurrency 1、Host KV 8 GiB、Host state 8 slots。
既存製品image `ninfer-all:ninfer-video` でweights 16.7 GiB、runtime 5.36 GiB、
Host state 598.6 MiB。warmup後のdevice-wide使用量22164 MiB、free 2163 MiB、
process VmHWM 12896580 KiB。Windows RAM総量49674076 KiB、free 13688672 KiB、
WSLC VMのMemAvailable 11009412 KiB。persistent snapshotはruntime全量より小さく、現時点の
RAM余裕内に収まる見込み。これは3領域の正確なphysical量やsuspend後の解放量の証明ではなく、
その計測と負荷時peak確認は引き続き必要。画像生成側の起動・GPU解放手順は未確認。

1. **受入用のartifactと実運用設定**: 明示path、量子化、160Kの正確なtoken数、MTP等、
   concurrency、Host cache量、および画像生成側の起動・解放手順。
   モデルの追加downloadやartifact再生成はこの計画では実施しない。
2. **20 GiBの判定**: SPEC §20は「20 GiB解放」、§23は「20 GiB利用可能」と表現が異なる。
   計測は両方を記録し、解放量20 GiB以上を基準案とする。最終受入では他プロセスの占有を分けて判断する。

上記の確認はP0の実機条件を確定するためのもの。overlay対応は合意済みで再確認しない。

### 並列数4での再開後の停止（2026-10-02）

ユーザー承認で ninfer_model_suspend_test のbuildを -j4 で再開し、既存attention CUDAソース8本を含む19/185まで完了。約50分はMemAvailable約14～20 GiB / swap約22 MiBで安定していたが、別コンテナ keen_ozark の起動後に共有メモリ使用量が13632 MiBへ増え、MemAvailableが316 MiBへ急減した。別コンテナとの因果関係は未確定だが、新たなRAM圧迫を確認したため、ユーザーの問題時停止指示に従い自分のコンテナ ninfer-suspend-dev のNinjaへSIGINTを送り中断した。別コンテナには操作していない。build成果物と変更は保持、P4 behavior testは未実行、P4は未commit。次は並列数2へ下げるか、別の負荷が終了してから並列数4で再開するかをユーザーと確認する。再開コマンドは cmake --build /tmp/ninfer-suspend-build -j2 --target ninfer_model_suspend_test（または承認済み並列数4）。完了後 ctest --test-dir /tmp/ninfer-suspend-build -R ^ninfer_model_suspend_test$ --output-on-failure。GPUテスト前には別負荷とGPU使用量を再確認する。

再開指示（2026-10-02）: ユーザーが別コンテナを停止し、通常並列での再開を指示した。WSLC一覧では自分のninfer-suspend-devのみ稼働、MemAvailable 23201 MiBを確認。既存treeで cmake --build /tmp/ninfer-suspend-build -j --target ninfer_model_suspend_test を再開した。実際のRAM圧迫を監視し、問題発生時は停止して確認する。

通常並列での再開結果（2026-10-02）: 別コンテナ停止済み / MemAvailable 23201 MiBから -j で再開したが、cicc/ptxas/cc1plusが36本起動し、約1分でMemAvailableが91 MiBへ減少した（swap22 MiB）。別負荷がなくても無制限並列はWSLC RAM約23.7 GiBに収まらないため、自分のNinjaをSIGINTで中断。P4テストは未実行、変更・完了済みbuild成果物は保持。次は通常並列を -j8 に制限するか、実績のある -j4 を使用するかをユーザーに確認する。

再開承認（2026-10-02）: ユーザーが -j8 を承認。既存 /tmp/ninfer-suspend-build の ninfer_model_suspend_test を並列数8で再開。RAM使用量を監視し、問題時は停止して確認する。

監視頻度の指示（2026-10-02）: ユーザー指示によりRAM/buildの確認は約5分間隔とする。-j8 buildは継続中（exec session 37914）。8本アセンブル時にMemAvailable約2983 MiBまで減ったがswap増加はなく、その後一部CUDAソースが完了してMemAvailable 7943 MiBへ回復。behavior testはまだ未実行。

区切りで停止の指示（2026-10-02）: ユーザーがbuild完了後、現在の作業単位が一区切りついたら一旦停止するよう指示。P4のbuild・関連behavior test・checkpoint commitまで進め、P5/画像MCPへは進まない。image整理について、現在のimage ninfer-vision-audit:latest (ed463cd4ed91) とコンテナ ninfer-suspend-dev を保持するよう回答。build成果物は同コンテナ内 /tmp/ninfer-suspend-build。その他imageは今回のsuspend検証には不要だが、ninfer-all:latest は停止済み northern_kunlun、ninfer-all:ninfer-video は停止済み ninfer-suspend-budget が参照しているため、それらコンテナの要否を確認して整理する。こちらではimage/containerの削除は実施していない。

### 現在の停止地点: ディスク整理のためbuild中断（2026-10-02）

ユーザーが明示的にbuild中断を指示したため、ninfer-suspend-dev 内のNinjaへSIGINTを送り停止した。exec session 37914はexit 1（interrupted by user）。ninja/cicc/ptxas/cc1plusが同コンテナに残っていないことを確認済み。-j8 再開分は10/166まで完了し、w3/w4のattention CUDA群とw5 h16/h24 cachedが完了済み。build成果物はコンテナの書き込み層 /tmp/ninfer-suspend-build に保持。ディスク整理時も ninfer-vision-audit:latest と既存コンテナ ninfer-suspend-dev は削除せず保持する（コンテナ停止・WSL shutdownは可能）。こちらではコンテナ停止、image削除、WSL shutdownは実施していない。

P4実装は未commit、GPU behavior test未実行。C++構文検証とdiffレビューの既存結果は上記P4欄。次回はユーザーの再開指示を待ち、コンテナが停止していれば wslc start ninfer-suspend-dev、その後 wslc exec ninfer-suspend-dev cmake --build /tmp/ninfer-suspend-build -j8 --target ninfer_model_suspend_test を実行する。RAM/build監視はユーザー指定の約5分間隔。build完了後にGPU利用状況を確認して ctest --test-dir /tmp/ninfer-suspend-build -R ^ninfer_model_suspend_test$ --output-on-failure を実行し、関連checkを通してP4 checkpoint commitを作成する。ユーザーの『区切りがついたら一旦停止』指示は継続して有効なので、このcommit後はP5/画像MCPへ進まず停止する。

### 新build imageからの再開準備と確認待ち（2026-10-02）

ユーザーが全WSLCコンテナを削除し、masterから tmp/dockerfile-ninfer-build で ninfer-all:build、製品Dockerfileで ninfer-all:latest を作成した。ninfer-suspendブランチがcleanであることとstash名 20261012_1110を確認し、同ブランチで stash@{0} をpopして全18ファイルを競合なく復元した（stashはdrop済み）。以前の /tmp/ninfer-suspend-build とコンテナ成果物はコンテナ削除により失われた。

ユーザーは新開発DockerfileのFROMを ninfer-all:build にし、masterの生成物を使う差分buildを指定。image d58c3ccbe086 を一時コンテナのlsで確認したところ /build は存在せず、/out/ninfer と /out/ninfer-serve のみ存在した。tmp/dockerfile-ninfer-build は /build を --mount=type=cache,id=ninfer-build で使用しているため、CMakeCache・object・static library はimageには含まれない。この前提の相違を報告し、問題時停止指示に従い確認待ち。次の具体案は FROM ninfer-all:build の開発Dockerfileで同じBuildKit cacheをmountし、その内容を /build へコピーしてimageの永続layerへ取り込むこと。baseline cacheが利用できることを確認してから既存source/build pathを維持して差分同期・reconfigure・-j8 targeted buildを行う。まだ新Dockerfile作成/build、P4 GPU behavior test、P4 commitは未実施。監視は約5分間隔、P4 checkpoint後は区切りで停止する指示を維持する。

### BuildKit cache取り込みによる開発環境再開（2026-10-02）

ユーザーがcacheを開発imageへ取り込む構成を承認。tools/suspend-dev/Dockerfile は FROM ninfer-all:build とし、id=ninfer-build のcache内configurationのSHA256で既存treeを選び、sourceとそのtreeのみを同じ /build パスへコピーする。Python 3.12.3を追加した開発image ninfer-suspend:dev (1a4eb2d469b5) のbuild成功。新コンテナ ninfer-suspend-dev (a6ce6aca37c0) を起動し、repoを /workspace:ro へmountした。prepare-build.sh は内容比較rsyncで未変更sourceのmtimeを保ち、既存tree /build/57382beee819500f31a1c6917b3f94a49d4c1f45c1915ebd70acfb6b0d97d4e3 でBUILD_TESTINGをONにする。初回の同期除外がnested models source/testも除外していた不備は、除外patternをルート限定に修正して解消。configureとshell構文検証成功、suspend test登録をdry-runで確認した。

差分targetは142件。Ninja explainにより共通 include/ninfer/types.h などの変更がCUDAを含むdependent targetの再compileを必要とすることを確認。全件の新規buildではないが、長いattention CUDAコンパイルは必要。build-test.sh で -j8 の ninfer_model_suspend_test と影響したloading-test objectをbuild中。ログ /tmp/ninfer-suspend-build.log、監視は約5分間隔。P4 GPU behavior testとcheckpoint commitはまだ未実施。次はbuild成功後にGPU利用確認とCTestを行い、P4をcommitして区切りで停止する。

### ユーザー再呼出し待ち: buildを継続して監視終了（2026-10-02）

ユーザーが『いったん止めて、半日後に呼び出す』運用を希望したため、agentの監視・作業を終了する。コンテナ内buildは継続する。自動再開・自動通知は設定していない。ユーザーの次の呼出しで再開する。

現在ブランチninfer-suspend、stash 20261012_1110はpop済み。P4実装と tools/suspend-dev/ 開発環境は未commit。開発コンテナ ninfer-suspend-dev、image ninfer-suspend:dev (FROM ninfer-all:build)。build directory /build/57382beee819500f31a1c6917b3f94a49d4c1f45c1915ebd70acfb6b0d97d4e3、buildログ /tmp/ninfer-suspend-build.log。build-test.sh が -j8 で ninfer_model_suspend_test と ninfer_qwen3_5_loading_test objectをbuildしている（exec session 62066）。sleep infinity のコンテナ自体はbuild終了後も稼働する。コンテナ停止・削除やWSL shutdownはbuildを中断するため、完了待ち中は稼働を維持する。

再開時はまず wslc exec ninfer-suspend-dev ps -C ninja -o pid,etimes,args とログ末尾を確認する。ninjaがまだ動いていれば既存buildへ重複起動しない。終了済みならログにFAILED/エラーがないか、対象 executableが存在するかを確認する。成功後はGPU使用状況を確認して ctest --test-dir /build/57382beee819500f31a1c6917b3f94a49d4c1f45c1915ebd70acfb6b0d97d4e3 -R ^ninfer_model_suspend_test$ --output-on-failure を実行する。必要な関連checks後、PLAN/SPEC/docsと全diffをレビューしてP4 checkpoint commitを作り、区切りで停止する。エラー・判断が必要な疑問は非同期Askではなく停止して確認する。

### P4 checkpoint検証完了とP5への再開地点（2026-10-02）

最新のユーザー指示は「自力で解決できる問題は修正して続行、ユーザー判断が必須の場合だけ停止」。以降のfixtureエラーは自律的に修正した。MRoPE、連続parentのprojection配置、推論用tokenizer設定、KV容量を対応契約へ揃えた。極小projection形状とattention-onlyのStateImage構成は既存実行経路に対応していないため、製品演算を拡張せず、supported dense geometryのattention/GDN各1層、hidden5120、FFN17408、vocab272、BF16/row-scaled FP8の生成artifactへ変更した。重みはゼロなので、生成一致は状態遷移のsmoke evidenceであり実モデル数値品質の根拠ではない。

開発コンテナninfer-suspend-dev、WSLC CUDA13.1/GCC13/RTX3090で build-test.sh の -j8 targeted buildが成功（ninfer_model_suspend_test executable、変更したninfer_qwen3_5_loading_test object）。ctest --test-dir /build/57382beee819500f31a1c6917b3f94a49d4c1f45c1915ebd70acfb6b0d97d4e3 -R '^ninfer_model_suspend_test$' --output-on-failure が1/1 PASS、14.48秒。disabled/非対応設定、active+queued100token要求を取消さない即時Busy、10回fresh-backing復帰後の3token一致、非READY zero-output拒否、2thread各20回管理競合、source変更のERRORと診断/snapshot保持、suspended/error destructorを確認した。関連diffをレビューしgit diff --check成功。helper shell構文検査と開発image build/configureも成功済み。

P4実装はtyped public APIとEngineCore state/admission排他として完了。P3の実continuation/Vision overlay再実行、P0 arena外allocation/workspace監査、実モデル20GiB/48GiB/100cycle受入は未完了でP6に残す。Windows nativeと全test suiteは未実行（今回の検証はWSLCの対象targetに限定）。timingのmap/read/H2D分離はP5。次はP4 checkpoint commit後、P5のCLI flag、HTTP management/status routes、GenerationService受付・media/zero-output gateとprotocol error、schema/HTTP testsを実装する。既存コンテナとbuild treeを保持し、prepare-build.sh完了後に必要targetを順次buildする。NInfer受入完了前に画像MCP実装へ進まない。

### GPU test: synthetic fixtureのMRoPE設定で停止（2026-10-02）

ユーザーの再開指示後、testのVMM能力照会をcuda.h、cuInit、cuDeviceGetAttributeへ修正した。prepare-build.shが完了してからbuild-test.shを実行し、ninfer_model_suspend_test executableと関連loading-test objectのbuildは成功。GPUは13 MiB使用・24314 MiB空き。記録済みbuild directoryで ctest -R '^ninfer_model_suspend_test$' --output-on-failure を実行したが、1.33秒で Qwen3.5 config: MRoPE sections exceed interleaved axis ranges により失敗。suspend/resume動作には未到達。head_dim=64、partial_rotary_factor=0.5なのでrotary pairは16個で、fixtureの {8,8,0} はconfig.cppのinterleaved軸配分に一致しない。各pairを3軸へ配分する {6,5,5} が有効な修正案。production設定検証の変更は不要。新たな問題があれば停止する指示に従い、fixture修正前に停止した。P4未commit、次はこのfixture修正の確認後に差分同期・短いtest build・CTestを行う。

### 再build: VMM能力照会APIのcompile errorで停止（2026-10-02）

ユーザー承認後、suspend test targetへninfer_coreを追加し、prepare-build.shの同期・configureが完了した。configure完了前にbuildを開始したため旧設定で一度失敗したが、完了後に再実行しCUDA include不足は解消した。production static libraryのlinkは成功。残るtest compileで cudaDevAttrVirtualMemoryManagementSupported が未定義となった。インストール済みCUDAヘッダーで確認したところ、VMM能力照会はDriver APIの CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED を使う。既存 tests/test_vmm_suspend.cu とCore poolも cuDeviceGetAttribute を使用している。次の修正案は test_model_suspend.cpp に cuda.h を明示includeし、cuInit(0) と cuDeviceGetAttribute によるVMM照会へ変更すること。問題時停止指示に従い、この修正前に確認待ち。GPU behavior testとP4 commitは未実施。build directoryと再開コマンドは上記と同じで、CUDA objectを再compileする必要はない。

### build終了確認: test target依存不足で停止（2026-10-02）

ユーザー再呼出しで確認したところNinjaは終了、ログ末尾は subcommand failed。失敗箇所は tests/models/qwen3_5/test_model_suspend.cpp のcompileのみで、runtime/engine/model_instance.h から core/device.h をincludeした際に cuda_runtime.h が見つからない。test targetの LIBRARIES は ninfer_engine ninfer::json のみで、内部Coreヘッダーを使う既存testが指定する ninfer_core が不足している。修正案は tests/models/qwen3_5/tests.cmake の ninfer_model_suspend_test の LIBRARIES へ ninfer_core を追加すること。CUDA/headerの問題はtest依存設定であり、production実装のcompile errorではない。GPUは13 MiB使用で空きあり。GPU behavior testはまだ未実行、P4は未commit。ユーザーの問題時停止指示に従い修正・再build前に確認待ち。既存の完了済みCUDA objectを保持して、承認後にprepare-build.shで差分同期・configure、build-test.shで残りのcompile/link、CTestを実行する。

### P5 checkpoint検証完了とP6への再開地点（2026-10-02）

CLI/serveの --enable-model-suspend をEngineへ伝搬し、認証付きPOST suspend/resumeとGET residencyを公開した。model aliasはslashを含む明示aliasにも対応し、既存model detail routeと競合しない。POSTは{}のみ、未知model404、disabled400、Busy409、ERROR管理500。GenerationServiceのcapacity reservationと管理入口を同じ短lockで調整し、media準備・token count・応答保持もBusy対象にした。I/O中にcapacity lockは保持しない。認証後のpre-route gateと最終admissionにより非READY推論は503となり、SSE/media取得/zero-outputへ迂回しない。背景実行は既存Responses契約で非対応のまま。診断/statusは利用でき、healthはavailabilityに従い503。EngineCoreが唯一のモデルstate所有者である点は維持した。

VMM map時間を3領域のcreate/map/accessから計測し、persistent H2Dとweight upload pipelineから分離した。weight read/H2D bytesは既存materializerのactual counts。読込・転送pipelineはoverlapするためweight_restore_secondsはmapを除くpipeline全体と定義しSPEC/docsへ記載した。docs/serving.md、docs/cli.md、engine architecture、tests READMEを更新した。

WSLC CUDA13.1/GCC13/RTX3090で -j8 targeted build成功: ninfer、ninfer-serve、ninfer_model_suspend_test、ninfer_model_residency_http_test、ninfer_cli_options_test、ninfer_serve_options_test。共通types.hの追加変更はなく、CUDA演算の長時間再compileは発生していない。CTest CLI options PASS 0.00秒、serve options PASS 0.97秒、Engine GPU PASS 14.83秒。最終HTTP GPU test PASS 2.78秒（認証401、slash alias、未知model404、body400、zero-output応答予約のBusy409、suspended管理/診断/health、5推論入口503、明示resume後の3token HTTP生成、idempotency、source変更ERROR/500/診断snapshot保持）。テストfixtureのmessage不足とlogger header不足は修正済み。全diffレビューとgit diff --check成功。全suite/Windows nativeは未実行。コンテナsourceに.gitを同期していないため製品build idはunknownという既存生成警告があるが、機能buildは成功している。

次はP5 commit後にP6実機受入。コンテナninfer-suspend-devにbuild成果物があり、build directoryは /build/57382beee819500f31a1c6917b3f94a49d4c1f45c1915ebd70acfb6b0d97d4e3、P5 buildログ /tmp/ninfer-suspend-p5-build.log。実モデルhost path C:\AI\ninfer-rtx3090-windows-x64-0.11.0-rtx3090\models\huihui-Qwen3.8-27B-abliterated-NInfer-v3\Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer を存在確認（20,437,521,664 bytes）。既存build containerはこのmodels bindを持たないため、成果物をE:のworkspaceへ取り出し、指定model directoryをread-only mountした別acceptance containerで実行する方針（artifactをC:へ複製しない）。GPU空きを確認して、context/KV163840、rk8v4、MTP3、draft head、FP16 GDN、Vision overlay、merged16384、concurrency1、HostKV8GiB/State8で20GiB解放・48GiB RAM・continuation/Graph/overlay/100cycleを検証する。P0 workspace/arena外allocation監査、P3 continuation/Vision再実行の出口条件と故障検証はまだ残る。NInfer単体受入後にのみqwen-image-runtime実装へ進む。

### P6: モデルのWSLCネイティブ配置と初回受入（2026-10-02）

ユーザー指示によりモデル配置をWindows bind mountからWSLCネイティブのnamed volume `ninfer-suspend-models`へ変更した。元の明示モデルdirectoryから全4ファイルをcp -aでコピーし、volume上で `sha256sum -c SHA256SUMS` がモデルとconversion JSONの両方でOK。元ファイルは保持。前節の「artifactをC:へ複製しない」はこの新しい指示により更新され、約20.4GBのコピーをWSLCが管理するLinux filesystemへ置いている。

受入containerは `ninfer-suspend-acceptance`、image `ninfer-suspend:dev`、host port18082。モデルは `-v ninfer-suspend-models:/models:ro`、artifact `/models/Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer`。結果・server binary・Graph interposerは `E:\koji\work\20260813\NInfer\tmp\suspend-acceptance:/acceptance`。設定はcontext/KV163840、rk8v4、MTP3、draft head、FP16 GDN、Vision overlay、merged16384、concurrency1、HostKV8GiB/State8を維持。開発container ninfer-suspend-devとbuild成果物は保持した。

追加した tools/suspend-dev/acceptance.py は直接HTTPでstored Responsesの継続生成、suspendedの推論503、実VRAM、snapshot/restore/statusを検証する。graph-trace.cpp は受入専用LD_PRELOADで実CUDA Graph capture/instantiate/launch/destroyを観測する。製品APIや実装へ診断機能は追加していない。WSLC Python3.12.3、CUDA13.1/GCC13/RTX3090でinterposerのcompileと実モデル1cycle実行が成功。継続出力とoutput token数が対照と一致し、capture18/instantiate10/destroy0を維持した既存GraphExecへのlaunchを確認した。これは文字列出力とtoken数の一致であり、raw token vectorの直接比較はまだ残る。

Windows bindでの初回trace計測はsuspend5.54秒、resume15.39秒、weight restore13.64秒、試験iteration21.62秒。native配置の初回はsuspend3.400秒、resume7.671秒、weight restore6.057秒、persistent D2H2.562秒/H2D0.511秒、VMM map0.866秒、試験iteration11.591秒。単発比較であり、suspend時間の差を保存場所の効果とは断定しない。モデルは依然artifactから復元し、全weight RAM mirrorは追加していない。

physical backingはweight17,918,066,688 bytes、persistent4,756,340,736 bytes、workspace218,103,808 bytes、合計22,892,511,232 bytes（21.3203GiB）。snapshot logical bytesは4,754,642,176。suspendedで3領域のretained bytesは0、GPU空き23993MiB、継続生成のcached tokens41を確認した。Windows側初回process VmHWMは17.16GiB、swap0（これはprocessの測定であり全host RAM予算の結論ではない）。

旧100cycle試験は14回成功後、配置変更のためユーザー指示に沿ってクライアントだけSIGINTで停止し、serverがREADYへ戻ったことを確認してから停止・再作成した。partial reportとログは /acceptance/windows-bind-100-cycle-report.json、windows-bind-server.log、windows-bind-graph-trace.logとして保持。KeyboardInterruptは意図的中断であり製品故障ではない。

nativeの1cycle結果は /acceptance/native-one-cycle-report.json。native100回試験を次のコマンドで開始済み（exec session40645）。再開時はまず `wslc exec ninfer-suspend-acceptance pgrep -af acceptance.py` と /acceptance/100-cycle-report.json のcomplete/error/cyclesを確認し、稼働中なら重複起動しない。長い試験の進捗は約5分間隔で確認する。

    wslc exec ninfer-suspend-acceptance python3 /acceptance/acceptance.py --cycles 100 --trace /acceptance/graph-trace.log --report /acceptance/100-cycle-report.json

P6とNInfer全体は未完了。次は100cycle結果と各cycleのcontinuation/cache/Graph/VRAM/RAMを確認し、P0のproduction ownership/workspace監査、P3の実Vision overlay KV loan/weight fallback再実行、production workspace poison、raw token一致、残る故障検証を進める。P0/P3の未チェック項目は免除しない。画像MCPはNInfer受入完了後。

この受入tool/configuration単位のcheck: interposer実compile/run、native1cycle PASS、WSLC Python3.12.3 py_compile PASS、git diff --check PASS。READMEへnative volume配置・Graph trace・直接HTTP試験手順と検証範囲を記載し、全対象diffを確認した。native100cycleはこのcheckpoint時点で稼働中であり、完了チェックは付けない。

### P6単体受入完了: ownership / workspace / continuation / endurance（2026-10-02）

RTX3090、WSLC CUDA13.1/GCC13/Python3.12.3、指定Qwen3.8-27B/rk8v4/160K/MTP3/Vision overlayで native100cycle が complete=true、errorなし、100/100 PASS。全回GPU usedはresume後22166MiBで一定、suspendedは334MiB/free23993MiB、解放21.3203GiB。Graph capture18/instantiate10/destroy0は試験中不変で、同じ10個のGraphExecだけをlaunchした。stored Responses継続出力・token数は全回一致、cached tokensは全回65。process VmHWM最大18,343,396KiB（17.49GiB）、VmSwap0。resume後RSSは12.64〜12.81GiB、最後10回は13,427,864〜13,428,324KiBに収まり、継続的な増加は観測されない。

平均suspend3.6758秒（3.1981〜5.3813）、resume7.9281秒（7.3991〜8.8503）、weight pipeline6.0855秒（5.8761〜6.7709）。復帰直後の継続要求TTFTはserver.logのreq#6〜105の100件で平均38.899ms（32.8〜54.4ms）。reportは /acceptance/100-cycle-report.json。単発21.6秒はresume単独の値ではなく、旧Windows bind配置の全iteration値だった。

#### 所有権・stream・arena外allocation監査

| 領域 | 実所有者とview | 容量・physical bytes | relevant work |
|---|---|---|---|
| WeightArena（overlay） | Modelのartifact backingにあるEvictableWeightPool。MaterializedArtifact/Parametersのarena/Tensorはborrowed view | HTTP構成physical17,918,066,688。直接Engine構成logical17,904,356,352 | materializerのtransfer streamをupload完了まで同期。idle Vision window/weight evictionなし |
| persistent（overlay KV tier） | ProgramImpl::kv_arena（EvictableKVPool）が全persistent backingを所有。persistent DeviceArenaはborrowed view | HTTP whole snapshot logical4,754,642,176、physical4,756,340,736。直接Engine構成logical4,726,981,888 | compute/transfer/Vision/hybrid restoreをidle判定し全drain。request/transaction/lease/replay完了が前提 |
| workspace_storage | ProgramImplのowning DeviceArena | logical216,270,848、physical218,103,808 | 同じrelevant streamsをdrain。fresh backingでper-unit scratch/Vision bridgeのみ書き直す |

HTTPと直接Engineでは公開・sampling等の予約条件が異なるためpersistentのlogical容量は異なる。直接Engineの量をHTTP snapshot容量へ流用しない。fault probeは実行構成のmemory_summaryからlogical容量を取得して全snapshotへ注入する。

src全体のcudaMalloc/cuMemAlloc系呼出しを検索し、直接device allocationはCore arenaとpipeline stage_linkに限定されていることを確認した。single-GPU suspendはpipelineを起動時拒否するためstage-link bufferと追加rank arenasは作成されない。Tensorは非所有view。残るcontext/modules/Graph/streams/eventsのdriver資源は維持する。HostKV、HostState、pinned ingress/egress、Vision result/weights、limited weight-window mirrorはHost側所有物で維持し、17GiB weight全体mirrorは追加していない。materializer stagingは64MiB×最大4slotとalignment paddingで、復帰時に生成し終了後解放する。

Programのstartup persistent bindingsを追跡し、KV/slab tables、StateImages、GDN replay/fold、DFlash persistent state、RoundState/Frame、prefill_hidden、sampling/token counts/grammar controlsが全persistentへ属することを確認した。workspaceはstartupのOp容量計画からper-unit WorkSpanとtransient Vision bridgeを提供し、ordinary prefill/decode、MTP verify/draft、Vision input/window/bridgeは読取り前に書き込む。request/context materialization/Vision windowが残る間はsuspendしない。境界を越えるCPU cache/page/state/lease/sequence metadataはそのまま維持される。新たなsnapshot領域やarena分割は不要だった。

#### 実Programのfresh workspaceとoverlay再実行

追加standalone target `ninfer_qwen3_5_suspend_real_test` は明示artifact/PNG/KV容量で実Engineを使用する。backendはrk8v4、FP16 GDN、MTP3、Vision overlay。acceptance-only backing-probe.soでphysical workspace全218,103,808 bytesを各resume後83/84/85で汚し、同じworkspace VAを観測した。製品debug APIや通常initializationは追加していない。

- KV163840: prefix reuseを無効にし、赤/青224px PNGを必ず再encode。3/3回、text9tokensとVision4tokensのraw token vectorが対照に完全一致。overlay_windows=1、exclusive=0で実KV loanを確認。
- KV2048: 同じ画像と対照で3/3回raw tokens一致、overlay_windows=1、exclusive=1で実weight fallbackを確認。fallbackを強制するための既存startup設定差であり、160Kの容量受入は前記100cycleで行った。
- 同じKV163840で別プロセスgpu-borrowがsuspended中に20GiBをcudaMalloc、全bytesをfill、先頭/末尾をD2H確認しfreeして終了。その後のtext/Vision一致を3/3回確認。
- 4096px/16384 merged tokensの画像frontierをcachedにした対照をsuspend前に繰返して安定性確認。resume後3/3回、16403 cached tokensを維持してraw token vector一致。最終Default retention構成でもworkspace汚染＋別プロセス20GiB使用を各回行い3/3回一致。
- 各direct Graph traceは最初のlaunch以降capture/instantiateなし、同一GraphExecのみをlaunchし、destroyはEngineの通常終了時だけ。borrow/cache-final traceにも同じ条件を確認した。

最初の大画像HTTP比較ではcold出力とcached出力の不一致があった。直接Engineで最初のsuspend前に cold_equals_cached=0 を再現し、suspend flagなしの既存allocation経路（control mode）でも同じ差を確認した。これは今回のrestoreの影響ではない。比較は同じcached実行経路のsuspendなし対照と復帰後のraw token列に揃え、完全一致した。cold/cache間の数値差の原因・解消はこのresidency変更では扱っていない。画像入力を実際に再encodeする受入は別途prefix reuse無効で完了している。

#### 故障・RAM・検証範囲

実モデルのone-shot注入はhost/d2h/h2d/unmap/release/access全6ケースPASS。Host確保失敗はbacking/snapshotを変えずREADY、生成一致とsuspend再試行成功。D2H失敗はbackingを解放せずERROR。H2D/access失敗はsnapshotを保持してERROR、推論拒否。workspace unmap失敗はpersistentを先に解放したpartial状態、pool release失敗はunmap後handle残存状態を実際に通し、どちらもERROR/snapshot/診断保持、正常destructor終了。Coreのcreate/map/access途中cleanupはP1 GPU testで検証済み。artifact read/immutable-source failureはP2/P4/P5で検証済み。永久的なCUDA context喪失の実機注入は行っていない。

追加試験中、Windows Win32_OperatingSystemを約2秒間隔で106sample取得。total49,674,076KiB（47.37GiB）、最低free6,309,560KiB（6.02GiB）、観測最大使用41.36GiB。OS/既存IDE・agent等を含む実ホスト値であり、process RSSだけで予算を判定していない。WSLC guest MemTotal24,285,784KiB（23.16GiB）。HostKV8GiB/State8とoverlay pinned Vision295,719,424 bytes、window capacity1,107,296,256 bytes、whole snapshotを含めNInfer単体が安定した。これはsampled peakであり瞬間的peakの厳密上限ではない。OpenCodeハーネスと画像runtime同時常駐はこの単体試験に含めず、今回のNInfer完了条件には含めない。

関連実装とdocsのdiffレビュー、git diff --check、追加C++ target/link、probe g++ build、gpu-borrow nvcc buildが成功。明示model/imageありの実GPU試験が上記PASS。引数なしCTestは意図したskip77（0.62秒）でありGPU PASSの代用にはしていない。P4/P5のfixture/options/HTTP checksは既にPASSし、その後製品実装は変更していない。全suite/Windows native、未用意DFlash/DFlash2 artifact、Hybrid cache、resident/cpu Visionの実モデル組合せは未実施。通常none backendはP4の生成fixtureで確認済み。今回の実モデル受入対象は指定single-GPU/C1/MTP3/overlay。

この受入時点では画像MCP実装を次に予定していた。最新の再開地点は末尾の監査修正欄を参照。NInfer standalone受入は完了し、P0/P3残項目も完了。画像MCP連携は最新ユーザー指示により対象外。ninfer-suspend-acceptanceは停止済みで100cycle結果を保持、ninfer-suspend-realはsleep infinityで現在GPUモデルなし、ninfer-suspend-devの差分build成果物も保持。必要時 wslc start ninfer-suspend-acceptance で同じnative volume構成のHTTP serverを再開できる。生ログはE:\koji\work\20260813\NInfer\tmp\suspend-acceptance、repoにbinary/model/logはcommitしない。

### 監査指摘修正（2026-10-02）

- Weight/KV両poolのattach_backingで、複数pieceの途中のcreate/map/access失敗時に、そのattachで作成したmapping/handleを即時rollbackする。unmap失敗はmappingとhandleを保持、release失敗はhandleを保持し、他pieceのcleanupは継続する。poisoned/ERROR契約は維持し、保持資源のみdestructorで再試行する。
- CPU-onlyゼロ出力submissionの最終READY確認とhandle作成をEngineCoreのqueue lock内で行い、suspend/resumeのstate publicationと同じ受付境界へ接続した。自動resumeやGPU実行は追加しない。
- Linux専用standalone ninfer_suspend_pool_failure_testを追加。両poolとも正常same-VA再attachを確認した後、第2pieceのcreate/map/access失敗 × cleanup成功/unmap失敗/release失敗の18ケースを実CUDA driverとlinker wrapで検証する。即時解放、保持資源の正確なaccounting、destructor後のmapping/handle残存0を確認。
- ninfer_model_suspend_testに1000件のゼロ出力submitと10回のsuspend/resumeの並行試験、各SUSPENDED境界での受付拒否を追加した。実行順によりREADYで受付済みのhandleは遷移後にwait可能（GPU資源は使用しない）。
- SPECの受付・partial-map cleanup契約、serving説明、tests READMEを更新。P6 docsは既に実装済みのため完了チェックを補正。最新ユーザー指示に従い、画像MCP連携検証をNInfer完了条件から除外した（未実施をPASSとは扱わない）。
- 検証環境: 既存ninfer-suspend-dev、WSLC Linux、RTX3090、CUDA13.1/GCC13。prepare-build.sh後、-j8でninfer、ninfer-serve、ninfer_model_suspend_test、ninfer_model_residency_http_test、ninfer_suspend_pool_failure_testの差分build/link成功。build logは /tmp/ninfer-suspend-audit-fix-build.log。
- チェック: pool failure 18ケースPASS（最終0.33秒）、Engine受付競合/通常生成/遷移/障害/破棄PASS（最終18.43秒）、HTTP管理API PASS（2.82秒）。全差分（新規testを含む）レビュー、git diff --check実施。
- 未実施: 全suite、Windows native、100cycle実モデルの再実行。既存100cycle報告は保存済みであり今回再実行した結果ではない。Coreの旧2テストは共有ninfer_testsがこの開発treeに未生成のため監査時にNot Run、新規standaloneで両poolの正常attachと失敗cleanupを検証した。今回の修正は数値演算・snapshot形式・正常upload経路を変更しない。
- 残作業: このNInfer監査修正の範囲ではなし。画像MCPは別作業。既存開発container/build treeと受入モデルvolumeは保持し、必要なら末尾記載のbuild pathと各standalone CTest名で再検証できる。

### 再監査: ゼロ出力受付の決定的な回帰検証（2026-10-02）

- 指摘: 1000件の並行submitは成功/Unavailableの双方を許容し、初回availability確認から最終admissionまでの間にsuspendが完了する順序を保証していなかった。
- 修正: Linuxのninfer_model_suspend_testだけにlinker --wrapを設定し、既存runtime::resolve_sampling(ModelSamplingDefaults, SamplingMode, SamplingOverrides)の処理後で、指定したsubmitスレッドをpromise/futureで停止する。初回is_availableはREADYで通過済み、その間にメインスレッドがsuspendを完了し、submitを再開してUnavailableによる拒否を必須とする。同期点到達は10秒timeout、例外時も停止スレッドを解放・joinする。製品コードや公開APIへテストフックは追加していない。従来のstress試験も維持する。
- 検証: ninfer-suspend-dev / WSLC Linux / RTX3090 / CUDA13.1 / GCC13、prepare-build.sh後、ninfer_model_suspend_testを-j8で差分build/link成功。復元済み製品コードで同実行ファイルの全ケースPASS、追加caseは「PASS deterministic zero-output admission rejection」を出力。
- Mutation検証: hostリポジトリを変更せず、コンテナ内 /build/src/src/runtime/engine/engine.cppのゼロ出力最終accept_immediate_submission呼出しだけを一時的に迂回し、同targetをrebuild。テストはexit1、理由「zero-output submission bypassed final residency admission」で必ず失敗した。finallyで元のsource bytesを復元し、再build後の同testはexit0/PASS。通常版とmutation版の違いをテストが検出することを確認済み。
- 記録: /tmp/ninfer-suspend-admission-mutation-build.log、/tmp/ninfer-suspend-admission-mutation-test.log、/tmp/ninfer-suspend-admission-restored-build.log。ローカルの再現補助は .cache/suspend-admission-mutation.py（検証用、commit対象外）。tests READMEへ決定的な実行順と検証範囲を追記。全diffレビュー・git diff --check実施。
- 制限: linker wrapを使う決定的caseはLinux専用。Windows native/全suite/実モデル100cycleは今回未実施（製品コード変更なし、テストだけの修正）。SPECの受付契約は変更不要。画像MCP連携は引き続き対象外。
- 残作業: 今回の再監査指摘は解消済み、NInfer suspendの修正範囲に未完了項目なし。再検証は ninfer_model_suspend_test targetをbuildし、その同名CTestまたは実行ファイルを実行する。既存container/build tree/model volumeは保持。
