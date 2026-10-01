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
- 次の作業: P2 materializerの既存destinationへのupload共通化と、ModelのReader/placement保持。
  P0の各arena正確容量・physical量とproduction workspace監査は継続中。予備実測用の
  `ninfer-suspend-budget` コンテナ（host port 18082）は停止済み。
- 各実装段階の完了時に、本節へ変更内容、実施した検証、未検証事項、次の作業を記録する。
- 2026-10-02に実装開始と作業単位ごとのcommitを承認済み。AGENTS.mdのsuspend checkpoint規約に従い、
  検証済み単位をcommitして問題がなければ次段階へ継続する。

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

- [ ] weight/persistent/workspaceの所有者、容量、物理確保量、stream、arena外allocationを一覧化する。
- [ ] workspace内のrequest境界を越える依存を監査する。依存があれば該当領域だけpersistentへ移す等で解消する。
  decode graph、MTP/DFlash、Vision bridge、context transfer scratchを対象とする。
- [x] 既存VMMテストを拡張または専用テストを追加し、capture/instantiate後にD2H、unmap、release、
  新規create、same-VA map、H2D、同一GraphExec replayを1000回以上反復する。
- [x] pointerを含むpersistent相当データとfresh workspaceを含め、値・アドレス・GraphExec同一性を検証する。
- [ ] RTX 3090・Qwen3.8-27B・context/KV 163840を暫定160K設定として容量を計測する。
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

- [ ] materializerを「初回storage/view構築」と「既存device destinationへのupload」に分離する。
  既存のread coalescing、pinned staging、async H2D、transcodeを共通経路として維持する。
- [ ] suspend有効時のみ、復元に必要なReaderとdevice placement情報をModel側の所有物として保持する。
  Readerのアドレスを安定させ、dangling `plan.source` やHostPlacement payloadの重複保持を避ける。
- [ ] multipart artifactも起動時と同じsourceから読み、destination容量・offset・alignmentを検査する。
- [ ] artifactはEngine存続中に変更しない契約を明記する。復元元不整合、破損、read errorはERRORにする。
  既存artifact検証を再利用し、毎回17 GiBの余分なmirrorや全量二重readを追加しない。
- [ ] weight parent/Parameters/既存GraphExecを作り直さず、起動時と同じdevice bytesを復元する。

出口条件: 通常・overlay双方のweight所有者で、同一VAへの復元と既存GraphExec replayが成功すること。

### P3: Programのpersistent snapshotとworkspace復帰

- [ ] persistentのlayout/constructor contractを保ったまま、通常arenaまたはKV poolへP1を接続する。
- [ ] `persistent.capacity()` 分の通常Host RAMをlazy allocateし、whole-arena raw D2H/H2Dを実装する。
  used bytes、live KV、lendable prefixだけのコピーに縮めない。
- [ ] workspaceをfixed-VA化し、fresh backingで必要な初期化のみ実行する。
- [ ] snapshotはSUSPENDED中と復旧判断に必要な失敗時に保持し、resume成功後に解放する。
- [ ] CPU側sequence/cache/lease metadataを維持し、既存continuationからの生成とoverlay再実行を検証する。

出口条件: persistent全bytesの復元とfresh workspaceでの推論が成立し、graph再構築が不要なこと。

### P4: Engine state machineとpublic契約

- [ ] EngineOptionsからModel/Program構築へenable flagを伝搬し、VMM capabilityをallocation前に検査する。
- [ ] §3の状態遷移、即時busy判定、worker排他、availability、終了処理を実装する。
- [ ] single-GPU Generation以外の非対応組合せは起動時に明示拒否する。flagなしの既存機能は維持する。
- [ ] queue投入との競合、同時suspend/resume、状態照会、失敗注入をモデルなしでも検証できる範囲でテストする。

出口条件: 非READY状態からGPU実行へ到達せず、busy操作がrequest完了待ちにならないこと。

### P5: HTTP、起動option、計測

- [ ] `--enable-model-suspend` とhelp/options testを追加する。
- [ ] POST `/v1/models/{model}/suspend`、POST `/resume`、GET `/residency` を追加する。
  model alias、認証、body検証、例外変換は既存serverの規約へ合わせる。
- [ ] SPEC §13のidempotencyを実装する。遷移中は409、ERRORの管理操作は500。
- [ ] Chat Completions、Responses、Anthropic Messages等の推論入口を503へ統一する。
  SSE開始前に拒否し、background requestを含む受付済み要求の存在もidle判定へ接続する。
- [ ] 管理APIと診断APIはSUSPENDED中も使用可能にし、health/readinessの意味を既存契約と整合させる。
- [ ] SPEC §15のbytes・時間・last_errorを公開する。読込、H2D、snapshot、map/unmap、totalを区別する。
  statusは一貫したCPU snapshotから返し、unmapped device memoryを読み取らない。

出口条件: HTTP schema/状態遷移テストが通り、resumeを明示的に呼ぶまで推論を再開しないこと。

### P6: 実機受入とドキュメント

- [ ] 下記検証表を実施し、利用した環境・設定・測定値・未実施項目を本書へ記録する。
- [ ] `docs/serving.md`、`docs/cli.md`、`docs/maintainer/engine-architecture.md`、関連するmemory/overlay説明を更新する。
- [ ] OpenCode → tool call → suspend → 画像生成 → 画像生成側のGPU解放 → resume → 画像評価を確認する。
  orchestratorは既存連携または最小の検証手順を使い、汎用model managerは実装しない。

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
