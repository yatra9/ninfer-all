# ninfer ローカル動画・chunked Vision 統合計画

作成日: 2026-10-01

## 0. 実装進捗と再開地点

- 作業ブランチ: `ninfer-all` リポジトリの `ninfer-video`。
- **2026-10-01 ブランチ全体監査のP1不具合: 修正・実機回帰完了**。途中で寸法またはpixel formatが
  変わる動画は初回索引で拒否する。遅延Readerの入力・ファイル変更エラーをtyped `RequestError`へ
  変換し、Programが失敗レーンを清掃した後はEngineがResourceManagerの論理所有権だけを解放する。
  二重abortによる`Program did not consume aborted sequence`とworker全体のfailed化を解消した。
- 全体監査の再検証: WSLC/GCC 13で共有ライブラリを再buildし、`video_pipeline_behavior`、
  local-video URL、local-video plan、vision patchifyのCPUテスト成功。RTX 3090・指定Qwen3.8-27B・
  context/KV 163840・rk8v4・MTP3・overlay・chunk上限16384で、通常data-URI動画＋local動画2本の
  混在要求はstream/non-streamともHTTP 200（streamは正常DONE）、同時送信したtext要求もHTTP 200。
  98,304 Vision token／99,403 prompt tokenの動画は約100.7秒でHTTP 200、上限超過はHTTP 413。
  これはmax-concurrency=1の検証であり、複数GPU laneの同時実行を検証したものではない。
  監査用サーバー`ninfer-branch-review`は停止済み。再現スクリプト・要求・応答・JSONLは
  `../.gpu-validation/audit/branch-review.*`と同ディレクトリの`*-review.*`等に保存した。
- 修正後の実機回帰: WSLC製品image
  `sha256:4e218aca59b70e3d10f5e326f2f0bc1a39d3c7eda1558c8b080958624943e82f`をRTX 3090、
  指定Qwen3.8-27B、context/KV 163840、rk8v4、MTP3、Vision overlayで検証した。65,207-tokenの
  動画要求を実行中に、索引済みで待機中の別動画を差し替えたところ、対象要求だけがnon-streamで
  HTTP 400 `invalid_media`、streamでは既存SSE error要素になった。各失敗直後のtext要求は同じ
  resident EngineでHTTP 200 / `OK`。worker crashはなく、可変寸法TSもprepare時HTTP 400となった。
  製品build/link、正式local-video payload test、video-labの`video_pipeline_behavior`も成功した。
- **次の具体作業**: 修正はcommit `efd5bd03`まで完了し、`ninfer-all` working treeはclean。
  新しい監査指摘か追加要件がなければ本計画の実装を完成扱いにできる。実機回帰はユーザー指定と同じ
  `max-concurrency=1`であり、複数GPU lane構成の性能・同時実行検証は追加範囲とする。
- 再監査修正完了: mixed promptでは最大local-video chunk 1個分を通常mediaと合わせて論理的に
  admissionし、合計が`media_live_bytes`を超えるrequestをRAM待機前に拒否する。通常payloadの準備完了後に
  実容量を予約し、全local動画/chunkが直列再利用するため、cache hit/singleflight producerを塞がない。
- 再監査修正完了: 複数mixed requestが通常payloadを相互に保持して予約待ちする循環を防ぐため、共有cacheごとに
  local-video promptのmedia準備を1件ずつadmitするpermitを追加した。通常mediaだけの準備並列性は維持する。
- 再監査修正完了: 予約state作成後に公開wrapper確保が失敗した場合も、state destructorだけが容量を
  返すよう所有権を一本化した。故障注入で保持512 bytes、計上512 bytesの一致を確認した。
- 再監査修正完了: 完成索引の再利用時にもrequest固有の`max_pixels`と`max_scan_frames`を再検査する。
  full/chunk Vision実モデルtestは、出力がbit一致する場合を含め全BF16要素のNaN/Infを先に拒否する。
- 再監査検証: WSLC製品build成功（image
  `sha256:312d24dc3290ee328d52ebb0f472ac7ee2161a3b86dc5bdb67c58eff6395a06a`）。正式H.264
  payload testで索引再利用上限、mixed-media予約、chunk exact比較を含め成功。検証imageは
  `sha256:543271f6ff6e87e39e8c5b09e4886e7397fd39fe49c8b9ac0df404659e52de69`。
  RTX 3090/Qwen3.8-27B実モデルの81,920 BF16要素比較も成功し、cosine 0.999908、RMSE
  0.0133091、worst-token cosine 0.999886、normalized RMSE 0.028944を再確認した。
- 再監査検証: mixed-media境界、local-video準備permit競合を含む正式payload test成功。検証imageは
  `sha256:b23179f7a42fbacdec2e3d3b8efdba15294d60b507a5e8062a4df243812f4bbc`。最終変更を含む
  WSLC製品build/link成功、imageは
  `sha256:8f17b93339964c0651717fa3511f212d915c45a00a581264a3ce5b1daeee217b`。
- 前回修正はcommit `e89ba3c1`まで完了。現在の未修正点と次の具体作業は本節冒頭の全体監査記録を参照する。
- 完了: `video-lab` のCPU動画ライブラリとCLI。索引再利用、chunk読出し、CFR/VFR、
  seek、crop、scale、bwdif、エラー・キャンセルを検証済み。
- 完了: `ninfer-video` ブランチを作成。
- 完了: `ninfer-all/AGENTS.md` に、まとまりごとのcommitと、commit前のPLAN/SPEC/関連文書更新を必須化。
- P0設計調査完了: 現行Vision attentionは`segment_length=h*w`、`segment_count=t`で
  temporal groupごとに分離される。patch projection、position、MLP、mergerにもgroup間状態はない。
  ただし現行`VisionPrefillSession`はitem全体を一括encodeし、text prefill chunkはBF16 patch量を制限しない。
- P0文書完了: `docs/maintainer/local-video-pipeline.md`に、1 logical itemを完全なtemporal group範囲で
  実行分割する方針、product/frontend/Program/executionの所有権、prefix reuse無効化、失敗時の寿命を記録。
- SPEC更新完了: 初期版autotoneは`0`/`false`のみ受理し有効化を拒否すること、初回索引走査と
  deinterlacer最小lookaheadを`end_frame`即時終了の例外とすることを契約全体で統一。
- 検証状況: `git diff --check`成功。追加したmaintainer文書への2か所の参照と実ファイルを確認済み。
  このcheckpointは文書のみで製品ビルド対象変更なし。
- P0 control実装完了: `slice_vision_control`を追加し、item-wide controlから完全で連続した
  temporal group範囲のpatch、position、補間table、global scatter列をchecked viewへ切り出せる。
  全範囲、各単一group、末尾複数group、不正範囲をruntime mechanisms testで保護した。
- 検証状況: WSLC製品ビルド（`ninfer`、`ninfer-serve`）成功。slice APIの独立CPU probe成功。
  正式test sourceの`-fsyntax-only`成功。`ninfer_tests`全体ビルドは873 targetを再構築するため
  250/873で中止したが、失敗はなかった。GPU Vision出力のfull-item/chunk数値比較はP3まで未実施。
- P1 media共有完了: 検証済み実装を`src/media/local_video/video_pipeline.*`へ移し、
  `ninfer::media::local_video`名前空間と`ninfer_local_video` targetを追加した。Linux/WSLCのFFmpeg依存に
  libavfilterを加え、製品build/runtime imageにも必要なpackageを追加した。Windows nativeは対象外のまま。
- ミニアプリ共有完了: workspace rootをDocker contextにして製品と同じ正本を直接buildする構成へ変更。
  旧header/cppの複製は削除した。`video-lab:dev` buildと`video_pipeline_behavior`（CFR/VFR、B-frame、
  interlace、seek、chunk/index reuse）が成功。image SHAは`e1a35039ee82738d93a408f52ecf6bb03be7a8726961f67bd0066e1bdd4b1aed`。
- 製品検証: `wslc build -t ninfer-all:ninfer-video .`成功。`ninfer`、`ninfer-serve`、
  `ninfer_local_video`をbuildし、runtimeにもlibavfilterを収録した。image SHAは
  `27c5357141a25cd7450d2fd6045ebad558f4f5f88e0bdd81cbaee82e3deea7bb`。
- P1 URL parser完了: `product/local_video/local_video_url.*`に`LocalVideoSpec`を追加し、scheme、
  authority/fragment、percent encoding/NUL、未知・重複query、int64 overflow、NaN/Inf、bbox、
  autotone、deinterlaceをdecode開始前に一度だけ厳密検証する。`skip_frame+1` overflowも拒否する。
- P1 path認可完了: root未設定を無効として拒否し、requested/rootのcanonical component比較、
  regular-file確認によりtraversal、文字列prefixが似た隣接directory、symlink escapeを拒否する。
- 検証状況: parser/path単独CPU testをGCC 13 C++20、`-Wall -Wextra -Wpedantic -Werror`で成功。
  有効な全parameter、通常URL非routing、異常query、root外、symlinkを検証した。
  `wslc build -t ninfer-all:ninfer-video .`も成功し、`ninfer_local_video_url` targetを製品buildで確認。
- P1 serve policy完了: `--local-media-root PATH`を追加。絶対container pathのみ受理し、既定の空値では
  `ninfer-video`を無効のままにする。usageとserve-options testに有効値・既定値・相対path拒否を追加した。
- 検証状況: option追加後の`wslc build -t ninfer-all:ninfer-video .`成功。image SHAは
  `5d4d40b7c3a197d502eb05f4bd2794d4b4b7bb426a4153b025d9e46173d2894f`。GPUなしのcontainerでは
  executable loaderが`libcuda.so.1`を要求するためhelp実行は不可だったが、serve targetのcompile/linkは成功。
- P1 request routing完了: Chat Completions `video_url`とResponses `input_video.video_url`が
  `ninfer-video`だけを`SourceKind::LocalVideo`へ分類する。imageと通常HTTP(S)/data URIは従来経路を維持。
  generation acquisitionでURL全体をparse/root認可し、canonical pathと選択条件を公開型
  `OwnedLocalVideo`へ移して`OwnedMedia::local_video`に保持する。bytes取得関数はこのkindを明示拒否する。
- 検証状況: 変更したChat/Responses/generation/media-acquireの4 sourceをCUDA 13.1 devel環境で
  `g++ -std=c++20 -fsyntax-only`成功。追加・更新したChat/Responses/serve-options test source 3本も
  同環境で構文検証成功。製品full buildは既存CUDA target 480/694のコンパイルが長時間停止したため中断し、
  typed routing後の最終linkは未確認（その直前のserve option checkpointでは製品build/link成功）。
- 既知の未接続点: frontendは`OwnedMedia::local_video`をまだmaterializeできず、実リクエストは未動作。
- P2 media metadata API完了: `VideoSource::plan()`を追加。再利用可能な完全索引から、指定範囲・skip後の
  source frame index、PTS由来timestamp、crop/scale/alignment後寸法をRGB decodeなしで返す。
  readerと同じframe/pixel/変更検知/cancellation上限を適用し、末尾範囲とstride overflowを安全に処理する。
- 検証状況: ミニアプリの同一共有ソースからplanning APIをbuildし、planのframe index/PTS/timestampを
  実decode結果とexact比較。plan後の索引再利用counterも確認し、CFR/VFR/interlace/seek/chunkを含む
  `video_pipeline_behavior`全体が成功。image SHAは
  `97a4e28fb68f5591ffcfc9f444f32df3d9c374801eaef27b0617f819c607264e`。
- P2 Qwen layout plan完了: 可変の最終出力寸法と選択frame timingから、raw patch grid、
  temporal group、PTS由来group timestamp、merged token総数、16Ki以下の連続chunk境界を作る
  `LocalVideoPromptPlan`を追加した。1024x768/256 frameは128 group、98,304 token、7 chunkとなる。
  640x480、768x1024、奇数末尾複製、総量超過、1 group超過、非整列寸法もtestで固定した。
- 検証状況: `git diff --check`成功。現在のホストセッションでは`wslc.exe`が
  `ERROR_FILE_NOT_FOUND`で起動できず、新testのcompile/runは未実施。実装はCPU専用で正式test targetへ登録済み。
- P2 frontend接続完了: 公開`OwnedLocalVideo`をbytesへ戻さずfrontend内部のtyped mediaへ渡し、
  `VideoSource::plan()`を一度実行して共有source、reader options、`LocalVideoPromptPlan`をprepared promptに保持する。
  `count_tokens`もRGB decodeなしで同じ索引・PTS・placeholder layoutを使う。local動画を含むpromptは現段階では
  cross-request prefix reuseを無効化した。Windows native経路は明示拒否し、WSLC sourceだけを条件付きlinkする。
- P2 budget分離完了: 通常image/videoの既存32,768 aggregate上限と16,384 item上限を維持しつつ、
  local動画には独立した98,304 aggregate上限と16,384 execution chunk上限を適用する。mixed mediaでも
  通常media budgetへlocal動画のtokenを誤加算せず、prompt/context token数には全placeholderを含める。
- 検証状況: `git diff --check`成功。現在のホストセッションではWSLC/compilerを起動できないため、
  この接続checkpointのcompile/linkと正式testは未実施。
- 既知の未接続点: prepared promptのlocal動画slotにはeager BF16 payloadがなく、現行Vision sessionは
  まだそれを実行できない。実リクエストはVision実行時点まで到達するが完走しない。
- P3 chunk payload reader完了: prepared sourceからReaderを一度だけ作り、plan順に各chunkをdecodeして、
  source index/PTS/timestamp/寸法をprepared metadataと再照合した後、既存Qwenと同じ`value/127.5-1`の
  BF16丸めとtemporal/channel/spatial patch順序へ変換する`LocalVideoPayloadReader`を追加した。
  奇数末尾は動画全体の最後のgroupだけ最終frameを複製し、chunk末尾は常に完全groupなので複製しない。
  payload寿命は1 chunk単位で、Readerは順方向にだけ進み、全RGB/BF16を保持しない。
- P3 execution slice接続完了: local動画の1 logical itemを、prompt-visible token/span/positionは変更せず、
  prepared chunkごとの複数`VisionUseSpan`へ展開する。各useにtemporal範囲とchunk番号を保持し、
  `build_vision_execution_control()`がfull controlを`slice_vision_control()`で連続group範囲へ変換する。
  通常mediaは従来どおり1 use/full controlのまま。
- P3 遅延供給接続完了: request validation/workspace検査はlocal動画だけnull eager payloadを認め、各chunkの
  patch/token量で16Ki workspaceを検査する。Vision sessionはuse開始時に`LocalVideoPayloadReader`から
  対応payloadを生成し、GPU/overlay/CPU Visionへ渡す。local動画の非同期先読みは初期版では無効にし、
  active BF16 payloadを1つに制限する。service workはchunk数をprefill splitとして数える。
- control test更新: image full itemと2-group videoを、image＋video group 0＋video group 1の3 execution
  controlへ展開し、patch offset/countとglobal scatter indexを確認するcaseを追加した。
- P4 budget option完了: `--local-video-max-tokens N`（1..98,304、既定98,304）をserve CLIから
  `EngineOptions`、Qwen frontend、Processorへ伝播する。`--vision-max-merged`は従来mediaのitem上限に加え、
  local動画の実chunk上限として使うため、CPU residency等で小さくした場合もchunk計画が一致する。
  serve help、`docs/serving.md`、既定値・有効値・0/上限超過のoption testを更新した。
- P4実行制御完了: Engineのrequest cancellation flagとdeadlineをmaterializationから全Vision residencyへ
  保持し、local動画Readerのdecode checkpointとpatchify行境界へ`PreparationControl`として伝播する。
  decode途中のキャンセルは`Cancelled`、期限超過は`QueueTimeout`となり、失敗reader破棄後の次requestは
  同じ`VideoSource`と索引を再利用して正常に読めることを実動画testで確認した。
- 検証状況: H.264 7-frame fixtureによるstandalone payload test成功。変更したVision execution、通常
  materialization、hybrid materializationの3 translation unitをCMake相当フラグでWSLC構文検証成功。
  製品full targetは依存CUDAを205/627までエラーなく再構築した時点で、変更箇所の直接検証へ切り替えた。
- P4 request間索引cache完了: frontend processで最大8 `VideoSource`をLRU保持し、canonical path、size、
  mtimeが一致する後続requestは完成済み索引を再利用する。ファイル変更時は新sourceへ差し替え、evictionや
  差し替え後もactive requestはshared ownershipで安全に完走または変更検知エラーとなる。
- 検証状況: frontend/local prepare/processorの3 translation unitをWSLC構文検証成功。H.264 fixtureを
  2 requestでprepareし、source identity一致、index build 1回、reuse counter増加を確認した。
- P5 local動画diagnostics完了: request preparationに各動画の最終寸法、選択frame数、chunk数、Vision
  token数、先頭/末尾source index・PTS・timestamp、index build/reuse累計、初回走査時間を保持する。
  JSON request logの`preparation_seconds.local_videos`へ出力し、pathや動画内容は記録しない。
  request log schemaをv28へ更新し、`docs/serving.md`へフィールド契約を記載した。
- 検証状況: local prepare/processor/frontend/request logとrequest-log test sourceをWSLC構文検証成功。
  H.264 payload testでprepared diagnosticsのbuild/reuse/scan timing保持を含めて再実行成功。
- 最新製品image build成功: RTX 3090が空き（13 MiB使用）であることを確認後、WSLC full buildを
  中断せず待機し、`ninfer`、`ninfer-serve`、`ninfer_local_video`、`ninfer_local_video_url`のcompile/linkと
  runtime image作成が終了コード0で完了した。413修正を含む最終imageは`ninfer-all:ninfer-video`、SHAは
  `4751c1d402cc6c56165dacbb411872efa1b99f53f0a61f0219b52dea6b615e76`。
- 既知の制約: prefix reuseは無効なので途中chunkから再開しない。
- 実モデルGPU受入成功: `ninfer-all:ninfer-video`をRTX 3090で、提示済みQwen3.8-27Bモデル、
  max-context/KV 163,840、rk8v4、MTP3、overlay、chunk上限16,384で起動。96x64・8 frame動画は
  24 Vision tokenとして2回応答し、2回目はindex build 1のままreuse 2、prepare 86.7 ms→1.74 msとなった。
- 96Ki受入成功: 1024x768・256 frame H.264を98,304 Vision token、7 chunkとして処理し、prompt
  99,403 tokenから正常応答した。prefillは約99秒・約1.16k tok/s。直後の小動画requestも成功し、
  overlay/Program資源が次requestへ回収されたことを確認した。
- 実server上限回帰完了: 1024x768・258 frameは`media_budget_exceeded`としてOOM前に拒否される。
  初回実測でHTTP 400を検出し、Request/取得双方のmedia budget mappingをSPECどおり413へ修正した。
  修正版image（SHA `4751c1d402cc6c56165dacbb411872efa1b99f53f0a61f0219b52dea6b615e76`）を
  buildし、同じ実server/requestでHTTP 413とerror bodyを再確認した。commitは`8d15289c`。
- 実server cancellation回復完了: 96Ki requestをclient側3秒timeoutで切断し、serverがcancelledとして
  約3.7秒で終了、output 0となることを確認。直後の小動画requestが正常応答し、Reader、Program、overlay
  資源が次requestへ回収された。
- GPU chunk数値比較完了: 同じ256x256・8 frame（256 Vision token）をoverlayで上限64の4 chunkと
  上限256の1 chunkとして別engineで実行。greedy first tokenは同じ`The`、選択token logprob差0.07572、
  top-20候補は19個共有した。既存resident対CPU real test基準（token一致、選択logprob差0.25以下、
  top-8共有6以上）を満たす。prompt時間は4 chunk 914 ms、1 chunk 723 msだった。
- WSLC部分build検証完了: 権限付きWSLC環境で変更した製品translation unit 12本
  （local plan/reader/processor/frontend、Vision execution/control/request planning/materialization、
  Engine option、serve option/generation）をCMakeの実compile commandで全てcompile成功。検証imageは
  `bdea34ed5f0733d723314000ba9f21bce414be689a116359ce13cee3ff1f2fd2`。
- local layout test実行完了: 初回実行で「1 group超過」caseが4096x4096＝ちょうど16,384 tokenだった
  test入力誤りを検出し、4128x4128＝16,641 tokenへ修正。再実行で全case成功した。
- residency初期化修正: 検証後の実行経路再監査で、local payload reader配列の初期化がoverlay constructorに
  しか入っていないことを検出。resident/overlay/CPUの3 `VisionPrefillSession` constructorすべてで
  prepared media item数に初期化し、resident/CPUで最初のlocal chunk参照が範囲外になる問題を修正した。
- test source構文検証完了: `test_runtime_mechanisms.cpp`と`test_serve_options.cpp`を同じCUDA 13.1
  WSLC imageのGCC 13 C++20で`-fsyntax-only`成功。
- P5 frontend error分類完了: local layout計画にtyped errorを追加し、総token/1-group chunk上限超過を
  `MediaBudgetExceeded`、不正寸法・timing・decode/index失敗を`InvalidMedia`へ変換する。cancel/deadlineの
  `RequestError`と`bad_alloc`は変換せず維持する。変更source/testのGCC 13構文検証とlayout test再実行に成功。
- patchify共通化完了: 通常のimage/video eager経路とlocal動画chunk経路が、同じ
  `append_vision_patch_pair()`を使ってRGBからQwen BF16 patchを生成するようにした。これにより正規化、
  BF16 round-to-nearest-even、channel/temporal/y/x順序を二重実装せず、両経路の入力layoutを一致させた。
  48x32の2 frameから非原点patchを生成し、独立計算した全1,536要素とのexact比較、出力offset保持、
  寸法不一致・範囲外patch・不完全RGB・出力不足の拒否を行うCPU testを追加した。
- patchify検証完了: CUDA 13.1 WSLC imageのGCC C++20で新testをcompile/runし成功。変更した
  `processor.cpp`、`local_video_prepare.cpp`も同じ環境で`-fsyntax-only`成功し、`git diff --check`も成功。
- local payload実動画回帰完了: ffmpegで生成した96x64・7 frame・H.264/MP4を、12 token上限により
  2 chunk（4 frame＋3 frame）へ分割して読出した。別Readerで全frameをdecodeして作った参照payloadと、
  2 chunkを連結した全BF16要素がexact一致した。最終奇数frame複製、chunkの順序違反・再読出し拒否、
  metadataで決めた可変寸法、index build 1回と複数Readerによる再利用も同じ回帰で確認した。
- 実動画回帰の検証環境: 製品sourceをCUDA 13.1 WSLC/GCC C++20でFFmpegへ直接linkしてtest executableを作り、
  ffmpegを含む既存`video-lab:dev` imageでfixture生成と実行に成功。正式CTestにもLinux/WSLC限定で登録し、
  ffmpeg executableがないtest hostではreturn code 77でskipする。C++ test sourceの`-fsyntax-only`と
  `git diff --check`も成功した。
- 監査修正完了: 初回index走査へ元requestのcancelを伝播し、local BF16 chunkを既存の
  `--media-live-mib` accountで予約・解放する。各chunkの返却前後でsource identityを検査する。
  path認可を型付き分類し、未存在404、root外403、資源上限413へ接続した。pixel上限とlive-byte上限の
  typed回帰、index cancel非公開、chunk間mutation拒否、payload exact一致をCPU実動画testで確認した。
- 修正後検証: `video_pipeline_behavior`、local URL test、local payload実動画testが成功。
  `ninfer-all:ninfer-video-audit`のfull製品buildが成功し、image SHAは
  `ddf28aa24af31c289109298f1344b1a5e9889379141d52072d1495e6920d853f`。
- host live-memory競合回帰完了: 共有accountをちょうど1 chunk分に制限し、2 requestのReaderが同時に
  chunkを生成した場合、後続requestが上限超過せず待機し、先行payload解放後に進行すること、最後に
  live bytesが0へ戻ることを実動画testで確認した。
- 修正版実serverのHTTP分類確認: RTX 3090実モデルserverで、未存在pathはHTTP 404
  `local_video_not_found`、root外pathはHTTP 403 `local_video_forbidden`、98,304 token超過動画は
  HTTP 413 `media_budget_exceeded`となることをresponse bodyまで確認した。検証serverは停止済み。
- Vision embedding直接比較を再監査・修正: 旧fixtureは256要素周期により全temporal groupが同一だった。
  groupごとにseedを変えた同一`[-2,2]`分布のhash入力へ変更し、NaN/Infを明示拒否した。RTX 3090と
  実Qwen3.8-27B Vision towerで、81,920 BF16要素の全体cosine 0.999908、RMSE 0.0133091、
  token最小cosine 0.999886、最大token NRMSE 0.028944を確認した。回帰条件は全体・token cosine
  0.999以上、全体RMSE 0.03以下、token NRMSE 0.05以下。差の原因はroute-level証拠なしに断定しない。
- 96Ki RSS peak実測完了: 163,840 context/KVのRTX 3090起動条件で、request前12,897,168 kB、
  VmHWM 13,259,916 kB、増分362,748 kB（354.25 MiB）。完了後12,970,668 kB、基準差73,500 kB
  （71.78 MiB）。98,304 Vision token、prompt 99,403 tokenのHTTP 200応答は100.41秒で完了した。
- 再監査4件を修正: 同一sourceのindex build待機を10ms単位でcancel/deadline確認可能にし、index生成は
  single-flightのまま維持した。FFmpeg interrupt callbackで元のRequestErrorを保存・再送出する。
  codec headerのsource pixel超過をdecoder open前にResourceLimit分類する。Vision fixture周期性と
  非有限値判定も上記のとおり修正した。CPU正式payload回帰と実GPU Vision回帰は成功。
- 再監査修正の製品WSLC build成功: `ninfer-all:ninfer-video-reaudit`、image SHA
  `86583238e536beb01cec645db43e5d45e6d1e6ce041c07bcb2629368d85fd1a8`。正式payload回帰は
  7-frame H.264 fixtureで成功し、CPU再現probeでも待機cancelとtyped pixel limitを確認した。
- 再監査修正はcommit `6e1ae9c9`、mixed-media待機修正はcommit `e89ba3c1`まで完了。
- 全体監査で見つかった遅延動画エラー後のEngine停止を修正した。索引時geometry/format検査、
  Reader例外のtyped化、Program失敗後のResourceManager論理レーン解放を実装し、製品buildと
  stream/non-stream実サーバー故障注入に合格した。commit `efd5bd03`、working tree clean、
  検証コンテナ停止まで確認済み。
- 人手確認用PowerShell clientを追加した。`tools/Invoke-NInferVideo.ps1`はAPI base/model、コンテナ内
  動画path、prompt、start/end/skip frame、bbox、scale、deinterlace、生成上限・samplingを受け取り、
  percent-encodeした`ninfer-video://`を含む非streaming Chat Completions requestを送る。回答、reasoning、
  finish reason、usageを表示し、request/raw JSON表示、API key、dry-run、pass-throughにも対応する。
- client検証完了: Windows PowerShell 5.1で構文解析、全動画optionと日本語promptのdry-run、無効path・
  frame範囲・bboxの事前拒否、localhost mock APIへの実POSTとanswer/reasoning/usage表示に成功した。
  `docs/serving.md`へURL契約・実行例・通常mediaと96Ki local動画のbudget差を、`tools/README.md`へ導線を
  追記した。commit `4d3c8497`、working tree clean。SPEC変更なし。次の作業は実際の利用者動画による
  対話的な内容確認のみ。
- 動画機能の仕様・計画をリポジトリで履歴管理できるよう、workspace直下の`SPEC.md`と`PLAN.md`を
  repository rootの`SPEC_VIDEO.md`と`PLAN_VIDEO.md`へ移動した。`AGENTS.md`のcommit checkpointと
  maintainer文書を含む旧path・旧名参照も新しい場所へ更新した。

この節を各commitの直前に更新する。完了項目、実行した検証、既知の制約、次の一手を、
会話履歴なしで作業を再開できる粒度で残す。

## 1. 成果物と対象

`SPEC_VIDEO.md` に基づき、`ninfer-serve` の既存 Chat Completions の `video_url` に
`ninfer-video:///videos/example.mp4?...` を追加する。完成済みの `video-lab`
ライブラリを利用し、指定フレームを順次読み、Qwen Vision に必要なchunkだけ前処理する。
通常の HTTP(S)、data URI、image、text 入力は既存挙動を維持する。

対象環境は WSLC/Linux コンテナ、RTX 3090、現行 Qwen3.8 27B `.ninfer` モデル。
Windows native 対応は今回の完了条件に含めない。既存の Windows build tree は変更しない。
本ファイルは実装計画であり、実装済みを意味しない。

完了の目標は、同じ起動条件の `--max-context 163840 --kv-capacity 163840`、
`--kv-dtype rk8v4 --gdn-state-fp16 --spec mtp --draft-tokens 3 --lm-head-draft`
および `--vision-residency overlay` で、合計98,304（96Ki）Visionトークンを持つ
ローカル動画リクエストを扱えること。GPU実測前に収容可能と断言しない。
本文、timestamp、特殊トークン、出力用予約もcontextに含めて事前検査する。

## 2. 決定事項と仕様の整理

- 初期対象は主にH.264/MP4。CFRもVFRも元フレーム番号とPTSを保持する。
- 初回は正確な索引を作るため全走査を許容する。以後、同じソースの索引を再利用する。
- 全動画bytes、全RGB、全BF16パッチをRAMへ事前展開しない。必要なchunkを元動画から読む。
- 隣接chunkではReaderを維持して順次decodeする。chunkごとにopen/seekし直さない。
- 画像一時ファイル、パッチのディスクspoolは導入しない。軽量索引と必要なembeddingはRAMを使う。
- `scale` とモデルのgrid alignmentで最終寸法を決める。後段で再resize・再samplingしない。
- autotoneは後回し。`0`/`false`のみ受理し、`1`/`true`は未対応として400を返す。
- 自動回転は導入しない。bboxはcoded pixel座標とし、回転metadataの扱いを説明する。
- `--local-media-root` 未指定なら新schemeを無効化する。WSLCではコンテナ内の絶対パスを使う。
- URLにchunkサイズは追加しない。モデル側のtemporal単位と実行上限から内部計算する。
- 入力・出力解像度は固定しない。1024×768は計算・受入試験の一例であり、対応サイズを限定しない。
  動画ごとのsource寸法、bbox、scaleから最終寸法を算出する。縦長・横長も同じ規則で扱う。
  同一動画の途中で寸法/pixel formatが変わる入力は、現行ライブラリに合わせ初期版では明示的に拒否する。

実装開始時にSPECの次の不整合を修正し、一つの契約に揃える。

1. §15に合わせ、§25.8・§26.10のautotone受入条件と有効化例を初期版向けに更新する。
2. §10.2/§26.12の終了条件に「索引作成の初回走査」と「bwdifに必要な最小lookahead」の例外を明記する。
3. §12のYUV ROIは最適化目標。初期統合では検証済みRGB cropを利用し、余分なRGB一時領域を計測する。

## 3. 現行コードで確認した制約

| 箇所 | 現状 | 必要な変更 |
|---|---|---|
| `src/serve/openai_chat_request.cpp` | video_urlを解析 | 新schemeの受理・既存形式維持 |
| `src/serve/generation_service.cpp` | `acquire_media`でbytesを取得 | acquisition前にローカル動画を分岐 |
| `src/product/media_acquire/` | URL/path/data acquisition | URL/spec解析とroot検証の境界 |
| `include/ninfer/types.h` | Engineへの入力・オプション | bytes以外の明示的な動画ソース契約 |
| `src/models/qwen3_5/frontend/processor.cpp` | sampling/resize/patchify/timestamp | 最終寸法のRGBと正確なPTSからchunk前処理 |
| `frontend/prepared_prompt.h` | 全itemのBF16 payloadを保持 | immutable動画計画と遅延payload供給 |
| 同上 | item上限16,384、prompt合計32,768 | 実行chunk上限と新経路の総量上限を分離 |
| `frontend/frontend.cpp` | 全量patch容量からminimum_liveを計算 | 新経路は有界chunkのlive memoryを計画 |
| `program/vision_control.cpp`、`program/vision_prefill.h` | grid・span・実行計画 | chunkと元promptの対応を保持 |
| `program/planning/startup.cpp` | Vision workspaceをitem上限から計画 | 総量96Kiをworkspaceに直結させない |
| `execution/vision.cpp`、`execution/vision_overlay.cpp` | 既存Vision実行 | chunk payload供給と寿命・overlay維持 |

`--vision-max-merged` は現行CLI/serveで64..16384に制限されている。
これはtext prefillのchunkサイズとは別で、単純に「Visionが常に16kずつ処理される」という意味ではない。
既存のprepared itemを分割するだけでは、前段の全BF16保持を解消できない。

## 4. トークンとメモリの設計

対象モデル設定がpatch_size=16、spatial_merge_size=2、temporal_patch_size=2の場合:

```text
tokens_per_temporal_group = (aligned_width / 32) * (aligned_height / 32)
total_vision_tokens = ceil(selected_frames / 2) * tokens_per_temporal_group
```

1024×768なら1 temporal group（2フレーム）=768トークン。
98,304トークンは128 group=256フレーム。320フレームは122,880トークン。
16,384以下に収めるchunkは最大21 group=42フレーム（16,128トークン）。
これらはモデルgridから導出し、固定フレーム数として埋め込まない。
一般には `groups_per_chunk = floor(chunk_token_limit / tokens_per_temporal_group)` とする。
同じトークン上限でも最終寸法によってchunkのフレーム数と収容可能な動画フレーム数は変わる。

98,304トークン全体を現在の形式でpatchifyすると、
`98,304 * 4 * 1,536 * 2 = 1,207,959,552 bytes`（1,152MiB）のBF16入力になる。
この全量保持を避ける。chunk上限は安全上限であり、毎回上限まで詰める必要はない。
初期は少数のtemporal groupから正確性を確認し、RAM/VRAMと処理時間を見て既定値を選ぶ。

以下を別々に管理する。

- 動画全体の軽量索引・選択metadata: O(source frame数)。画像を含めない。
- decode/filter/RGB/patch: O(chunk容量)。Readerの先読み・変換一時領域も計上する。
- Vision出力: main embeddingと既存経路が要求する追加出力を含め、dtype/hidden幅から計算する。
- text用KV/GDN/workspace: 既存context容量・prefill設定から計画する。

既存`--vision-max-merged`の挙動は維持し、新経路の実行chunk上限としても参照する。
新経路専用の総量上限オプション（案:`--local-video-max-tokens`、初期値98,304）を追加する。
image/通常videoを含むrequest合計とcontext上限を合わせて検査し、通常経路の既定budgetは変えない。
1 temporal groupだけで実行上限を超えた場合は、空間分割や暗黙の縮小をせずエラーにする。

## 5. 責務と所有権

### 5.1 再利用ライブラリ

検証済み`VideoSource`/`VideoReader`を単一ソースとして共有する。
`ninfer-all/src/media/local_video/`を正本とする配置へ移し、ミニアプリも同じCMake library targetを参照する。
公開headerからFFmpeg型、JSON、PNG、NInfer/CUDA型を排除する。
ミニアプリ単独ビルドも維持し、必要なbuild contextとDockerfileを更新する。
ninfer-allの通常の`wslc build -t ninfer-all .`で、外部の兄弟ディレクトリなしにビルドできる状態にする。
コードの二重コピー・手動同期は採用しない。

### 5.2 product/serving

schemeの解析、許可rootの検査、入力所有権、deadline/cancelの伝達を担当する。
モデル層へURL解析や任意pathのopen権限を持ち込まない。
Engineへの入力は、検証済みソースと選択条件を表す明示的な型を追加する。
既存OwnedMediaを偽のbytesやmagic filenameで流用しない。

### 5.3 frontend

動画索引・寸法から全体のgrid、選択タイミング、token span、MRoPE、context使用量を先に確定する。
これらをimmutableなprepared planとして保持する。画素データはこの時点で全量作らない。
Qwen固有のnormalize/patchifyとtemporal groupingはfrontend側のadapterが所有する。

### 5.4 runtime/program

requestごとのReaderと進行状態を持つ。準備済みplanを消費する際、必要なchunkのpayloadだけを取得する。
同じprepared planを再利用してもmutable Readerを共有しない。
GPU buffer/workspace/streamの所有権は現行Programに保つ。
入力供給は有界とし、GPUからの要求を待って進める。初期段階で非同期prefetchを増やさない。

### 5.5 索引・キャッシュ・ファイル寿命

server内に容量制限付きのソース/索引キャッシュを置く。画素や全patchは入れない。
同じファイルの索引をrequestをまたいで再利用できるよう、VideoSourceの寿命をrequestから分離する。
キーはcanonical pathとファイルidentity/更新情報。eviction後も利用中Readerはshared ownershipで生存する。
初回走査失敗・キャンセルはキャッシュに公開しない。

現行ミニアプリのファイル変更検査は主に開始/終了時である。
統合時はprepared plan作成後、chunkの利用前、処理終了時にも検査し、変更時はrequest全体を失敗させる。
同サイズ・同mtimeの変更までcontent同一性を保証するものではないため、path+mtimeを
既存SHA-256 media digestの代用品としてprefix cacheに流用しない。
初期版ではローカル動画を含むpromptのcross-request prefix reuseを無効化し、索引再利用とは切り離す。
強いcontent identityを後で導入する場合は別途検証する。

## 6. 実装順序と各段階の完了条件

### P0: 契約・接続箇所を固定

- SPECの前述不整合を更新する。
- `docs/maintainer/engine-architecture.md`の該当境界に沿って入力型・provider・Programの寿命を具体化する。
- 現行Vision attentionの分割単位、positional embedding、merger、追加Vision出力の依存を確認する。
- temporal group境界の分割がfull-batchと等価であることを小さい入力で示す。
- decoder scanはGPUなし、VisionのGPU実行は既存Engine/Program内、という境界を固定する。

完了条件: group独立性と、prepare→executeをまたぐ所有権が説明できること。
group間依存が見つかれば、勝手に独立扱いせず正しい分割境界へ計画を修正する。

### P1: mediaライブラリの共有とURL入口

- ライブラリを単一配置へ移し、FFmpegのlibavfilterを含むCMake/linkを接続する。
- scheme/path/queryを一度だけparseし、静的条件をdecode前に検査する。
- 未知・重複parameter、overflow、NaN/Inf、壊れたpercent encoding、NUL、fragment、非空authorityを拒否する。
- root配下判定はpath componentで行い、文字列prefix比較をしない。symlink escapeを拒否する。
- Chatの通常acquisitionより前に分岐する。新schemeの解析エラーを通常URLへfallbackさせない。
- Responses等は既存契約を維持する。共通コード変更で誤って許可範囲を広げない。

完了条件: ミニアプリの回帰テストが通り、scheme/root/parameterの成功・失敗がAPIから観測できること。

### P2: 動画計画と正確な時間情報

- 索引から選択数、最終寸法、group数、token総量を算出するための読み取り専用metadata APIを補う。
- PTS/time base由来のtimestampと元frame indexを全段階に伝達する。先頭時刻を0へ戻さない。
- 現行processorの2フレームtimestamp平均・末尾複製を再現し、入力PTSからgroup時刻を求める。
- 奇数枚のpaddingは動画末尾だけで行う。chunk末尾で独自paddingしない。
- promptのtimestamp文字列の既存表示丸めと、内部の正確なtimestampを区別する。
- token列、token_types、positions、rope_delta、spanを全体の順序で生成し、chunkで原点をリセットしない。

完了条件: CFR/VFR/非ゼロ開始/skip/奇数枚のmetadataとprompt layoutが参照結果と一致すること。

### P3: 遅延patch供給と小容量Vision統合

- prepared promptに全patchの代わりにtypedなlocal-video planを保持する経路を追加する。
- Readerから必要なtemporal groupを読み、最終寸法RGBをnormalize/patchifyする入口を追加する。
- existing smart-resize/samplingを新経路で再適用しない。
- group境界でchunkを切り、Vision出力とprompt spanを正しく接続する。
- 同一mediaを複数text prefill chunkが参照する間、必要なVision出力を保持する。
- 全RGB/全BF16保持がmedia cacheやminimum_live計算を通じて復活しないようにする。
- overlayの重み転送とtext復帰を現行寿命に合わせる。groupごとの不要な再転送を避ける。

完了条件: 小さい動画のfull-batchとの数値比較が通り、chunk変更で入力意味が変わらないこと。

### P4: 総量96Kiと実行予算

- 総量上限とchunk上限をfrontend、runtime、serve optionで一貫して扱う。
- 既存32768上限の参照箇所を追跡し、新経路だけの大容量許可を明示する。
- context使用量、frame/pixel/scan/host byte制限を事前検査し、途中の動的違反も同じ分類で通知する。
- 実行中の例外、切断、deadline時にReaderを停止し、Program資源を回収する。
- 単独requestから開始し、既存concurrency設定下でもhost/GPU予約量を超えないことを確認する。
- RSS、VRAM peak、初回index時間、再利用時間、decode/patchify/Vision/prefill時間を記録する。

完了条件: 対象RTX 3090の実モデルで96Ki Vision入力から応答でき、予算超過はOOM前に明示されること。

### P5: API回帰と利用手順

- malformedは400、root外は403、未存在は404、resource超過は413を基本に既存error型へ接続する。
- streaming開始前の失敗は通常HTTP error、開始後は既存SSE error規約に従う。正常終了に偽装しない。
- raw動画/base64をlogせず、解決寸法、選択数、PTS範囲、index reuse、chunk数、token数を診断可能にする。
- `docs/serving.md`、CLI help、ミニアプリREADME、Docker build/run手順を更新する。
- 完了した一時計画の内容は恒久ドキュメントへ反映し、PLAN_VIDEO.mdは役割終了時に整理する。

## 7. 検証マトリクス

| 対象 | 確認内容 |
|---|---|
| ライブラリ | 既存CFR/VFR/B-frame/open-GOP/mixed interlace/crop/scale/seekテスト |
| Reader | chunk=1,2,3,16、末尾部分chunk、EOF反復、例外保持、統計、キャンセル、破棄 |
| 索引 | 同じ動画のrequest間再利用、上限eviction、構築中断、ファイル変更 |
| URL/API | defaults、全parameter、encoded path、未知/重複、不正数値、root外、symlink |
| timing | 正確なPTS、非ゼロ開始、VFR、奇数padding、group平均、prompt表現 |
| patchify | 同じRGB列を使ったfull-batchとstreamingのBF16入力比較、二重resizeの不存在 |
| 可変解像度 | 640×360、1920×1080、1080×1920、非整列寸法、crop後の非標準aspect比。各寸法のalignment・token数・chunk境界を確認 |
| Vision | 同じpatch入力のfull-batchとchunked出力、必要な追加出力も比較 |
| prompt/prefill | token/position/span一致、text chunkがmedia spanを横断する場合 |
| cache | 同じpathの別内容をprefix一致と誤認しない、複数requestでReaderを共有しない |
| budgets | 16k近傍、32k超、96Ki、上限超過、1 group超過、context不足 |
| memory | 長い動画で全bytes/全RGB/全patch相当の増加がない。索引とembedding増加は区別 |
| serving回帰 | 通常HTTP(S)/data video、image、text、混在media、stream/nonstream |
| overlay | Vision後のtext/MTP処理、キャンセル後の次request、資源解放 |

画素・index・PTS・整数layoutはexact比較する。BF16境界を明示し、同じ演算ならpatchもexact比較する。
GPU出力はFP32参照または既存の独立した数値検証基準を使い、dtypeに応じた許容誤差を
事前に定義する。full-batchとの一致は分割等価性の証拠であり、モデル回答の自然さだけを正しさの根拠にしない。

## 8. 実環境の受け入れ手順

1. WSLCで対象C++/CUDAターゲットと関連テストをビルドする。
2. 指定モデルと現在の起動パラメータで小動画を処理し、従来の通常動画入力も回帰確認する。
3. 動画をread-onlyで`/videos`にmountし、`--local-media-root /videos`を付ける。
4. 1024×768・256選択フレームの動画で98,304 Visionトークンになることをmetadataで確認する。
   このケースは代表例とし、別解像度でも実際の最終gridからtoken数を計算し、総量上限付近を検証する。
5. 96Ki経路で応答・メモリ・時間を計測し、同一動画の2回目で索引再利用を確認する。
6. 上限超過、切断、処理中キャンセル後に次requestが正常動作することを確認する。

GPU検証は空きVRAMと既存serverの稼働状況を確認して行い、ユーザーのserverを無断停止しない。
モデルpathはユーザー提示の明示pathを用いる。GPU検証できなかった場合は、その段階を未完了として報告する。

## 9. 完了判定

P0〜P5の成果物が揃い、SPECの初期版受入条件、通常入力の互換性、chunk数値等価性、
有界pixel/patchメモリ、および対象実機の96Ki入力が確認できた時点で統合完了とする。
ミニアプリのCPUテスト成功だけではninfer統合完了とはしない。
性能の追加最適化（YUV ROI、prefetch、索引永続化、autotone）は、この受入条件を満たした後の別作業とする。
