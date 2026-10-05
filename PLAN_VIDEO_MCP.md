# NInfer Video MCP 実装計画

更新日: 2026-10-06。状態: 実装中。WSLC 増分ビルド環境を準備済み、共有動画サービスを実装中。

## 1. 成果物と前提

正本は [SPEC_VIDEO_MCP.md](SPEC_VIDEO_MCP.md) と [mcp/tool-schema.json](mcp/tool-schema.json)。本計画は作業順序、既存コードとの差分、検証方法、未確定事項を管理する。

- 既存 `ninfer-serve` の同一プロセス・同一ポートに `/mcp` を追加する。
- `get_video_metadata`、`resolve_video_time`、`inspect_video` を schema どおり公開する。
- MCP と既存動画入力は同じ source metadata/index を共有する。
- `inspect_video` は resource link を返す。別 LLM 呼び出し、GPU 推論、全フレームの RGB キャッシュを行わない。
- 開発・ビルド・テストは Linux/WSLC。ユーザー指定の `FROM ninfer-all:build` を使用し、開発イメージを `ninfer-all:dev`、コンテナ名を `ninfer-all-dev` とする。ホストの Windows ビルドツリーには手を加えない。
- 実装開始をユーザーから依頼済み。AGENTS.md の `ninfer-video-mcp work checkpoints` に従い、検証済みの独立した作業単位ごとにコミットし、その前に本計画・仕様・関連文書を更新する。判断が必要な問題がなければ、コミット後も次の作業へ進む。

## 2. 調査済みの既存実装

| 場所 | 現状と利用方針 |
| --- | --- |
| `src/media/local_video/video_pipeline.*` | `VideoSource`、metadata probe、presentation PTS index、keyframe seek、`VideoReader` が実装済み。index は整数 PTS を保持し、EOF まで構築して公開する。同一 source 内での同時構築抑止、変更検出も存在する。これを拡張する。 |
| `src/models/qwen3_5/frontend/local_video_prepare.*` | `LocalVideoSourceCache` に size/mtime と LRU を使う共有機構がある。MCP から model frontend に依存させず、source cache を共通層へ移す。 |
| `src/models/qwen3_5/frontend/processor.*` | 動画準備に source cache を渡している。生成元から呼び出し先まで参照を更新し、MCP と同じ service に到達させる。 |
| `src/product/local_video/local_video_url.*` | URI parser、型付き選択条件、`--local-media-root` による canonical path 認可がある。builder は未実装。ここへ追加する。 |
| `src/serve/generation_service.cpp` | URI を認可し、型付き動画入力へ変換する。MCP も同じパス認可を通す。 |
| `src/serve/http_server.*` | cpp-httplib で既存 API を提供。認証、ready gate、WebUI fallback と `/mcp` を整合させる。 |
| `src/serve/mcp_proxy.*` | WebUI 用 `/cors-proxy` の中継であり、今回の MCP server ではない。既存動作を維持する。 |
| `src/media/CMakeLists.txt` | `ninfer_local_video` は Linux 側でビルドされる。MCP 動画機能も同じプラットフォーム条件に合わせる。 |
| `tools/video-lab/`、`tests/cmake/ProductTests.cmake` | 既存 pipeline 回帰テスト、URI・serving テストの登録先。共通実装を直接検証する。 |
| `tools/suspend-dev/Dockerfile`、`prepare-build.sh` | BuildKit cache を開発イメージへコピーし、元の絶対パスと未変更ファイルの mtime を維持する手順がある。今回の開発環境に応用する。 |

現在の `Info` には average/nominal FPS の区別、audio stream count、観測済み interlace 状態などが不足する。既存 timestamp は生の PTS × time base で、先頭 frame を 0 秒に正規化していない。欠損 PTS と非増加 PTS は既存 index が拒否する。

## 3. ユーザーと確定した事項

### Q1. OpenCode の対象と作業範囲（回答済み）

ユーザー指定の `E:\koji\work\20260813\opencode\opencode.exe` を E2E client として使用する。OpenCode v1.18.34 に既存 patch を適用して単一 exe に build 済み。新しい OpenCode patch の作成を前提とせず、この exe との結合を対象とする。

参照済みファイル:

- `E:\koji\work\20260813\opencode\ninfer-video-opencode-project\ninfer-video-opencode-project\README.md`
- 同ディレクトリの `patches\opencode-v1.18.34-ninfer-video.patch`
- 同ディレクトリの `examples\opencode.jsonc`

README の目的は「同じ Qwen3.8-27B agent が動画を視覚入力として見る」こと。provider ID は例示どおり `opencode-ninfer`、MCP は remote 設定、NInfer と MCP は WSLC、OpenCode exe は Windows ホストで動かす。対応 protocol version は patch だけでは確定できないため、実装 Phase 0 でこの exe の negotiation を確認する。

patch は `resource_link` の URI が `ninfer-video://` で、MIME が `video/` で始まる場合のみ attachment 化し、さらに lowering では `ProviderShared.VIDEO_MIMES` に含まれるか判定する。このため MIME の省略は不可。受理される MIME の集合は Phase 0 で確認する。README は native runtime を provider に対応づけて説明しているが、提示 patch はフラグの既定値を true に変更しているため、実 exe と指定 provider で経路を検証する。

### Q2. 「動画開始からの秒数」の原点（回答済み）

確定: source frame 0 の presentation PTS を 0 秒とする。元の整数 PTS は seek 用に保持し、MCP 出力とモデルへ渡す timestamp は同じ原点に統一する。既存の非ゼロ開始 PTS のモデル入力表示もこの定義へ変更する。

公開時刻は `(frame_pts - first_frame_pts) * time_base`。内部 PTS が 100 秒から始まる動画の `time_seconds=10` は内部時刻 110 秒に対応する。途中範囲を選択しても、その範囲の先頭を 0 秒には戻さない。

最終 frame の timestamp より後は、最後の frame の表示期間内であっても仕様 §8.5 に従って error とする。duration と最後の PTS は別の値として扱う。

### Q3. inspect_video の path（回答済み）

確定: `inspect_video.path` は NInfer から見える動画ファイルの絶対パスのみを受け付ける。例: `/videos/demo.mp4`。URI と相対パスは拒否し、query によるオプション指定は受け付けない。

選択条件は各 tool 引数で指定する。省略時は `start=0 / end=EOF / skip=0 / scale=1 / deinterlace=auto / bbox=全体`。URI との引数結合・優先順位処理は不要。認可・検証後に結果として canonical `ninfer-video://` URI を生成する。入力 path を query として分解しない。ファイル名そのものの予約文字は URI 出力時に escape する。

ユーザーが更新した schema に合わせ、metadata/resolve を含む 3 tools の path は local file path のみとし、URI 入力をサポートしない。metadata の任意 URI 受付という旧仕様も削除する。

## 4. 実装側で採用する設計

### 4.1 共有 service と所有権

- `src/media/local_video/video_source_service.*` に `VideoSourceService` を置く。JSON、HTTP、Qwen geometry への依存を持たない。
- 既存 `LocalVideoSourceCache` の source identity/LRU をここへ移し、モデル側の重複 cache を削除する。プロセス内の標準 service accessor は一つとし、MCP と frontend の標準経路が同じ instance を参照する。テストには独立 instance を注入可能にする。
- 公開内部操作は `open`、`metadata`、`resolve_time`、`create_reader` 相当。source handle は共有所有権を持ち、reader の FFmpeg/filter 状態は request-local のままにする。
- root 認可は product/serving 側で毎回実施し、cache hit で省略しない。service 自体は認可済み local regular file を扱う。
- canonical path + size + mtime を identity とする。構築の前後に確認し、途中変更された metadata/index は公開しない。変更後は新 entry、実行中の旧 reader は変更エラーとして終了する。
- LRU は既存に合わせて既定 8 sources。読み取り中の handle を破壊しない。eviction 後に使用中 entry が残っている場合も、同じ identity に再合流できる管理にして二重構築を防ぐ。
- global lock 中に probe/decode を実行しない。source ごとに probe/index の構築状態を管理し、同一 source は合流、別 source は並行可能にする。失敗時には全 waiter を起こし、途中 index を成功扱いしない。
- 既存 scan/pixel 上限、deadline/checkpoint を引き継ぐ。LRU 上限は使用中 reader を含む絶対 RAM 上限ではないことを明記する。decoded frames の常駐 cache は追加しない。

### 4.2 Metadata と時刻解決

- `VideoSource::Info` を拡張し、time base、nominal/average rate、audio count、field order、duration の有効性を保持する。metadata 取得は原則 probe のみ。
- container の `nb_frames` を無条件に exact と扱わない。信頼できる exact 情報がなければ `frame_count=null / frame_count_exact=false`。完全 index があれば exact count に更新する。
- CFR/VFR の未確定は `null`。完全 index の PTS 間隔と rational rate/time-base 量子化を照合する。単純な nominal/average rate 比較や、丸めによる 1 tick の揺れだけで VFR と断定しない。判断根拠不足なら `null` を維持する。
- interlace は reported 情報と decoded frame 観測を区別する。混在を観測すれば `mixed`。metadata だけで不明なものは `unknown`、不明 field order は `null`。
- duration 不明時は内部で unknown を保持する。出力 schema は数値必須なので、その場合に限って共通 scan から終端と frame duration を求める。それでも正確な値を得られなければ、架空の 0 秒を返さず説明付き tool error とする。
- `resolve_time` 初回は既存の完全 index を構築し、以降は binary search。frame number は source presentation order のまま保持する。平均 FPS による代用はしない。
- time は有限・非負を要求する。最近傍比較は入力の小数を早期に整数 tick へ丸めず、整数 PTS と rational time base に基づいて行う。同距離は小さい frame number。radius は既定 2、0～16、返却端は有効範囲で切る。
- index が正しい終端を確定した後はその範囲を使う。container の推定 duration だけで、有効な frame を範囲外と誤判定しない。
- 欠損／重複／逆行 PTS は既存の拒否方針を維持し、`video_index_failed` 等で原因を説明する。PTS の再ソートで source frame 番号を変えない。

### 4.3 URI と inspect_video

- `local_video_url.*` に builder と共有 selection validation を追加する。MCP 内に独立 query builder/parser を作らない。
- `parse(build(spec))` の意味的 round trip、UTF-8、空白、`%`、`?`、`#`、`&` の escaping を検証する。
- Linux/WSLC が開く path はコンテナ内絶対パス。Windows drive path を Linux のファイル名として開いたり、暗黙に `/mnt/...` へ変換したりしない。Windows host path は起動時の bind mount で container path に対応させ、未対応の drive URI は明確に拒否する。native Windows decoder 対応は今回の範囲外。
- root・存在・可読性、source 寸法内 bbox、scale、skip、deinterlace、開始終了順、既知 exact count を検証する。bbox clamp は行わない。
- inspect のためだけに未知 frame count の全 scan は強制しない。未知の終端は後段 decode が検証する。既知の index は再利用する。
- 結果は instruction を含む text と canonical URI の `resource_link`。ユーザー承認済み（2026-10-06）: custom native resource の MIME は compatibility tag `video/mp4` に統一する。実ファイルの container/codec は FFmpeg が判定し、元ファイルの拡張子や bytes を変換しない。`inspect_video` には outputSchema がない現行 schema を尊重する。

### 4.4 MCP transport と schema

- MCP SDK/ライブラリ選定を先行 spike とする。公式 SDK 一覧には調査時点で C++ が掲載されていないため、利用可能な C++ 実装の source、license、protocol 対応、既存 cpp-httplib への handler 組み込み可否を確認する。名称だけで採用を確定しない。
- 採用条件は「既存 HTTP server/port 内」「別 listener/process 不要」「Streamable HTTP」「固定可能な依存」「schema/result を忠実に表現できる」。SDK 内部の HTTP 型・JSON 型との競合も確認する。
- 適合 library があれば薄い adapter で組み込む。適合するものがなければ、その根拠と最小の自前 adapter 案を提示し、仕様 §11.2 との整合を確定してから transport 実装へ進む。
- protocol version を OpenCode と合意後に固定する。仕様の initialize/session 記述と将来版の transport を混在させない。SDK commit/version、ライセンス、取得方法を記録する。
- 旧 lifecycle 型を採る場合は initialize、initialized notification、ping、tools/list、tools/call を実装。同期 JSON 応答を基本とし、server-initiated SSE 不要なら GET は規約に沿う 405。session の有無と DELETE は採用版・SDK に合わせ、必要のない独自 session state を追加しない。
- JSON-RPC envelope/method エラーと tool error を区別する。tool 引数・動画処理の失敗は採用 MCP 仕様に合わせた `isError` と判別可能な code/message を返す。SDK が validation を担う場合も同じ外部契約をテストする。
- `mcp/tool-schema.json` をビルド時に埋め込んで登録する。C++ 内の手書きコピーを正本にしない。公開 schema の一致と input/output validation を別々に検証する。
- metadata/resolve 成功時は `structuredContent` と互換用 JSON text を返し、outputSchema に適合させる。未知 property、必須不足、型、整数 overflow、非有限数、enum、nested bbox を検証する。
- `/mcp` を API path として扱い、既存 API key 認証を適用する。WebUI 静的配信の GET 認証例外や `/cors-proxy` の例外へ入れない。Origin、Content-Type/Accept、protocol header、HTTP method は採用版どおり処理する。
- 起動 readiness は既存 server に合わせる。model suspend 中でも CPU-only tool は GPU resume/inference を起動しない。長い index scan の同時実行数と deadline を bounded にし、終了・cancel 時に waiter と reader を解放する。

## 5. WSLC 開発環境

確認済み: サンドボックス内の `wslc images` / `wslc list --all` は `ERROR_FILE_NOT_FOUND` になるが、権限昇格した同じコマンドは成功した。`ninfer-all:build` は存在し、確認時 image ID は `d10090be01d6`、既存コンテナはなし。以後の wslc 操作は必要な昇格で実行する。WSLC 自体の故障として扱わない。

`../tmp/dockerfile-ninfer-build` は `/build` と `/ccache` を BuildKit cache mount に置き、image layer へコピーするのは `/out` の実行ファイルのみ。このため `FROM ninfer-all:build` だけでは Ninja tree/object を継承できない。既存 suspend-dev と同じ cache 取り込みを追加する。

実装開始時に `tools/video-mcp-dev/{Dockerfile,prepare-build.sh,build-test.sh,README.md}` を追加する。

1. `FROM ninfer-all:build`。同じ builder/session の cache ID `ninfer-build` を `/baseline-cache` に locked mount する。
2. `configuration` の hash で baseline tree を特定し、`src`、`configuration`、該当 hash directory を元と同じ `/build/...` へコピーする。build-dir marker を作る。cache 不在時は停止し、自動で全 CUDA rebuild に進まない。
3. source/compiler/CUDA architecture/options と baseline cache の一致を確認する。必要な FFmpeg fixture CLI、Python 3.11、選定した MCP テスト client のみ追加する。Ubuntu 24.04 の既定 Python を 3.11 とみなさない。
4. ホスト checkout は `/workspace:ro`、動画は `/videos:ro` へ mount。ソースは container 内 `/build/src` に checksum 同期し、未変更ファイルの mtime を維持する。sync の削除対象はこの container 内 source に限定する。
5. 元 build directory を保持して `BUILD_TESTING=ON` と必要設定だけ再 configure。CUDA toolchain/architecture を意図せず変更しない。`cmake --build <build-dir> -j --target ...` で対象だけ増分 build する。
6. ccache は永続 volume を使用可能にする。最初の build で Ninja の再ビルド理由を確認し、不要な全 CUDA 再コンパイルが発生する場合は先に原因を修正する。

予定コマンド（`ninfer-all` 内。Dockerfile/scripts 作成後に実行）:

```powershell
wslc build -f tools/video-mcp-dev/Dockerfile -t ninfer-all:dev .
wslc run -d --name ninfer-all-dev --gpus all `
  -v "${PWD}:/workspace:ro" `
  -v ninfer-all-dev-ccache:/ccache `
  ninfer-all:dev
wslc exec ninfer-all-dev sh /workspace/tools/video-mcp-dev/prepare-build.sh
wslc exec ninfer-all-dev sh /workspace/tools/video-mcp-dev/build-test.sh
```

CPU テストだけなら GPU を使用しない。実動画・モデルの mount と host localhost への port 公開は E2E 設定時に追加し、`--local-media-root` は mount 後の container path を指定する。GPU E2E 前に空き VRAM・既存利用を確認する。計画段階では image の一覧確認までで、dev image/container はまだ作成していない。

## 6. 作業順序と各フェーズの出口

| Phase | 作業 | 完了を判断する証拠 |
| --- | --- | --- |
| 0 | 確定事項を前提に WSLC dev 環境、MCP library/protocol spike | 増分ビルド再利用を確認。選定 library が同一 server に接続でき、対象 client と discovery が成立する見込みを具体的に示す。 |
| 1 | cache/service 共通化、既存 frontend の接続変更、metadata 拡張 | 従来の video-lab/動画準備回帰が通る。MCP 相当 caller と reader が同じ source を共有し、変更・eviction・並行呼び出しを処理できる。 |
| 2 | index 公開 API、resolve、時刻原点・metadata semantics | CFR/VFR/B-frame/非ゼロ開始 PTS の既知 fixture で frame 番号・時刻・端点が一致する。2 回目 resolve は再 scan しない。 |
| 3 | shared URI builder、絶対パスと個別引数の inspect validation | parser/builder round trip、URI 入力拒否、root policy、bbox/range、特殊文字を検証。inspect は inference を呼ばない。 |
| 4 | MCP handler、schema 埋め込み、3 tools、HTTP 登録 | 実 HTTP で initialize/discovery/call、schema 一致、エラー分類、認証、GET/DELETE を検証。既存 OpenAI/Anthropic/proxy 回帰が通る。 |
| 5 | OpenCode 結合、同一 agent 動画 E2E、docs 整備 | resource link が後続 `video_url` となり、NInfer が同じ source/index を使って動画を取り込み、同じ agent が fixture 内容を回答する。 |

各 Phase の完了時に本計画へ変更点、実行コマンド、結果、未完了事項、次の作業を追記する。stable な契約は仕様と既存 reference に反映し、計画だけに残さない。

## 7. 検証項目

| 領域 | 検証 |
| --- | --- |
| fixtures | 小さな CFR MP4、VFR MP4/MKV、B-frame、音声あり/なし、progressive/interlaced/mixed、非ゼロ開始 PTS を用意。生成条件と期待 PTS を記録。ffprobe はテスト oracle としてのみ利用し、production tool は subprocess を使わない。 |
| resolve | 0、fractional time、既知 midpoint tie、末尾ちょうど、末尾超過、duration 内でも最終 PTS 超過、radius 0/2/16、端での件数減少。VFR は average_fps 計算と異なる正解を必須にする。 |
| metadata | 全 outputSchema 項目、nullable、未確定→index 後 exact への更新、unknown duration、field order、time-base 丸め CFR を検証。 |
| cache | metadata → resolve(10) → resolve(20) → inspect → 実 reader を同一 identity で実行。metadata snapshot 構築 1 回、完全 index 構築最大 1 回、再利用カウンタを確認。同時構築、別 source 並行、LRU、size/mtime 変更、構築失敗・中止後の再試行も検証。 |
| FFmpeg I/O | cache reuse と request-local decoder の reopen を区別する。既存 `Input` は open ごとに stream discovery を行うため、probe 1 回という仕様を単なる命名で満たさない。cached stream 情報で省ける discovery を省き、実 `avformat_find_stream_info` 回数も計測する。demux/decoder 初期化に不可欠な open は別カウンタで説明する。 |
| inspect/URI | 絶対パス限定、URI/相対パス拒否、query オプションを解釈しないこと、個別引数と既定値、crop→scale、skip の意味、既知 frame 範囲、root 未設定/外部/symlink escape、空白・日本語・予約文字、Linux path、未対応 Windows drive の明確なエラーを検証。 |
| transport | 実 MCP client で discovery、全 tools、正しい structuredContent/resource_link。malformed JSON、unknown method/tool、通知、schema 不適合、unsupported version、認証、Origin、HTTP method、採用する場合の session lifecycle。 |
| coexistence | `/v1`、`/health`、WebUI、`/cors-proxy` を確認。scan 中の通常 HTTP 応答、model suspend 中の CPU-only tools、shutdown/cancellation を検証。 |
| E2E | 読める文字や色・frame 番号を含む動画を使う。OpenCode の同じ agent が tool 後の次の model call に `video_url` を付与し、別 LLM 呼び出しなしで視覚内容を回答する。 |

テストは `tests/cmake/ProductTests.cmake` 等へ登録し、新規 service/tool/HTTP テストの CPU-only 実行経路を用意する。既存動画回帰と関係する serving schema テストを対象指定で実行する。GPU 推論がないと検証できない E2E は分けて結果を記録する。

変更文書は `docs/serving.md`、`docs/maintainer/local-video-pipeline.md`、必要な README/CLI help、`tests/README.md`、開発環境 README。最終的に仕様 §19 の全項目を証拠へ対応付け、外部 patch や GPU が未検証なら v1 全体を完了扱いしない。

## 8. 現在の進捗と再開点

- [x] 仕様、schema、URI/decoder/cache、HTTP 統合箇所、既存 WSLC 開発手順を調査。
- [x] 昇格した wslc でベースイメージの存在とコンテナ一覧を確認。
- [x] 本計画を作成。
- [x] Q1 の回答と指定 OpenCode patch の統合条件を反映。
- [x] Q2～Q3 の回答を計画・仕様・schema へ反映。
- [x] fastmcpp pinned transport と同一 HTTP server 統合、CPU HTTP 検証。
- [x] dev image/container 作成と BuildKit 中間生成物の再利用確認。
- [x] Phase 1～4 の実装と CPU 検証。
- [x] Phase 5: 実 server / 指定 OpenCode の同一 agent E2E。

v1 完了。追加の実装・ユーザー判断待ちはない。再検証は tools/video-mcp-dev/README.md の手順を使用する。

### 実装チェックポイント: WSLC 開発環境

- `tools/video-mcp-dev/` に baseline cache 取り込み Dockerfile、checksum source 同期スクリプト、README を追加。
- `wslc build -f tools/video-mcp-dev/Dockerfile -t ninfer-all:dev .` と README の run コマンドで `ninfer-all-dev` を起動済み。元 Ninja tree は `/build/cdce3b3aef6da28fdc9d6a913a78b062d76df75a1ffe09d47dde128f92923bc3`、取り込み容量 6.3 GB。
- `prepare-build.sh` 成功。CUDA 13.1.115、sm_86 を維持。変更前の `ninja -n ninfer-serve` は build-id stamp、main.cpp、link の 3 steps のみで CUDA 再コンパイルなし。
- コンテナの Python は確認済み `/usr/bin/python3` 3.12.3。CMake registration と汎用 fixture スクリプト用に使用する。既存 maintainer Python 3.11 環境はこの Ubuntu image にない。Python product/converter の変更はない。
- OpenCode source の MCP SDK は 1.29.0、native lowering の MIME 集合は `video/mp4`, `video/webm`, `video/quicktime`。C++ 候補 `fastmcpp` の license/source を取得して評価中。

### 実装チェックポイント: 共通 source service と時刻 API

- `VideoSourceService` を media 共通層へ追加し、モデル frontend の source cache を移行。標準 instance をプロセス共有し、認可済み source の LRU・size/mtime invalidation・同一 source の probe 合流を実装。eviction 後も active source に再合流する。
- `VideoSource::metadata/resolve_time` を追加。audio/rate/container metadata、完全 index 後の exact count、CFR/VFR・interlace 観測、整数 PTS 最近傍探索、前 frame tie、最終 PTS 範囲検証を実装。公開 timestamp と既存 reader/plan を frame 0 基準へ統一。
- 元 source の codec parameters/time base を保持し、index scan と reader の reopen 時に `avformat_find_stream_info` を繰り返さない。decoder は request-local で、`decoder_opens` と `metadata_probes` は別に計測する。
- WSLC CUDA 13.1/sm_86 で `ninfer_video_source_service_test` と `ninfer_qwen3_5_local_video_payload_test` を増分ビルドし、CTest 2/2 成功（0.80 秒）。新規 fixture は CFR/B-frame/内部開始 100 秒/audio、VFR、interlaced。同時 index、LRU active reuse、mtime invalidation、cancel/retry、reader の時刻原点も検証。CPU-only、CUDA kernel の再コンパイルなし。
- GPU 推論、MCP HTTP、OpenCode E2E は後続フェーズで検証。次: shared URI builder と MCP SDK の組み込み。

計画確定時の検証: schema の JSON parse、3 tools の存在、文書内の相対リンク、変更ファイルの whitespace check を確認済み。今回の変更は計画・仕様・schema・作業規則のみ。実装 build、動画処理テスト、OpenCode E2E は実装未着手のため未実行。

参照した外部一次資料（2026-10-06 確認）:

- [公式 SDK 一覧](https://modelcontextprotocol.io/docs/2026-07-28/sdk)
- [MCP 2025-11-25 Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — initialize/session 型の互換候補。採用版は OpenCode と照合して確定する。

### 実装チェックポイント: 共通 URI builder

- `build_local_video_url` を既存 parser と同じ product module に追加。UTF-8・空白・予約文字を percent escape し、既定値を省いた一定順序の query を生成する。生成後に同じ parser で制約を検証する。
- WSLC 内で既存 `test_local_video_url.cpp` を CPU-only で直接 build/run。既存 path/root/symlink 検証と、日本語・予約文字・scale 精度・全引数の round trip が成功。
- MCP HTTP/実モデルはまだ未検証。次: fastmcpp transport の同一 server 登録、schema 埋め込みと3 tools の実装。

### 実装中チェックポイント: MCP HTTP / 判断待ち

- fastmcpp 3.4.7.1 の pinned transport subset を追加し、既存 httplib server への route 登録 adapter を実装。MCP 2025-11-25 / 2025-06-18、POST JSON、GET 405、DELETE session 終了を実装。JSON-RPC envelope、HTTP media headers、session/version を検証。
- schema 正本を CMake configure で埋め込み、3 tools と schema validation、root 認可、geometry/range validation、4 並行 call 上限、120 秒 checkpoint、cancel/shutdown を実装中。
- WSLC で `ninfer-serve` と CPU HTTP fixture を増分 build 成功。CTest `video_mcp_http|video_source_service|local_video_payload` 3/3 成功（1.01 秒）。HTTP discovery は正本 schema と完全一致し、malformed JSON、protocol error、tool error、Host/Origin、DELETE、特殊文字 URI を検証した。実 server の認証・suspend 中 tools・OpenCode E2E は未検証。
- この単位は未 commit。草案の inspect MIME は custom native resource の compatibility tag として `video/mp4` を付けるが、計画 §4.3 は実 container MIME を要求していたため採用判断待ち。指定 OpenCode の lowering が video/mp4・video/webm・video/quicktime のみで MKV を拒否することを確認済み。無断で方針変更していない。
- 実モデル `.ninfer` は workspace 内に見つからず、WSLC volume は ccache のみ。GPU は nvidia-smi で 0 MiB / 24 GiB、process なし。E2E に使用する明示的なモデル artifact path を回答待ち。
- 次: MIME 方針の回答を SPEC/PLAN に反映し、transport lifecycle と cancellation の検証を補強、既存 serving 回帰、実 server / 指定 OpenCode の E2E を実行して本単位を commit。開発 container と build directory は上記チェックポイントのまま。

### 実装チェックポイント: MCP transport / 3 tools 完了

- 前節の判断待ちは解消。互換用 MIME `video/mp4` をユーザー承認どおり採用し、SPEC と計画 §4.3 を更新。指定モデルは `C:\AI\ninfer-rtx3090-windows-x64-0.11.0-rtx3090\models\huihui-Qwen3.8-27B-abliterated-NInfer-v3\Huihui-Qwen3.8-27B-abliterated-ninfer-v3.ninfer`。native volume `ninfer-video-mcp-models` へコピーし SHA256SUMS 2/2 一致。
- Linux `/mcp` を既存 HttpServer に登録。API path/CORS/Host/Origin/既存 auth と統合。fastmcpp pinned subset の protocol/session transport を使用し、正本 schema を埋め込んだ strict tool validation と3 tools を実装。全 tools が共通 source service を使用し、inspect は inference を呼ばない。
- initialized 通知、session DELETE、protocol/media/envelope error を検証。4-call 上限、120秒 deadline、cancel/shutdown を source probe・per-source lock 待ち・index scan へ伝播。debug log は canonical source、metadata hits/probes/evictions、index builds/reuses/count/time、decoder opens、resolve、inspect URI を記録。
- `ninfer-serve`、CPU fixture、既存 `ninfer_tests` bundle を WSLC で build。元モデル CUDA kernels は再コンパイルなし。関連 CTest 10件が成功: source service、URI、OpenAI/Anthropic schema、Responses store、MCP proxy、HTTP error/transport、MCP HTTP、local-video payload。MCP fixture は cancellation と再試行、media-root 無効 discovery、2025-06-18/11-25、MP4/MKV link、malformed/unsupported requests を検証。
- source fixture を coarse 30000/1001 CFR MKV と progressive/interlaced mixed MKV へ拡張。concat 元の container duration により境界 PTS が重なった fixture を duration 指定で修正し、service test 再実行成功。欠損・重複 PTS を許容する product 変更はしていない。
- 未検証: 実 server 認証・model suspend 中 tools、指定 exe discovery/同一 agent native-video E2E。次は Phase 5。E2E container `ninfer-video-mcp-acceptance` (image ninfer-all:dev, model volume read-only, port localhost:18081) を準備済み。モデルはまだ起動していない。dev binary の一時配信は開発 container の `/build/.../apps`、port 18080。

### 実装チェックポイント: 実 server 検証済み / OpenCode 結合で停止

- `e4a8c2ac` に MCP transport / 3 tools の実装を commit 済み。Phase 5 の独立検証と再実行手順を `tools/video-mcp-dev/` に追加。結果は `ACCEPTANCE.md` にまとめた。
- 指定 model + WSLC + RTX3090 で実 NInfer listener の auth、metadata→resolve→resolve→inspect→native video inference が成功。red と回答し、debug counters は metadata probes=1 / index builds=1 / index reuses=3 / decoder opens=3。実 reader も同じ source/index を再利用。
- `--enable-model-suspend --suspend-snapshot-memory pageable` を使い、model suspended のまま3 toolsを実行成功。明示 resume と既存 Responses endpoint も成功。CPU HTTP test に scan 中 health と10/20秒の resolve を追加し、再検証済み。
- 指定 OpenCode exe (`0.0.0--202610051423`, SHA256 `4BDF1B83DC9030030D3575DCA643160D7787F72C6A4892284F809E798C60D90D`) の discovery と tool call は成功。しかし受け取った resource_link が後続 model request の video_url にならない。default は ai-sdk、明示 native flag=true でも video_url 0件。native run は blue と回答したが、wire と server media 記録がないため視覚 E2E 成功とは扱わない。
- `check-opencode.py` は実 wire / events の1 session・1 resource linkを照合し `complete=false, native_video_urls_matched=[]` で正しく失敗。外部 exe を改変せず、ユーザーに patch 適用状態の確認・修正・再ビルドをこちらで進めるか、差替え exe を指定するか判断待ち。
- SPEC §19 は OpenCode E2E 以外を確認済みに更新。v1 全体は未完了。次はこの client 問題を解決して同じ neutral-name fixture を再実行し、後続 request の native video_url と視覚回答を両方確認する。
- 生の wire/events/runtime log は host `.cache/video-mcp`、実 server reports/logs は停止済み `ninfer-video-mcp-acceptance` の `/acceptance` に残る。observer と acceptance GPU server は停止し、開発 container / model volume は保持。再開は README / start-acceptance.sh に従う。

### 実装チェックポイント: 再ビルド OpenCode の E2E 成功 / v1 完了

- ユーザーが指定 patch を実際に `git apply` して exe を再ビルドしたため、同じ指定 path で再検証。新 version `0.0.0--202610051714`、SHA256 `52C335CDB867B11523CD144083EEDE0268AA8E2ED4C59E4554E3328C5A8C026F`。
- `acceptance.py` を再実行して `complete=true, native_video_answer=red, auth_and_suspend=passed`。model / WSLC / listener / suspend 中3 tools の検証は引き続き成功。
- native opt-in flag を追加せず通常の `run-opencode.ps1` で runtime=native。neutral-name `/videos/mystery.mp4` を inspect し、resource_link→file attachment→同じ agent の後続 `video_url` を wire/events で確認。NInfer の後続 request は media=1。
- frames 2–3 は blue、frames 0–1 は red と正しく回答。両 run の `check-opencode.py` が1 agent session・1 inspect link・matching native video URIありで `complete=true`。後者の inspect 時点で同一 mystery source は index_builds=1 / index_reuses=1 / metadata_probes は増加せず、前 run reader の source/index を再利用。
- SPEC §19 の全完了条件を満たした。残る実装・判断待ちはない。変更は検証記録と完了 status の更新のみで、以前通過した CPU tests の再実行は不要。
- 成功 reports は host `.cache/video-mcp/{http-report,opencode-blue-report,opencode-red-report}.json`、blue events/runtime log は suffix blue を付けて保存、最新 events/log は red run。wire summary と container `/acceptance` の証拠も保持。検証用 observer / GPU server を停止し、model volume / development container は保持。

### 追加実装チェックポイント: 単一フレーム画像モード（実装・検証中）

- 追加要求: `ninfer-video:///videos/demo.mp4?frame=N` と `inspect_video(frame=N)`。0始まりの1枚をQwen3.8へ画像として入力し、start/end/skipとの同時指定を既定値でも拒否する。その他の既存空間加工オプションを許可する。
- URI parser/builder、公開OwnedLocalVideo、serving acquisition、Qwen frontend/processor、lazy payload計画、MCP schema/validationを更新。画像用tokens/modality/位置情報を使用し、同じRGBを2つのtemporal patch slotに複製。従来の1枚rangeは動画入力のまま。
- SPEC_VIDEO.md §5.0、SPEC_VIDEO_MCP.md §9.2/9.3、serving guide、PowerShell helperに反映。MCPは互換MIME video/mp4を保持し、既存OpenCodeにURIをそのまま渡す方針。
- WSLCの既存ninfer-all:dev / ninfer-all-dev / baseline build treeで増分build中。公開入力型への変更が参照先C++/CUDAの再コンパイルを発生させる。GPUは開始時0 MiB。CPU/frontend/API/MCPテストと指定exeによる単一フレームE2Eを追加中。
- 実施済み: PowerShell helper DryRunでframe=0&scale=2生成とFrame+StartFrame=0拒否を確認。
- 次: build完了後に関連CTest、実モデルのimage/range回帰、OpenCode同一agentでframe URI forwardingを確認。結果を本checkpointへ追記してcoherent unitをcommit。現段階でユーザー判断待ちはない。

### 再開用メモ: ユーザー指定でビルド監視を停止

- ユーザーが「buildはかなりかかるので、やることがなくなったら止め、半日後に再度呼ぶ」と指示。ビルド自体は中断せず、このチャットの作業・監視を停止する。未検証のため本追加単位はまだcommitしない。
- 進行中のbuild command: `wslc exec ninfer-all-dev sh -c 'cmake --build "$(cat /build/video-mcp-build-dir)" --target ninfer-serve ninfer_tests ninfer_video_mcp_http_fixture -j'`。このチャットのexec session_id=69688。再開時にsession出力が取得できなければ、同じcommandを実行する前にcontainer内のninja/nvccプロセスが終了していることを確認する。
- buildは公開types.h変更による250ステップ。最後に確認した出力は19/250、CUDA cicc/ptxasが実行中。WSLC RAM23GiB、swap未使用、GPU0MiB。baseline build treeは `/build/cdce3b3aef6da28fdc9d6a913a78b062d76df75a1ffe09d47dde128f92923bc3`。
- 初回source sync/build開始後に `tests/models/qwen3_5/test_frontend.cpp` の画像modality/token/位置情報一致テストを追加した。現在のbuild終了後、prepare-build.shで最新sourceを再同期し、ninfer-serve / ninfer_tests / ninfer_video_mcp_http_fixture / ninfer_qwen3_5_local_video_payload_testを増分buildすること。fixtureは64x64 PPMをlocal sourceとして使用し、通常画像とtoken_ids/positions/token_types/rope_deltaが一致することを検証する。
- 独立CPU検証済み: 最新host sourceからg++で `/tmp/video-frame-url-test` をbuild/runし成功。frame=0/100、spatial option roundtrip、負数/小数/overflow、range3引数の明示既定値併用と引数順序入替の拒否を確認。PowerShell helper DryRun/排他拒否も成功。
- OpenCode sourceをread-onlyで確認: session/tools.tsのresource_link処理はninfer-video scheme + video MIMEをそのままfile.urlへ、packages/llm/src/protocols/openai-chat.tsはその文字列をvideo_url.urlへ渡す。frame queryの制限なし。exe修正不要の設計だが実exe検証は未実施。
- 次の検証: 関連CTest（frontend、local_video_payload、local_video_url、video_mcp_http、video_source_serviceと既存OpenAI/Anthropic/HTTP回帰）。その後README手順で既存acceptance/observer containerを起動し、最新ninfer-serveをtransferしてacceptance.pyを実行。追加したframe0=red/2=blue/3=blue（crop+scale）、併用/負数/範囲外API400、suspend中frame toolを確認する。
- 指定exeのOpenCode E2E: `.cache/video-mcp/opencode-observer.json` を使いrun-opencode.ps1へinspect_video(frame=2)を要求するpromptを渡す。wire/eventsをcheck-opencode.pyで照合し、同一agentの後続video_urlにframe=2 URIとblue回答を確認。完了後GPU server/observer停止、結果をACCEPTANCE.mdと本PLANに記録しdiff/checksを確認してcommit。
- 実model artifact・volume・container・portは前のv1完了checkpointのまま。GPU acceptanceとobserverは現時点で停止中。ユーザー判断待ちや既知の実装問題はないが、新機能の関連build/CTest/実model/実exe検証は未完了。

### 停止前の並行検証結果

- 主buildに触れず `/tmp/video-frame-cpu` へ最新host sourceのMCP handler・URI builder・HTTP fixtureをCPU-onlyで別途compile/link。既存の変更していないtransport/media/spdlog librariesを利用し、最新embedded schemaで実HTTP regressionを実行成功（video MCP HTTP contracts passed）。frame=0/7 + bbox/scale/deinterlace、明示既定値range引数の併用拒否、負数/小数/bool/overflow/範囲外の拒否、既存HTTP/session/cancellation回帰を確認。
- frontend画像一致テストに使う64x64 PPMを既存production VideoSourceでprobe/planし、source frame0のみ・64x64 geometryが成立することを確認。fixtureの成立は確認済み、frontendのtokens/positions一致そのものは主build後のCTestで確認する。
- Python3.12.3（既存WSLC fixture interpreterの確認済み例外）でrun_video_mcp_http_test.pyとacceptance.pyのsyntaxを確認。git diff --checkも成功。独立検証helperはignored `.cache/video-mcp/frame-cpu-check.py`、成果物はcontainer `/tmp/video-frame-cpu` にあり、repository commit対象には含めない。
- 主buildは中断していない。初回build後の最新source再同期・関連target build/CTest・実model・OpenCode検証・結果記録・commitは依然として次回作業。ここでユーザー指定に従い停止する。

### 追加実装チェックポイント: 単一フレーム画像モード完了

- ユーザーの再開指示に従い作業を再開。継続buildは250/250成功。最新sourceを再同期し、ninfer-serve / ninfer_tests / ninfer_video_mcp_http_fixture / ninfer_qwen3_5_local_video_payload_testの増分buildも成功（追加C++ testのみ）。WSLC CUDA13.1 / sm86 / ninfer-all:dev /既存baseline treeを使用。
- 関連CTest 11/11成功（22.77秒）: source service、URI、OpenAI/Anthropic schema、Responses store、MCP proxy、HTTP error/transport、MCP HTTP、Qwen frontend、local-video payload。画像modeのtoken_ids/token_types/positions/rope_deltaが通常imageと完全一致、temporal=1、timestamp無しを確認。従来の1枚videoはvideo modalityのまま。payloadはsource frame3のみを選択し同じRGBのtemporal複製を確認。
- 指定実model / RTX3090でacceptance.py成功。frame0=red、frame2/3=blue（bbox+scale併用）。併用start/end/skipの明示既定値、負数、存在しないframeがHTTP400。既存range入力、auth、model suspended中3 toolsおよびframe mode、resume、Responses回帰も成功。
- 指定OpenCode exeを変更せず通常native実行。inspect_video(frame=2)→resource_link→file attachment→同一agentの次のvideo_urlに完全一致するframe URIを確認しblue回答。check-opencode.py complete=true、session=ses_ef1c99f19ffeJgC2qPbfwk0kwU、call=call_bdcc37db172d5b5b。OpenCode修正・再build不要を実証。
- SPEC_VIDEO.md §5.0と型、SPEC_VIDEO_MCP.md §9.2/9.3、schema/tool説明、serving guide、PowerShell helper -Frame、acceptance再実行手順・記録を更新。image modeでも互換MIME video/mp4を保持し、NInferでimage tokens/位置情報へ変換する。
- 結果はtools/video-mcp-dev/ACCEPTANCE.md、raw evidenceはhost .cache/video-mcpのhttp-frame-report / opencode-frame-report / opencode-frame-events / opencode-frame-run / frame-wireとacceptance container /acceptanceに保持。検証用GPU server/observer/一時build HTTP配信を停止。GPU0MiBを確認。
- 完全diff reviewとgit diff --checkを実施。build id unknownはrsync checkoutにgit metadataがない既知のbuild表示制限のみ。要求範囲の未検証事項・既知の不具合・判断待ちはない。単一のcoherent feature unitとしてcommitする。次の実装作業なし。
