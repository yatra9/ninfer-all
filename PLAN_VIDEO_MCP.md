# NInfer Video MCP 実装計画

更新日: 2026-10-06。状態: 計画作成済み、ユーザー確認事項は回答・反映済み。MCP 本体の実装は未着手。

## 1. 成果物と前提

正本は [SPEC_VIDEO_MCP.md](SPEC_VIDEO_MCP.md) と [mcp/tool-schema.json](mcp/tool-schema.json)。本計画は作業順序、既存コードとの差分、検証方法、未確定事項を管理する。

- 既存 `ninfer-serve` の同一プロセス・同一ポートに `/mcp` を追加する。
- `get_video_metadata`、`resolve_video_time`、`inspect_video` を schema どおり公開する。
- MCP と既存動画入力は同じ source metadata/index を共有する。
- `inspect_video` は resource link を返す。別 LLM 呼び出し、GPU 推論、全フレームの RGB キャッシュを行わない。
- 開発・ビルド・テストは Linux/WSLC。ユーザー指定の `FROM ninfer-all:build` を使用し、開発イメージを `ninfer-all:dev`、コンテナ名を `ninfer-all-dev` とする。ホストの Windows ビルドツリーには手を加えない。
- この依頼の成果物は本計画。実装開始時に各フェーズを進める。ユーザー指定により、実装中は AGENTS.md の `ninfer-video-mcp work checkpoints` に従い、検証済みの独立した作業単位ごとにコミットし、その前に本計画・仕様・関連文書を更新する。

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
- 結果は instruction を含む text と canonical URI の `resource_link`。指定 OpenCode patch が受理する正しい `video/*` MIME を必ず付ける。MKV 等も一律 `video/mp4` にせず、実 container と client の受理集合を照合する。対応 MIME がない場合は黙って text 化させず、互換性の未解決事項として扱う。`inspect_video` には outputSchema がない現行 schema を尊重する。

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
- [ ] MCP library/protocol の選定と同一 HTTP server 統合 spike。
- [ ] dev image/container 作成と BuildKit 中間生成物の再利用確認。
- [ ] Phase 1～5 の実装と検証。

次の作業: 実装依頼後は Phase 0 から開始する。現時点で追加のユーザー回答待ちはない。MCP library/protocol の技術選定と実機互換性検証は Phase 0 の作業として残る。

計画確定時の検証: schema の JSON parse、3 tools の存在、文書内の相対リンク、変更ファイルの whitespace check を確認済み。今回の変更は計画・仕様・schema・作業規則のみ。実装 build、動画処理テスト、OpenCode E2E は実装未着手のため未実行。

参照した外部一次資料（2026-10-06 確認）:

- [公式 SDK 一覧](https://modelcontextprotocol.io/docs/2026-07-28/sdk)
- [MCP 2025-11-25 Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — initialize/session 型の互換候補。採用版は OpenCode と照合して確定する。
