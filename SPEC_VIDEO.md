# NInfer カスタム・ローカル動画パイプライン仕様

## 1. 目的

本仕様は、`ninfer-serve`
に対して、サーバーローカルの動画ファイルを参照し、明示的なフレーム選択・前処理パラメータを指定できる特殊形式の
`video_url` を追加する拡張を定義する。

目的は以下のとおり。

-   既存の `video_url` 処理との完全な後方互換性を維持する。
-   専用のカスタム動画 scheme
    に一致した場合のみ、新しいパイプラインを使用する。
-   圧縮動画ファイル全体をホストメモリへ読み込まない。
-   サンプリング対象となった全 RGB フレームをメモリ上に保持しない。
-   指定されたフレーム範囲のみを処理する。
-   `frame=N` により1枚だけ選択し、Qwen3.8 Vision Encoder へ画像として入力する。
-   `start_frame`、`end_frame`、`skip_frame`
    によって決定論的にフレームを選択する。
-   crop、scale、デインタレースをサポートする。自動色調補正は将来拡張として予約する。
-   処理済みフレームを既存の Qwen Vision Encoder 経路へ渡す。
-   可能な限り既存の serving/media 実装への変更を局所化する。

本拡張は主として、`ninfer-serve`
と対象動画ファイルが同一マシン上に存在する WSL
ローカル環境での利用を想定する。

------------------------------------------------------------------------

## 2. 非目標

本仕様では以下を要求しない。

-   既存の Qwen Vision Encoder / Vision Tower の置き換え。
-   通常の HTTP(S)、data URI、その他既存の動画入力の意味変更。
-   `ffmpeg` を別プロセスとして起動する方式の導入。
-   任意のリモートファイルシステムへのアクセス。
-   カスタムローカル動画 scheme を OpenAI 本家 API と互換にすること。
-   token budget や pixel budget から `scale` を自動決定すること。
-   `start_frame`、`end_frame`、`skip_frame` の自動決定。

必要な pixel 数や token budget に基づく `scale`
の計算は呼び出し側の責務とする。

------------------------------------------------------------------------

## 3. 用語

### 既存動画経路

通常の `video_url` に対して現在 NInfer が使用している経路。既存の media
acquisition および decode 処理を含む。

### カスタムローカル動画経路

本仕様で追加する専用 scheme の `video_url`
の場合のみ使用される新しい経路。

### 選択フレーム

指定されたフレーム範囲および skip 条件を満たし、画像前処理 / Vision
Encoding へ進む decoded source frame。

### Vision 前処理

既存 Qwen Vision Encoder
の直前で行う画像・動画処理。resize、normalize、temporal
grouping、patchify 等、モデルが必要とする処理を含む。

------------------------------------------------------------------------

## 4. API 互換性とルーティング

### 4.1 既存 API 形式

OpenAI 互換 Chat API のリクエスト形式は変更しない。

例:

``` json
{
  "model": "qwen3.8-27b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": {
            "url": "ninfer-video:///mnt/d/videos/example.mp4?start_frame=100&end_frame=1000&skip_frame=2&bbox=320,180,640,360&scale=1.5&autotone=0&deinterlace=auto"
          }
        },
        {
          "type": "text",
          "text": "Describe what happens in this video segment."
        }
      ]
    }
  ]
}
```

現在の NInfer API が `video_url` を `{ "url": ... }`
オブジェクトではなく直接文字列として受け付けている場合は、その既存位置で同じカスタム
URL 構文を受け付けること。

本拡張のために新しいトップレベル request schema を導入してはならない。

### 4.2 カスタム scheme

以下の scheme の URL の場合のみカスタムパイプラインを有効化する。

``` text
ninfer-video://
```

例:

``` text
ninfer-video:///mnt/d/videos/example.mp4?start_frame=100&end_frame=1000&skip_frame=2
```

### 4.3 fallback 規則

通常の media acquisition より前にルーティングを行う。

疑似コード:

``` cpp
if (is_ninfer_video_url(url)) {
    return process_local_video(parse_local_video_spec(url));
}

return process_existing_video_url(url);
```

`ninfer-video` scheme ではない URL は、すべて既存挙動を維持すること。

カスタム URL を既存の「media 全体を owned bytes
に読み込む経路」へ渡してはならない。

------------------------------------------------------------------------

## 5. URL パラメータ

すべて `ninfer-video` URL の query parameter として指定する。

例:

``` text
ninfer-video:///mnt/d/video.mp4
?start_frame=100
&end_frame=1000
&skip_frame=2
&bbox=320,180,640,360
&scale=1.5
&autotone=0
&deinterlace=auto
```

### 5.0 `frame` — 単一フレーム画像モード

```text
ninfer-video:///mnt/d/videos/example.mp4?frame=100
ninfer-video:///mnt/d/videos/example.mp4?frame=100&bbox=320,180,640,360&scale=1.5&deinterlace=auto
```

`frame` は表示順に数えた0始まりの source frame index。非負の64-bit整数のみを受け付ける。
`start_frame`、`end_frame`、`skip_frame` との同時指定は、明示した値が既定値でも拒否する。
その他の既存オプションとの同時指定は可能（`autotone` は引き続き `0` / `false` のみ）。
存在しないフレームはエラーとし、近いフレームへの置換はしない。

OpenAI互換APIの外側の content type は引き続き `video_url`。
サーバーがこのモードを解析し、選択された1フレームを **画像の modality** として渡す。
画像用 `<|vision_start|><|image_pad|>…<|vision_end|>`、画像用位置情報、temporal grid=1を使用し、
動画用 timestamp テキストや `<|video_pad|>` は生成しない。
Qwenの画像patch形式に従い、同じRGB画像を2つのtemporal patch slotへ複製する。
`start_frame=N&end_frame=N` は従来どおり動画モードであり、この画像モードと同一ではない。

既存のroot認可、source/index cache、seek、decode、deinterlace→crop→scale→32-pixel alignment、
chunk payload account、Vision容量・`--local-video-max-tokens` 上限をそのまま使用する。
画像モードでも指定geometryを勝手に縮小しない。source PTSは診断情報に保持するがpromptへ挿入しない。
以降のtimestamp付き動画promptの説明・疑似コードは `frame` を省略した範囲モードに適用する。

### 5.1 `start_frame`

型:

``` text
int64
```

デフォルト:

``` text
0
```

制約:

``` text
start_frame >= 0
```

意味:

選択対象となり得る最初の source frame index。

frame index は 0 始まりとする。

### 5.2 `end_frame`

型:

``` text
int64 または省略
```

デフォルト:

``` text
未指定
```

制約:

``` text
end_frame >= start_frame
```

意味:

選択対象となり得る最後の source frame index。inclusive とする。

省略時はストリーム終端まで処理する。

### 5.3 `skip_frame`

型:

``` text
int64
```

デフォルト:

``` text
0
```

制約:

``` text
skip_frame >= 0
```

意味:

選択フレーム間でスキップする source frame 数。

選択条件:

``` cpp
selected =
    frame_index >= start_frame &&
    (!end_frame || frame_index <= *end_frame) &&
    ((frame_index - start_frame) % (skip_frame + 1) == 0);
```

例:

``` text
skip_frame=0 -> 全フレーム
skip_frame=1 -> 2フレームごとに1枚
skip_frame=2 -> 3フレームごとに1枚
skip_frame=29 -> 30フレームごとに1枚
```


### 5.4 FPS / timestamp の扱い

`fps` は URL パラメータとして指定しない。

source video の FPS および frame timing は FFmpeg の stream metadata / time base / decoded frame timestamp から取得する。

#### 基本方針

Qwen に渡す時間情報の正は、**sampling 後の実効 FPS ではなく、元動画上の source frame index と source FPS** とする。

CFR（固定フレームレート）動画では、各 selected frame の元動画上の index を保持する。

例:

```text
source_fps = 30
start_frame = 100
skip_frame = 2

selected source frame indices:
[100, 103, 106, 109, ...]
```

各 frame の元動画上の timestamp は:

```text
timestamp = source_frame_index / source_fps
```

で求める。

したがって上記の例では:

```text
100 / 30 = 3.333...
103 / 30 = 3.433...
106 / 30 = 3.533...
109 / 30 = 3.633...
```

となる。

`start_frame != 0` の場合でも timestamp を 0 秒へリセットしない。

また、selected frame が将来的に非等間隔になった場合も、各 source frame index をそのまま保持することで正しい時間情報を表現できる。

#### `effective_fps`

`skip_frame` に応じた実効 FPS は、必要に応じて参考値として自動計算してよい。

```text
effective_fps = source_fps / (skip_frame + 1)
```

例:

```text
source_fps=30, skip_frame=0    -> effective_fps=30
source_fps=30, skip_frame=1    -> effective_fps=15
source_fps=30, skip_frame=2    -> effective_fps=10
source_fps=29.97, skip_frame=2 -> effective_fps=9.99
```

ただし `effective_fps` は **Qwen に渡す時間情報の基準として使用しない**。

用途は以下に限定する。

- logging
- diagnostics
- UI / statistics
- sampling density の表示
- downstream component が参考値として要求する場合

sampling 後の frame を `0,1,2,...` と再 index し、`effective_fps` のみで元動画上の timestamp を復元する方式は使用しない。

#### 推奨内部 metadata

```cpp
struct SelectedVideoFrameTiming {
    std::int64_t source_frame_index;

    // CFR でも保持可能ならこちらを優先してよい。
    double source_timestamp_seconds;
};

struct VideoTiming {
    double source_fps;

    // 参考値。Qwen timestamp の正としては使用しない。
    double effective_fps;

    std::vector<SelectedVideoFrameTiming> selected_frames;
};
```

streaming 実装では全 selected frame の timing を vector に保持する必要はなく、各 temporal chunk と一緒に timing metadata を逐次渡してよい。

#### Qwen への時間情報の受け渡し

Qwen3-VL 系では video frame の timestamp 情報を prompt / temporal metadata の生成に使用する。

CFR の場合、原則として以下を Qwen handoff まで保持する。

```text
source_fps
+
original source frame indices
```

これにより Qwen 側で:

```text
timestamp = frame_index / source_fps
```

を計算できる。

元 frame index が 0 から始まる必要はない。

selected frame index が等間隔である必要もない。

したがって、以下のような入力も正しく扱える設計とする。

```text
source_fps = 30
selected frame indices = [300, 303, 306, 450, 900]
```

これは元動画上では:

```text
10.0 sec
10.1 sec
10.2 sec
15.0 sec
30.0 sec
```

を表す。

#### VFR（可変フレームレート）動画

VFR の場合、`frame_index / fps`、平均 FPS、`effective_fps` のいずれも正確な frame timestamp の代用として使用してはならない。

**VFR では、各 selected frame の PTS と stream time base から元動画上の正確な timestamp を算出し、その timestamp を Qwen に渡すことを必須要件とする。**

基本式:

```text
timestamp_seconds = frame_pts * time_base
```

FFmpeg の timestamp rescale API を使用する場合も、意味的に同じ元動画上の presentation timestamp を秒単位または Qwen handoff が要求する時間表現へ変換すること。

例:

```text
selected frame A -> 10.000 sec
selected frame B -> 10.041 sec
selected frame C -> 10.083 sec
selected frame D -> 10.167 sec
```

のように間隔が不均一でも、その値をそのまま保持して Qwen へ渡す。

VFR の selected frame を等間隔と仮定して timestamp を再生成してはならない。

また、平均 FPS から擬似的な frame index / timestamp を生成して Qwen に渡す fallback も禁止する。正確な PTS ベース timestamp を Qwen handoff まで伝達できない場合、その VFR 入力は正確に処理できないものとして明示的なエラーにすること。

したがって時間情報の正は以下とする。

```text
CFR:
    original source frame index + source FPS
    （または、それらから得られる正確な source timestamp）

VFR:
    selected frame ごとの PTS / time_base 由来の正確な source timestamp
```


------------------------------------------------------------------------

## 6. 未知・不正パラメータ

未知の query parameter は request error とする。

例:

``` text
?autoton=1
```

を黙って無視してはならない。

推奨レスポンス:

``` text
HTTP 400
unknown ninfer-video parameter: autoton
```

不正値も HTTP 400 とする。

例:

``` text
start_frame=-1
end_frame < start_frame
skip_frame=-2
scale=0
scale=-1.0
bbox width <= 0
bbox が decoded frame の範囲外
不正な deinterlace 値
不正な boolean 値
```

明示的に仕様化されていない限り、不正入力を黙って clamp しない。

------------------------------------------------------------------------

## 7. 内部表現

推奨データ構造:

``` cpp
enum class DeinterlaceMode {
    Auto,
    On,
    Off,
};

struct CropRect {
    int x;
    int y;
    int width;
    int height;
};

struct LocalVideoSpec {
    std::filesystem::path path;
    std::optional<std::int64_t> frame; // 単一フレーム画像モード。range指定と排他。

    std::int64_t start_frame = 0;
    std::optional<std::int64_t> end_frame;
    std::int64_t skip_frame = 0;

    std::optional<CropRect> bbox;

    double scale = 1.0;
    bool autotone = false;
    DeinterlaceMode deinterlace = DeinterlaceMode::Auto;
};
```

decode 開始前に URL を完全に parse / validate し、この表現へ変換する。

後段のコードで query string を繰り返し parse しないこと。

------------------------------------------------------------------------

## 8. セキュリティとローカルパス方針

カスタム scheme は serving process に server-local file
へのアクセス能力を与える。

したがって、信頼されたローカル環境以外へ公開する server では、任意の
filesystem access を許可してはならない。

推奨 server option:

``` text
--local-media-root /mnt/d/videos
```

カスタム動画 path は open 前に canonicalize する。

必須チェック:

``` text
canonical(requested_path) が canonical(local_media_root) 配下にあること
```

path traversal、symlink escape、設定 root 外への参照を拒否する。

拒否例:

``` text
ninfer-video:///etc/passwd
ninfer-video:///mnt/d/videos/../../etc/shadow
```

local media root
が設定されていない場合は、以下のいずれかを明示的に採用する。

1.  `ninfer-video` を無効化する。
2.  server が unrestricted local access
    を明示的に有効化した場合のみ許可する。

無制限の server-local file access を暗黙に有効化してはならない。

------------------------------------------------------------------------

## 9. FFmpeg 統合

### 9.1 同一プロセス内での library 利用

`ninfer-serve` プロセス内から FFmpeg library を直接使用する。

想定 library:

``` text
libavformat
libavcodec
libavutil
libswscale
libavfilter（デインタレースで必要な場合）
```

外部 `ffmpeg` executable を起動してはならない。

### 9.2 file-backed input

`ninfer-video` では FFmpeg の file-backed I/O で対象ファイルを直接 open
する。

圧縮動画全体を最初に `std::vector<uint8_t>` 等へ読み込んではならない。

目標構造:

``` text
local file
   ↓
avformat_open_input(path)
   ↓
demux/decode
```

以下の構造にはしない:

``` text
local file
   ↓
ファイル全体をRAMへ読み込み
   ↓
memory-backed AVIO
```

------------------------------------------------------------------------

## 10. seek とフレーム範囲

### 10.1 `start_frame` への seek 最適化

`start_frame` が動画後方にある場合、frame 0 から decode
することを避ける。

一般的な codec にはフレーム間依存があるため、任意 frame
への完全な直接アクセスは通常できない。

以下の方式を採用する。

``` text
start_frame 付近の適切な timestamp を算出
    ↓
それ以前の keyframe へ seek
    ↓
decoder を flush
    ↓
順方向へ decode
    ↓
正確な start_frame まで discard
```

必要に応じて `avformat_seek_file` 等の FFmpeg seek API を使用する。

### 10.2 `end_frame` での終了

``` text
frame_index > end_frame
```

となった時点で decode を即終了する。

残りの動画を decode する必要はない。

ただし、初回の正確なフレーム索引作成では `end_frame` より後を含む全ストリームを
走査してよい。また、stateful deinterlacerが最後の選択フレームを確定するために要求する
最小限のlookaheadは許容する。索引完成後の通常decodeでは、これらを除いて
`end_frame`到達後に残りの動画を処理しない。

### 10.3 `skip_frame` と decode cost

典型的な inter-frame codec では、skip 対象の P/B frame
も予測依存関係のため sequential decode が必要な場合がある。

したがって「skip された frame は codec decode
自体を常に省略できる」と仮定してはならない。

ただし選択されなかった frame には、不要な高コスト処理を行わないこと。

例:

-   filter が要求しない限り RGB conversion
-   crop
-   autotone
-   resize
-   Qwen 前処理
-   Vision Encoding

期待動作:

``` text
source frame を decode
  ↓
selected?
  ├─ no  -> 可能な限り早く破棄
  └─ yes -> 前処理へ進む
```

------------------------------------------------------------------------

## 11. 処理順序

選択フレームに対する論理的な処理順序は以下とする。

``` text
FFmpeg decode
    ↓
frame range / skip selection
    ↓
必要なら deinterlace
    ↓
指定されていれば bbox crop
    ↓
[将来版] automatic tone correction
    ↓
scale / resize
    ↓
Qwen が必要とする normalize
    ↓
temporal grouping / patchify
    ↓
既存 Qwen Vision Encoder
```

デインタレース方式によっては、小さな frame history を保持したり、最終
selection より前に filter を通す必要がある場合がある。そのような内部
buffering は許容する。

ただし discard 対象 frame に対して不要な crop/tone/scale/Vision
処理を行わないこと。

------------------------------------------------------------------------

## 12. Crop 実装

crop はデインタレース後、可能な限り早い段階で実行する。

可能であれば、巨大な RGB buffer へ変換した後ではなく、YUV 等の decoded
source pixel format 上で ROI を処理する。

推奨概念フロー:

``` text
decoded YUV AVFrame
    ↓
crop ROI
    ↓
scale / color conversion
```

以下より優先する:

``` text
full-frame RGB conversion
    ↓
RGB crop
```

特に 4K 以上の入力では、これにより memory bandwidth と temporary buffer
サイズを削減できる。

実装可能な範囲で FFmpeg/libswscale または同等の zero/low-copy
処理を利用する。

------------------------------------------------------------------------

## 13. Scale と出力 alignment

`scale` はユーザー指定の空間倍率である。

要求サイズ:

``` cpp
requested_w = round(crop_w * scale);
requested_h = round(crop_h * scale);
```

最終サイズは Qwen Vision の patch/grid
制約に合わせて調整が必要になる場合がある。

本プロジェクトで対象とする Qwen3.8 Vision 設定では:

``` text
patch_size = 16
spatial_merge_size = 2
```

したがって merged token の自然な空間 alignment は 32 pixel 単位となる。

既存 Qwen 前処理が受理可能な dimension を生成しつつ、aspect ratio
を可能な限り維持すること。

丸め規則は決定論的かつ文書化されていること。

推奨方針:

``` text
requested width/height を最も近い有効な Qwen grid size へ丸める
```

また最小有効サイズを保証する。

後段で独立した二回目の resize を行ってはならない。

**カスタム動画前処理経路を、patchify に渡す最終 spatial dimension
を決定する唯一の場所とする。**

------------------------------------------------------------------------

## 14. デインタレース

### 14.1 モード

`deinterlace=off`

``` text
デインタレースしない。
```

`deinterlace=on`

``` text
選択した deinterlacer を常に実行する。
```

`deinterlace=auto`

``` text
利用可能な FFmpeg の stream/frame interlace metadata に基づき、
必要な場合のみデインタレースする。
```

### 14.2 実装

同一プロセス内の FFmpeg-compatible filter path を使用する。

初期実装では `bwdif` 等の適切な FFmpeg deinterlacer を採用してよい。

採用アルゴリズムはコードまたは設定上で明記すること。

### 14.3 Auto 判定

初期の `auto` 実装では decoded frame / stream の interlace metadata
を利用してよい。

将来的に URL/API 契約を変更せず comb detection を追加できる構造にする。

------------------------------------------------------------------------

## 15. 自動色調補正（将来実装）

自動色調補正は**初期版の実装対象外**とする。

`autotone` は将来拡張用の API parameter として予約してよいが、初期版で `autotone=1` / `true` を指定された場合に、補正を行わず黙って成功させてはならない。未サポートであることを明示する。

将来実装する場合は、crop 後に tone analysis を行う方針とする。これにより、要求 ROI の統計だけを利用し、bbox 外の無関係な明部・暗部の影響を避ける。

将来の候補処理:

```text
luminance distribution を解析
    ↓
black/white point または robust percentile range を算出
    ↓
決定論的な luminance / contrast stretch
```

具体的なアルゴリズム、定数、gamma correction 等の詳細は将来版で別途仕様化する。

## 16. Streaming とメモリ要件

### 16.1 必須要件

新経路では圧縮動画全体をメモリへ読み込んではならない。

### 16.2 推奨要件

Vision 処理前に、選択された decoded frame 全体を保持しないこと。

推奨構造:

``` text
selected frame を decode
    ↓
preprocess
    ↓
Qwen が必要とする最小 temporal group のみ保持
    ↓
Vision Encode
    ↓
生成された Vision embedding/token を append
    ↓
pixel buffer を解放
```

### 16.3 Temporal grouping

Qwen の動画前処理では temporal grouping を使用する。

本プロジェクトの Qwen3.8 設定では temporal patch size は 2 を想定する。

したがって、動画全体を保持するのではなく、既存 Qwen 前処理 / Encoder
が受理する最小 temporal chunk を単位として設計する。

概念例:

``` text
selected frame A
selected frame B
    ↓
temporal pair を preprocess
    ↓
Vision Encode
    ↓
A/B の pixel buffer を解放

selected frame C
selected frame D
    ↓
...
```

最後の selected frame 数が temporal patch size の倍数でない場合、既存
Qwen/NInfer の挙動を厳密に再現すること。

たとえば現行 processor が最終 frame を複製して padding
しているなら、同じ処理を行う。

モデル semantics を変える独自の padding 規則を導入してはならない。

### 16.4 等価性要件

同じ selected frame 列に対して、streaming/chunked 前処理は、同等の
non-streaming 前処理と論理的に同じ Qwen Vision input/output
を生成すること。通常の浮動小数点誤差は許容する。

現行 Vision 実装に chunk 間依存があり完全な chunking
ができない場合、メモリ削減より correctness を優先する。

------------------------------------------------------------------------

## 17. Vision Encoder との統合

既存 Qwen Vision Encoder / Vision Tower は、chunked input
を可能にする最小限の API 拡張が必要な場合を除き変更しない。

新規コンポーネントの主な責務は以下である。

``` text
local video source
+ decoder
+ frame selection
+ image preprocessing
+ streaming/chunking adapter
```

新しい neural Vision model を実装するものではない。

handoff boundary は可能な限り既存概念:

``` text
pixel_values_videos
video_grid_thw
```

または NInfer 内部の同等表現と互換にする。

chunking のため incremental interface が必要な場合、既存 Vision
実装の周囲に最小限の adapter を追加する。

------------------------------------------------------------------------

## 18. Pixel / Token Budget の扱い

本 API は target pixel count ではなく `scale` のみを公開する。

必要な scale は呼び出し側で計算できる。

例:

crop size:

``` text
640 x 360
```

目標:

``` text
約500,000 pixels
```

の場合、呼び出し側は:

``` text
scale = sqrt(500000 / (640 * 360))
      ≈ 1.47
```

を計算し、

``` text
scale=1.47
```

を送信できる。

NInfer は Qwen grid alignment および実装上の安全制限を除き、指定された
scale を適用する。

この resize の後に、「Qwen公式 pixel budget」等を理由とした暗黙の追加
resize を行わないこと。

------------------------------------------------------------------------

## 19. 推奨安全制限

`scale` は caller-controlled であるため、異常な allocation から server
を保護する必要がある。

推奨 configurable limit:

``` text
最大 decoded source dimension
最大 post-scale frame dimension
selected frame 1枚あたりの最大 post-scale pixel 数
最大 selected frame 数
最大 total Vision token 数
必要に応じて最大 local file size
```

制限超過時は OOM させたり `scale`
を黙って変更したりせず、明示的なエラーを返す。

これらは運用上の安全制限であり、要求された `scale`
を暗黙に変更する機能ではない。

------------------------------------------------------------------------

## 20. エラー処理

推奨エラー:

### malformed custom URL

``` text
HTTP 400
invalid ninfer-video URL
```

### file not found

``` text
HTTP 400 または 404
local video file not found
```

### 許可 root 外の path

``` text
HTTP 403
local video path is outside configured media root
```

### 不正 frame range

``` text
HTTP 400
end_frame must be >= start_frame
```

### 不正 bbox

``` text
HTTP 400
bbox exceeds decoded frame bounds
```

### 不正 scale

``` text
HTTP 400
scale must be > 0
```

### 不正 deinterlace mode

``` text
HTTP 400
deinterlace must be one of: auto, on, off
```

### Resource limit 超過

``` text
HTTP 413 または 400
requested video preprocessing exceeds configured limit
```

エラーメッセージには問題となった parameter を含める。

------------------------------------------------------------------------

## 21. Logging

debug/info level では、必要に応じて path
の機密部分を除外したうえで、解決済み spec を簡潔に log する。

有用な項目:

``` text
source dimensions
source fps
effective fps（参考値）
selected frame の original source index / timestamp
resolved start_frame
resolved end_frame
skip_frame
selected frame count（判明可能な場合）
bbox
scale
resolved output dimensions
autotone
deinterlace mode
実際に deinterlace が適用されたか
seek target / 実際の開始 frame
Vision temporal group count
```

raw video contents や base64 data を log しない。

------------------------------------------------------------------------

## 22. 推奨実装分割

実際の filename は現行 repository structure
に従うこと。責務は以下のように分離することを推奨する。

### URL/spec parser

責務:

``` text
ninfer-video scheme の検出
path の parse
query parameter の parse
静的 parameter validation
LocalVideoSpec の生成
```

### Local video decoder

責務:

``` text
libavformat で file open
stream probe
start_frame 付近へ seek
順方向 decode
正確な frame index 管理
end_frame で停止
selected frame のみ後段へ emit
```

### Frame preprocessor

責務:

``` text
deinterlace
crop
[将来版] autotone
scale
format conversion
Qwen-aligned dimension
```

### Streaming Qwen video adapter

責務:

``` text
最小 temporal group を収集
normalize
patchify
video grid metadata を生成
既存 Vision Encoder を呼び出す
Vision embedding/token を append
pixel memory を解放
```

可能な限り通常の media acquisition/decode コードは変更しない。

------------------------------------------------------------------------

## 23. 疑似コード

``` cpp
Response handle_video_url(std::string_view url) {
    if (!is_ninfer_video_url(url)) {
        return existing_video_path(url);
    }

    LocalVideoSpec spec = parse_local_video_spec(url);
    validate_local_path(spec.path);

    LocalVideoDecoder decoder(spec.path);

    decoder.seek_near_frame(spec.start_frame);

    StreamingVisionPipeline vision_pipeline{
        .scale = spec.scale,
        .bbox = spec.bbox,
        // autotone is reserved for a future version.
        .deinterlace = spec.deinterlace,
    };

    while (auto frame = decoder.next_frame()) {
        const int64_t i = frame.index;

        if (i < spec.start_frame) {
            continue;
        }

        if (spec.end_frame && i > *spec.end_frame) {
            break;
        }

        const bool selected =
            ((i - spec.start_frame) % (spec.skip_frame + 1)) == 0;

        if (!selected) {
            continue;
        }

        vision_pipeline.push(std::move(frame));
    }

    vision_pipeline.finish();

    return vision_pipeline.result();
}
```

実際には stateful な deinterlace filter の都合により、最終 selection
より前に frame を filter へ流す必要がある可能性がある。

それでも selection により、不要な crop/tone/scale/Vision
処理を行わないこと。

------------------------------------------------------------------------

## 24. 後方互換性要件

以下の入力は現行挙動を維持すること。

``` text
https://...
http://...
data:video/...
現在サポートされているその他すべての通常 video_url
```

本機能が build に含まれただけで既存 request の挙動が変化してはならない。

新経路を選択するのは `ninfer-video://` のみ。

既存 image handling も変更しない。

------------------------------------------------------------------------

## 25. テスト計画

### 25.1 URL parser test

以下を test する。

``` text
default 値
全 parameter 指定
小数 scale
不正 scale
不正 bbox syntax
負の start_frame
start_frame より前の end_frame
未知 parameter
不正 boolean
不正 deinterlace mode
URL encoded path
```

### 25.2 Frame selection test

frame 番号が判別できる synthetic video を使用する。

検証:

``` text
start=0,end=9,skip=0 -> 0..9
start=0,end=9,skip=1 -> 0,2,4,6,8
start=1,end=10,skip=2 -> 1,4,7,10
start=100,end=100,skip=99 -> 100
```

### 25.3 Seek correctness

以下の selected decoded frame を比較する。

``` text
frame 0 から sequential decode
```

対:

``` text
start_frame 付近へ seek + forward decode
```

同じ source frame を取得すること。

### 25.4 FPS / timestamp test

CFR 動画について以下を検証する。

```text
source_fps=30, skip=0 -> effective_fps=30
source_fps=30, skip=1 -> effective_fps=15
source_fps=30, skip=2 -> effective_fps=10
source_fps=29.97, skip=2 -> effective_fps≈9.99
```

ただし `effective_fps` は参考値であり、Qwen timestamp の計算には使用しないことを確認する。

例えば:

```text
source_fps=30
start_frame=100
skip_frame=2

selected source indices:
[100, 103, 106, 109]
```

に対して、Qwen に渡す時間情報が元動画上の:

```text
[3.333..., 3.433..., 3.533..., 3.633...]
```

と一致すること。

また、非等間隔 frame index:

```text
[300, 303, 306, 450, 900]
```

を入力でき、対応 timestamp:

```text
[10.0, 10.1, 10.2, 15.0, 30.0]
```

が保持されることを確認する。

VFR test fixture を必須で用意し、各 selected frame の PTS / time base から算出した正確な timestamp が Qwen handoff まで保持されることを確認する。

平均 FPS や `effective_fps` から再生成した近似 timestamp が使用されていないことも確認する。

### 25.5 Crop test

既知 geometry の frame を使い、ROI が正確に抽出されることを確認する。

境界に接する bbox を test し、範囲外 bbox を拒否する。

### 25.6 Scale test

以下のような小数 scale を検証する。

``` text
0.5
0.75
1.0
1.25
1.5
2.0
```

Qwen grid に alignment された最終 dimension が決定論的であること。

### 25.7 Deinterlace test

以下を test する。

``` text
progressive source + auto
interlaced source + auto
interlaced source + on
interlaced source + off
```

### 25.8 Autotone test

初期版では省略、`0`、`false`を受理し、`1`、`true`は未サポートのrequest errorにする。
未知または不正なboolean表現もrequest errorにする。色調補正の数値テストは、将来
アルゴリズムを仕様化して実装する時点で追加する。

### 25.9 後方互換 test

既存 video-url test suite を変更せず実行する。

以下に影響がないことを確認する。

``` text
HTTP(S) input
data URI input
通常 image input
通常 text-only input
```

### 25.10 Memory test

非常に大きな local video file を使用する。

host memory が圧縮動画ファイルサイズ相当だけ増加しないことを確認する。

不可避な累積 Vision embedding/token を除き、selected RGB frame
数に比例して memory が増加しないことを確認する。

### 25.11 Streaming equivalence test

短い動画について、同じ selected frame を入力し:

``` text
existing/full-batch Qwen preprocessing
```

と:

``` text
new streaming/chunked preprocessing
```

を比較する。

Vision input/output が期待される浮動小数点 tolerance 内で一致すること。

------------------------------------------------------------------------

## 26. 受け入れ条件

以下をすべて満たした時点で実装完了とする。

1.  `ninfer-video://` request が通常 media acquisition
    より前に認識される。
2.  `ninfer-video://` 以外の request は既存実装を変更せず通る。
3.  server が FFmpeg library 経由で local video file を直接 open する。
4.  圧縮動画全体を RAM に読み込まない。
5.  `start_frame`、`end_frame`、`skip_frame` が決定論的な selected frame
    を生成する。
6.  実用上可能な場合、seek により動画先頭からの decode を回避する。
7.  `bbox` が source pixel 座標で selected frame を crop する。
8.  `scale` が正の小数値を受理する。
9.  resize は Qwen patchify 前に一度だけ実行される。
10. `autotone` の省略、`0`、`false`を受理し、`1`、`true`は未サポートとして明示的に拒否する。
11. `deinterlace` が `auto`、`on`、`off` をサポートする。
12. 初回索引走査とdeinterlacerの最小lookaheadを除き、`end_frame` 到達後は処理を停止する。
13. 非 selected frame に不要な高コスト前処理を行わない。
14. selected frame を、実用上最小の temporal buffering で streaming
    処理する。
15. 既存 Qwen Vision semantics を維持する。
16. 通常の既存 video/image/text 処理を変更しない。
17. 不正または未知の custom parameter に明示的なエラーを返す。
18. local filesystem access が明示的な security policy
    によって制限される。
