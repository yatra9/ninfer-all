---
name: ninfer-video
description: NInferのMCPツールでローカル動画を実際に見る。単一frameの画像確認、動画範囲の探索、時刻とsource frameの対応、cropやscaleでの詳細確認に使用する。
---

# NInfer video inspection

`get_video_metadata`、`resolve_video_time`、`inspect_video` はNInfer内蔵MCPサーバーのツール。
OpenCodeではMCP名を含むtool名として公開されるので、利用可能なtoolsから該当するものを選ぶ。
`inspect_video` が返すresource linkは、同じagentの次のモデル入力へ視覚情報として渡される。

## Path

- path引数は絶対local path。例：`/videos/demo.mp4`、map設定がある場合の `C:\Videos\demo.mp4` や `/home/koji/videos/demo.mp4`。
- Windows pathをJSONで書くときはbackslashをescapeする。例：`"C:\\Videos\\demo.mp4"`。`C:/Videos/demo.mp4` も使用できる。
- host pathはサーバーの `--reference-path-map HOST_DIR CONTAINER_DIR` が設定されている場合に使用する。
- mapはmountを作らない。返されるmetadata pathやnative URIはcontainer pathであり、それをhost側のread/fileツールで開こうとしない。
- pathに `file://`、`ninfer-video://`、選択用queryを付けない。選択条件は別のtool引数を使う。

## First inspection

まず `get_video_metadata(path)` でsource寸法、duration、FPS、CFR/VFR、frame count、interlace情報を確認する。
不明な値は不明のまま扱う。frame countがnullの場合は、duration×FPSをexact countとして扱わない。

1枚を見る場合：

```json
{"path":"C:/Videos/demo.mp4","instruction":"このframeの表示内容を確認して","frame":100}
```

frameは0始まり。あなた自身がそのframeを画像として見られる。
frameとstart_frame/end_frame/skip_frameは、明示した値が0でも同時指定できない。
bbox、scale、deinterlaceはframeと併用できる。

## Time range

CFRと固定FPSが確認済みなら、通常は `frame ≈ time_seconds × fps` で境界を求める。
正確な境界が必要、VFR、またはCFR/VFR不明の場合は、開始と終了について別々に
`resolve_video_time(path, time_seconds)` を呼び、返された実timestampからframeを選ぶ。
VFRにaverage_fps/nominal_fpsを掛けてframe番号を推定しない。

時刻はsource frame 0のPTSを0秒とする。nearestが前後に同距離なら前側を選ぶ。
開始境界は目的時刻以降、終了境界は目的時刻以前のframeを選ぶ必要がある場合、近傍framesも確認する。
最終frame timestampを超える時刻はerrorになる。

## Coarse to detailed inspection

```json
{"path":"/videos/demo.mp4","instruction":"画面が切り替わる箇所を探して","start_frame":0,"end_frame":299,"skip_frame":29,"scale":0.5}
```

- start/endはinclusiveなsource frame番号。skip_frame=Nは選択frame間でN枚飛ばす（29なら30枚ごと）。
- 長い動画は短い範囲と大きめのskipから探索し、気になる区間を絞って再確認する。飛ばしたframeの出来事まで断言しない。
- 詳細は `frame=N`、または短い範囲＋小さいskipで見る。bboxはsource pixel座標のx/y/width/height。
- cropしてからscaleが適用される。出力寸法は最寄りの32の倍数、最低32へ丸められる。
- deinterlaceは通常auto。明確な理由がある場合にon/offを指定する。
- tool/inferenceがbudget errorになったら、範囲、選択frame数、scale、cropを調整する。
- 見えない細部・sampling間隔内の出来事・音声内容を推測しない。これらのtoolsは音声をモデルへ送らない。

## Failures

接続やtool discoveryに失敗したらMCP URL、認証、provider設定を確認する。
path errorはmount、map、local-media-rootと実ファイル名を確認する。
resource linkが返っても視覚入力が届かなければ、指定patched OpenCodeと `opencode-ninfer` providerを確認する。
実際に視覚入力を受けていない場合は、その状態を明示する。
