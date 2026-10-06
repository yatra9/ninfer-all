# NInfer video

動画の内容を確認する依頼では、`ninfer-video` skillを読み、NInferのMCP動画ツールを使用する。
`inspect_video` の結果は同じあなた自身への視覚入力になる。実際に見た内容に基づいて答え、ファイル名やmetadataだけから内容を推測しない。

入力pathは絶対local pathを使う。コンテナから見えるpath、またはサーバーの
`--reference-path-map` に対応したWindows/Linux host pathを指定する。URIやquery optionsをpath引数に付けない。
1枚を見る場合は `inspect_video(frame=N)`、範囲を見る場合はstart/end/skipを使い分ける。
