# SAM 2.1 のマスク生成と動画追跡

`segment_object` は PNG、JPEG または短い MP4 を1点受け取り、SAM 2.1 Hiera Tiny を使って
指定した物体のマスクを生成します。対象は画像上の点か矩形で指定します。

```sh
uv run gpu-ai run segment_object --input frames/sample.png --parameters '{"objects":[{"object_id":1,"point":[120,80]}]}'
uv run gpu-ai run segment_object --input clips/sample.mp4 --parameters '{"start_frame":2,"frame_count":8,"objects":[{"object_id":4,"box":[30,20,120,100]}],"overlay":true}'
```

- `objects` は1〜4個です。各 `object_id` は重複しない1〜9999の整数です。
  `point` は物体内の1点 `[x,y]`、`box` は `[left,top,right,bottom]` です。右端と下端は排他的です。
- 動画の `start_frame` は元動画の表示順で0始まりです。既定値は0です。追跡する `frame_count` を指定します。
  元動画は8秒・48フレームまでです。1辺1280画素、1フレーム921,600画素、入力全体3200万画素までです。
- `overlay=true` で動画のマスクを色付きで重ねた `overlay.mp4` を追加します。重ね合わせ動画は音声を含まず、
  選択した区間の先頭を時刻0として出力します。

`masks/` の PNG は入力と同じ寸法で、画素値は0か255です。`tracks.json` は
元動画の `frame_index`、先頭からの `time_sec`、元の `pts` と `time_base`、
`object_id`、マスクへのパス、マスクから求めた外接矩形を記録します。マスクが空の場合の矩形は `null` です。
可変フレームレートでは元の表示時刻を保持します。`metadata.json` にモデルと版、平均 FPS、
フレーム数、処理時間とピーク割当メモリを記録します。

## T4 スモーク

短い動画と静止画を用意し、点と矩形をそれぞれ指定します。静止画の出力サイズ、フレームごとの
同一 `object_id`、時刻とコマ数、`peak_vram_mb`、成果物のハッシュ、停止後のランタイム残存を確認します。
T4 での推論・解放の実機確認は未実施です。

重みは `facebook/sam2.1-hiera-tiny` の固定リビジョンを safetensors で取得します。
Transformers 5.10.4 と Colab の CUDA 対応 PyTorch を使用します。T4 では FP32 とし、前処理とフレーム保持に CPU を使います。
PyAV 17.1.0 で MP4 の表示時刻を読み取ります。モデルと実装の使用条件は公式 Apache-2.0 ライセンスを参照してください。

- [SAM 2.1 Hiera Tiny モデルカード](https://huggingface.co/facebook/sam2.1-hiera-tiny)
- [公式 SAM 2 リポジトリ](https://github.com/facebookresearch/sam2)
- [Transformers の動画セグメンテーション API](https://huggingface.co/docs/transformers/model_doc/sam2_video)
