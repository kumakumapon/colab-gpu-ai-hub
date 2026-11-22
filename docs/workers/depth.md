# 相対深度推定

`estimate_depth` は JPEG または PNG を1〜4枚受け取り、Depth Anything V2 Small で
相対逆深度を推定します。画像内で値が大きい領域は手前として推定された領域です。
絶対距離の単位はありません。画像をまたいだ値の比較や物理距離への換算には使えません。

| パラメータ | 既定値 | 設定値 |
|---|---:|---|
| `input_size` | `518` | 280〜700 の 14 の倍数 |
| `device` | `cuda` | `cuda` または `cpu` |
| `normalization` | `imagenet` | チェックポイント既定の ImageNet 正規化 |

モデルのアスペクト比を維持する前処理を使用し、前処理後のサイズが同じ画像をまとめて推論します。
入力は合計1,600万画素まで、各画像の長辺と短辺の比は12以下です。
`device=cpu` はワーカーの診断用です。Colab バックエンドはジョブ契約で指定された GPU を確保します。

## 出力

- `depth.npy` は元画像と同じ高さ・幅の float32 相対逆深度です。
- `depth_preview.png` は画像ごとの最小値・最大値で正規化した 16 ビット PNG です。
- `metadata.json` は入力パス・画像サイズ・モデル ID とリビジョン・相対逆深度の範囲・バッチの
  推論時間・PyTorch のピーク割当メモリを記録します。

非有限値は有効範囲の端に置き換えます。有効値がない場合は処理が失敗します。
2枚以上の場合は入力順に `images/0001/`、`images/0002/` などへ保存します。

## T4 スモーク

[ワーカーの実行例](whisper.md#wsl2-からの-t4-スモーク)と同じ方法で、
ジョブのリクエストを次のように作ります。`workspace/frames/` に利用許可のある画像を置いてください。

```python
request = TaskRequest(
    operation="estimate_depth",
    inputs=["frames/sample.jpg", "frames/next.png"],
    parameters={"input_size": 518, "device": "cuda"},
)
result = runner.run(request)
print(result.model_dump_json(indent=2))
```

完了とGPU停止、各 `.npy` と元画像のサイズ一致、`inference_time_ms` と
`peak_vram_mb` を確認します。T4 でのモデル推論・計測は未実施です。
モデルは Apache-2.0 の Small チェックポイントを固定リビジョンで取得し、
Colab の PyTorch と Transformers 4.51.3 を使用します。

- [Depth Anything V2 Small モデルカード](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- [公式前処理設定](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/5426e4f0f36572d16453bbda7a8389317b1bef99/preprocessor_config.json)
