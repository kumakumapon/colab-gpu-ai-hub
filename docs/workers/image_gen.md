# Stable Diffusion v1.5 画像生成

`generate_image` は、Diffusers 0.43.0 と T4 向けの FP16 で画像を生成します。
モデルは `stable-diffusion-v1-5/stable-diffusion-v1-5` の固定リビジョン
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14` です。既定のスケジューラと安全チェッカーを使用します。

```sh
uv run gpu-ai run generate_image --prompt "A small cabin at sunrise" --parameters '{"seed":42,"steps":15,"images_per_prompt":2,"contact_sheet":true}'
```

| パラメータ | 既定値 | 設定値 |
|---|---|---|
| `prompt` | 必須 | 空白以外を含む文字列 |
| `negative_prompt` | 空文字列 | 除外する内容。設定時は guidance を1より大きくする |
| `seed` | 0 | 0〜4,294,967,295 |
| `width`、`height` | 512、512 | 各256〜768、64の倍数 |
| `steps` | 20 | 1〜40 |
| `guidance_scale` | 7.5 | 1〜15 |
| `images_per_prompt` | 1 | 1〜4 |
| `contact_sheet` | `false` | `contact_sheet.png` を追加 |

プロンプトはそれぞれ600文字までで、CLIP のトークン上限も確認します。画像は1枚ずつ生成します。結果は `image_0001.png` からの連番で、PNG は1枚4 MiB、一覧画像は16 MiBまでです。`metadata.json` に各画像の seed、モデルリビジョン、スケジューラ、生成時間、ピーク割当メモリを記録します。2枚目以降の seed は1ずつ加算します。安全チェッカーが拒否した場合はジョブを失敗にし、成果物の配布を中止します。

固定 seed と同じモデル・設定・実行環境での再現性は、2回の出力を比較して確認します。GPU、ドライバ、PyTorch の違いによって結果が変わることがあります。T4 スモークでは生成画像の枚数・サイズ、`generation_time_ms`、`peak_vram_mb`、停止状態とランタイム残存を確認します。モデル推論と解放の実機検証は未実施です。

モデルの CreativeML Open RAIL-M は利用制限への準拠を条件に商用利用を認めています。重みを再配布する場合は、ライセンスと利用制限を受領者に引き継ぐ条件を確認してください。

- [Stable Diffusion v1.5 モデルカード](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)
- [Diffusers Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img)
