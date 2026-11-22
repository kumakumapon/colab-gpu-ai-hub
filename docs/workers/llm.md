# Qwen テキスト推論

`llm_infer` は Qwen2.5 Instruct でチャット形式の応答を生成します。入力は
`parameters.prompt` の文字列、またはワークスペース内の UTF-8 `.txt` 1本です。
文字数は12,000まで、チャットテンプレート適用後の入力は4,096トークンまでです。
空の入力や上限超過はエラーになります。ファイルの事前検証で失敗した場合は
`input_validation_failed`、ワーカーの入力検証で失敗した場合は `invalid_worker_input` を返します。

| パラメータ | 既定値 | 設定値 |
|---|---|---|
| `model_id` | `Qwen/Qwen2.5-0.5B-Instruct` | 既定値または `Qwen/Qwen2.5-3B-Instruct` |
| `max_new_tokens` | `128` | 1〜512 の整数 |
| `seed` | `0` | 0〜4,294,967,295 の整数 |
| `temperature` | `0` | 0 は貪欲法、0 より大きく 2 以下はサンプリング |
| `stop` | `[]` | 最大4個、各1〜64文字の停止文字列 |

`response.json` には `text` と、モデル ID・リビジョン、入出力のトークン数、
トークナイザーとモデルの初期化時間、生成時間、生成速度、PyTorch のピーク割当メモリ、
終了理由が入ります。トークン数には EOS や除去した停止文字列に対応するトークンも含みます。
生成テキストと元のプロンプトはジョブ台帳の計測値には入りません。
生成結果に含まれる事実や数値は、利用する側で出典を確認してください。

## T4 スモーク

[ワーカーの実行例](whisper.md#wsl2-からの-t4-スモーク)の準備後、短い入力を使います。

```python
result = runner.run(TaskRequest(
    operation="llm_infer",
    parameters={"prompt": "日本語で短い挨拶を一文だけ書いてください。",
                "temperature": 0, "max_new_tokens": 24},
))
print(result.model_dump_json(indent=2))
```

`workspace/prompt.txt` を使う場合は `inputs=["prompt.txt"]` を指定し、
`parameters` から `prompt` を外します。0.5B で応答、トークン数、生成速度、
`cleanup_status=confirmed` を確認した後、同じ短文で `model_id` を 3B に変えて確認できます。
VRAM が不足すると `out_of_memory` を返します。モデルは指定されたリビジョンを
safetensors で取得します。Transformers 5.10.4、Jinja2 3.1.6、T4 向けの FP16 を使用し、量子化は行いません。
2026-11-22 に、同じ短い日本語プロンプトを使って T4 で両モデルを実行しました。

| モデル | モデルロード | 生成時間 | 出力トークン | ピーク割当メモリ |
|---|---:|---:|---:|---:|
| 0.5B | 8.96 秒 | 5.40 秒 | 7 | 960.9 MiB |
| 3B | 100.19 秒 | 5.35 秒 | 7 | 6007.5 MiB |

両モデルとも日本語応答を生成し、成果物の SHA-256 一致と停止後のランタイム 0 件を確認しました。

0.5B は Apache-2.0 です。3B は Qwen Research License の対象で、ライセンス上の
利用範囲は非商用の研究・評価です。商用利用には別途ライセンスが必要です。

- [0.5B のモデルカード](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [3B のライセンス](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
