# PaddleOCR

`extract_text` は、ワークスペース内の PNG、JPEG、PDF を1点受け付けます。PP-OCRv5 Mobile の検出器と、日本語・英語用の認識モデルを使用します。

```sh
uv run gpu-ai run extract_text --input samples/document.pdf --parameters '{"language":"ja","dpi":144,"markdown":true}'
```

| パラメータ | 既定値 | 設定値 |
|---|---|---|
| `language` | `ja` | `ja` は多言語の PP-OCRv5 Mobile Rec、`en` は英語専用版 |
| `dpi` | 144 | PDF の解像度。96〜200 |
| `markdown` | `false` | UTF-8 の `ocr.md` を追加 |
| `device` | `cuda` | `cuda` または `cpu` |
| `allow_cpu_fallback` | `false` | GPU が利用できない場合に CPU での実行を許可 |

PDF は 20 MiB、8ページまでです。PDFium でページごとに画像化し、暗号化 PDF や読み取れない PDF は拒否します。1辺4096画素、全ページ合計3200万画素までです。認識結果の座標はラスタライズ後の左上を原点とするピクセル座標です。JSON はページ順に並び、各ページの `reading_order` は認識器が返した検出順を示します。縦書き・横書きの認識はモデルの対応範囲に従い、縦横混在や表のセル構造の復元は行いません。日本語の文字列は Unicode で保持します。

`ocr.json` の各ページにサイズ、認識時間、本文、信頼度、四隅の座標を記録します。本文は成果物に保存し、ジョブ計測値にはモデル名、版、実行デバイス、CPU フォールバックの有無を記録します。

## T4 / CPU 確認

GPU 実行は、PaddleOCR 3.6.0、PaddleX 3.6.0、PaddlePaddle GPU 3.3.1 の公式 CUDA 12.6 配布を使用します。PDFium の Python バインディングは 5.15.2 です。ワーカーの子プロセスで認識器を実行します。CPU を明示した場合は PaddlePaddle の CPU 配布を使用し、Colab ジョブのランタイムには共通のプロビジョニング設定が適用されます。モデルと実装のライセンスは PaddleOCR 公式の Apache-2.0 ライセンスを参照してください。

利用を許可した日本語・英語・表の画像と2ページ PDF を使い、T4 と `device=cpu` で `ocr.json` の内容・座標・実行デバイス・処理時間を比較します。実行結果の停止状態とランタイム残存も確認します。Colab の CUDA／Paddle 組み合わせの実測、T4 スモーク、CPU との実機比較は未実施です。

- [PaddleOCR のモデル構成と API](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- [PaddlePaddle CUDA 12.6 配布](https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/)
- [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
