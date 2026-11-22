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

Colab の Linux x86_64・Python 3.13 では、公式 GPU wheel を分割取得し、SHA-256 の照合後に uv で専用の仮想環境へインストールします。既設の PyTorch と CUDA 依存の衝突により `paddleocr` の import が失敗したため、ワーカー専用の依存関係で実行します。wheel の取得時間と依存関係のインストール時間は推論時間と別に記録します。

利用を許可した日本語・英語・表の画像と2ページ PDF を使い、T4 と `device=cpu` で `ocr.json` の内容・座標・実行デバイス・処理時間を比較します。実行結果の停止状態とランタイム残存も確認します。

2026-11-22 の T4 実機スモークで、合成 PNG から日本語・英語・表の文字と画像範囲内の座標を取得しました。CUDA 12.6 を使い、CPU フォールバックはありませんでした。初回の wheel 取得は166.4秒、依存のインストールは70.5秒、モデルロードは9.50秒、推論は0.82秒でした。成果物の SHA-256 一致、自動停止、停止後のランタイム0件を確認しました。

同じ2ページ PDF を T4 と CPU で処理し、認識した行の文字列が一致しました。T4 の推論は0.88秒、CPU は3.23秒でした。どちらも2ページ目の日本語で一部誤認識がありました。T4 の PDF ジョブでも、座標の範囲、成果物の SHA-256 一致、自動停止、停止後のランタイム0件を確認しました。

- [PaddleOCR のモデル構成と API](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- [PaddlePaddle CUDA 12.6 配布](https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/)
- [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
