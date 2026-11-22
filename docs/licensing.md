# モデルとソフトウェアのライセンス

このリポジトリのコードは [MIT](../LICENSE) です。各モデルは実行時に upstream から取得します。モデルの利用・配布条件と、実行ライブラリの条件をそれぞれ確認してください。

| ワーカー | ソフトウェア | モデルと条件 |
|---|---|---|
| Whisper | [faster-whisper（MIT）](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.0/LICENSE)、[CTranslate2（MIT）](https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE) | [Systran/faster-whisper-small（MIT）](https://huggingface.co/Systran/faster-whisper-small) |
| 深度推定 | [Transformers（Apache-2.0）](https://github.com/huggingface/transformers/blob/main/LICENSE) | [Depth Anything V2 Small（Apache-2.0）](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)。Small 以外への切り替え時は各チェックポイントの条件を確認してください。 |
| Qwen | [Transformers（Apache-2.0）](https://github.com/huggingface/transformers/blob/main/LICENSE) | [0.5B Instruct（Apache-2.0）](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。[3B Instruct（Qwen Research License）](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE) は非商用の研究・評価向けです。 |
| PaddleOCR | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)、[PaddlePaddle](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE)（Apache-2.0） | PP-OCRv5 Mobile の [検出器](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det)、[多言語認識器](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec)、[英語認識器](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec)（Apache-2.0）。 |
| 画像生成 | [Diffusers（Apache-2.0）](https://github.com/huggingface/diffusers/blob/main/LICENSE) | [Stable Diffusion v1.5（CreativeML Open RAIL-M）](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)。利用制限と再配布時の条件を [ライセンス本文](https://huggingface.co/spaces/CompVis/stable-diffusion-license) で確認してください。 |
| SAM 2.1 | [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)、[SAM 2](https://github.com/facebookresearch/sam2/blob/main/LICENSE)（Apache-2.0） | [SAM 2.1 Hiera Tiny（Apache-2.0）](https://huggingface.co/facebook/sam2.1-hiera-tiny)。 |
| 動画分析 | 上記の Whisper、深度推定、SAM 2.1、Qwen と [FFmpeg](https://ffmpeg.org/legal.html) | 各段階のモデルに同じ条件が適用されます。既定の報告生成は Qwen 0.5B です。 |

CUDA、Colab 基盤イメージ、追加パッケージにも個別の条件が適用されます。使用バージョンとモデル ID は各ワーカーの文書と実行結果を参照してください。
