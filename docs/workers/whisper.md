# 日本語音声認識

`transcribe_audio` は WAV または MP3 を1本受け取り、`transcript.json` と UTF-8 の
`transcript.srt` を生成します。動画の音声は [前処理](../contracts.md#入力ファイルと成果物)
で WAV に変換できます。音声が空または読み込めない場合は `invalid_worker_input` を返します。
有効な音声から発話が検出されない場合は、空のセグメント配列と空の SRT を保存します。

| パラメータ | 既定値 | 設定値 |
|---|---:|---|
| `language` | `ja` | `ja` または `auto` |
| `vad_filter` | `true` | 真偽値 |
| `beam_size` | `5` | 1〜10 の整数 |

## WSL2 からの T4 スモーク

[セットアップ手順](../setup-wsl2.md)で Colab CLI と OAuth の疎通を確認し、
ワークスペースの `workspace/audio/sample.wav` に使用許可のある短い日本語音声を置きます。
以下を実行すると T4 の利用枠を消費する場合があります。

```python
from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts

settings = Settings()
runner = JobRunner(ColabCLIBackend(colab_worker_scripts(), settings=settings), settings)
result = runner.run(TaskRequest(
    operation="transcribe_audio", inputs=["audio/sample.wav"],
    parameters={"language": "ja", "vad_filter": True, "beam_size": 5},
))
print(result.model_dump_json(indent=2))
```

`completed` と `cleanup_status=confirmed`、入力の長さの範囲に収まる字幕時刻、
日本語テキスト、停止後のランタイム一覧を確認します。計測値はモデル取得・初期化・推論の
各時間と、音声の長さに対する推論時間の比です。GPU メモリは `nvidia-smi` が取得できた場合の
初期化前からの差分で、瞬間的なピークは含みません。通常 CI では短い WAV と日本語の模擬推論を使います。

モデルは多言語の `Systran/faster-whisper-small` です。Colab 側で faster-whisper 1.2.0 と
CTranslate2 4.6.0、PyAV 17.1.0 を使用し、CUDA 12 の cuBLAS と cuDNN 9、CUDA の FP16 を確認します。
faster-whisper 1.2.0 が音声の読み込みに使う引数は PyAV 18 で削除されたため、互換版を固定しています。
CUDA ライブラリはランタイムに用意されているものを使用します。
2026-11-22 の T4 実機スモークでは、約4.52秒の合成日本語 WAV から JSON と SRT を生成しました。
モデル取得 3.92 秒、ロード 0.71 秒、推論 1.48 秒、実時間比 0.33、GPU メモリ差分 726 MiB でした。
成果物の SHA-256 一致と停止後のランタイム 0 件を確認しました。

- [faster-whisper の GPU 要件](https://github.com/SYSTRAN/faster-whisper/tree/v1.2.0#gpu)
- [faster-whisper Small モデル](https://huggingface.co/Systran/faster-whisper-small)
