# colab-gpu-ai-hub

Claude Code と Codex から、必要なときに Google Colab の GPU へ推論を委譲するための Python プロジェクトです。ジョブの共通スキーマ、Colab CLI アダプター、ジョブ台帳、安全なファイル転送と成果物保存の基盤を提供します。

## 開発環境

Python 3.11 以上、uv、FFmpeg（`ffmpeg` と `ffprobe`）を用意し、リポジトリのルートで実行します。テストは CPU のみで動作し、Google 認証は不要です。

```sh
uv sync
uv run ruff check
uv run pytest -q
```

設定と JSON 入出力の例は [共通契約](docs/contracts.md) を参照してください。

Colab CLI の導入と OAuth 診断は [WSL2 セットアップ](docs/setup-wsl2.md) を参照してください。

## MVP ワーカー

- [Whisper 日本語音声認識](docs/workers/whisper.md)
- [Depth Anything V2 相対深度推定](docs/workers/depth.md)
- [Qwen テキスト推論](docs/workers/llm.md)

[統一 CLI とローカル MCP の使い方](docs/cli-mcp.md)

### 追加ワーカー

- [PaddleOCR 画像・PDF の文字認識](docs/workers/ocr.md)
