# colab-gpu-ai-hub

Claude Code と Codex から、必要なときに Google Colab の GPU へ推論を委譲するための Python プロジェクトです。現在はジョブの共通スキーマ、設定、ワーカー登録、ローカルの MockBackend が利用できます。

## 開発環境

Python 3.11 以上と uv を用意し、リポジトリのルートで実行します。テストは CPU のみで動作し、Google 認証は不要です。

```sh
uv sync
uv run ruff check
uv run pytest -q
```

設定と JSON 入出力の例は [共通契約](docs/contracts.md) を参照してください。
