# Codex で GPU AI Hub を使う

[CLI と MCP の設定](docs/cli-mcp.md)に従い、WSL2 のローカル MCP を登録します。診断は `uv run gpu-ai doctor` で実行できます。

実 GPU の起動はユーザーが明示的に依頼したジョブに限ります。許可された入力をワークスペースからの相対パスで指定し、失敗したジョブを自動で繰り返さないでください。

MCP の推論ツールが返すジョブ ID を `get_job_status` でポーリングし、`get_job_result` の成果物パスと `cleanup_status` を報告します。停止が未確認の場合は、同じワークスペースで `uv run gpu-ai reconcile` を実行して確認します。ユーザー所有の他のセッションを停止しないでください。
