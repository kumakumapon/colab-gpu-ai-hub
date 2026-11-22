# 統一 CLI とローカル MCP

WSL2 Ubuntu のリポジトリルートで `uv sync --locked` を実行し、[Colab CLI と OAuth](setup-wsl2.md) を準備します。音声と画像の事前検証には FFmpeg と FFprobe が必要です。ユーザーが転送を許可した入力だけを `workspace/` に置きます。

## CLI

```sh
uv run gpu-ai doctor
uv run gpu-ai run transcribe_audio --input audio/sample.wav --parameters '{"language":"ja"}'
uv run gpu-ai run estimate_depth --input frames/sample.png
uv run gpu-ai run llm_infer --prompt-file prompt.txt
uv run gpu-ai run extract_text --input docs/sample.pdf --parameters '{"language":"ja","markdown":true}'
uv run gpu-ai run generate_image --prompt "A small cabin at sunrise" --parameters '{"seed":42,"steps":15}'
uv run gpu-ai status JOB_ID
uv run gpu-ai results JOB_ID
uv run gpu-ai cancel JOB_ID
uv run gpu-ai reconcile
uv run gpu-ai reconcile --retry
```

`run` は処理完了まで待ち、共通の `JobResult` を JSON で返します。ジョブ ID を事前に決めて別の端末から追跡するには `--job-id UUID` を指定します。同じ設定とジョブ ID で実行し直すと台帳にある結果を返します。入力パスはワークスペースからの相対パスです。`--input` は繰り返し指定できます。`--gpu` を省略すると設定の GPU（既定 T4）を使います。

結果は `results/<job-id>/` に保存されます。`status` で成果物のローカルパスと停止状態を確認できます。`cancel` は処理段階の境界で適用されます。停止が未確認のジョブには、同じワークスペースで `reconcile --retry` を実行します。

## Claude Code と Codex に登録する

次をリポジトリルートで実行します。`--directory` には登録時の絶対ディレクトリを渡します。`uv_bin` も WSL 側の絶対パスです。

```sh
project_dir="$(pwd -P)"
uv_bin="$(command -v uv)"
claude mcp add --scope local --transport stdio gpu-ai -- "$uv_bin" --directory "$project_dir" run --locked python -m gpu_ai.mcp_server
codex mcp add gpu-ai -- "$uv_bin" --directory "$project_dir" run --locked python -m gpu_ai.mcp_server
```

CLI と MCP が別の作業ディレクトリから起動する場合は、同じ入力ルートと成果物ディレクトリを絶対パスで設定します。TOML を使う場合は、サーバー起動コマンドの末尾に `--config /path/to/settings.toml` を指定し、CLI には `gpu-ai --config /path/to/settings.toml` を指定します。MCP ホストにも WSL 側の Colab CLI の PATH とキーリングを使える環境を引き継いでください。

Claude Code は `claude mcp list` とセッション内の `/mcp`、Codex は `codex mcp list` とセッション内の `/mcp` で接続を確認します。登録後、ホストで `transcribe_audio`、`estimate_depth`、`llm_infer`、`extract_text`、`generate_image`、`get_job_status`、`get_job_result`、`cancel_job` の8ツールが見えるか確認してください。ツール一覧の取得で GPU は起動しません。Python MCP クライアントによる stdio 検証を自動テストに含めています。両ホストの対話セッションでの表示は未検証です。

- [Claude Code の公式 MCP 手順](https://code.claude.com/docs/en/mcp)
- [Codex の公式 MCP 手順](https://developers.openai.com/codex/mcp)

## MCP でジョブを実行する

1. ユーザーが明示的に実行を依頼したとき、対応ツールを1回呼びます。例として `transcribe_audio` の引数は `{"request":{"inputs":["audio/sample.wav"],"language":"ja"}}` です。テキスト生成では `{"request":{"prompt":"日本語で短く挨拶してください。","max_new_tokens":24}}` を渡せます。
2. 返されたジョブ ID を使い、`get_job_status` に `{"job_id":"<UUID>"}` を渡します。数秒間隔で確認し、`completed`、`failed`、`cancelled`、`cleanup_pending` のいずれかになるまで待ちます。
3. 完了時は `get_job_result` で `artifact_paths` と `cleanup_status` を確認します。応答には成果物ファイルのパスが含まれます。
4. 中止する場合は `cancel_job` に同じ ID を渡し、停止状態を再確認します。接続が途切れた場合も、同じワークスペースの台帳を CLI から照会できます。解放が未確認なら `gpu-ai reconcile --retry` で回復を試みます。

ユーザーが許可した入力ルートに置いたファイルだけを使います。ジョブの失敗や接続断を理由に実 GPU を自動で繰り返し起動しないでください。結果を報告するときはジョブ ID、成果物パス、停止状態を含めます。
