# ジョブと GPU の診断

同じワークスペース設定とジョブ ID を使って、状態、処理の結果、成果物、GPU の停止を確認します。

```sh
uv run gpu-ai doctor
uv run gpu-ai status JOB_ID
uv run gpu-ai results JOB_ID
uv run gpu-ai reconcile
```

| 症状 | 確認と対応 |
|---|---|
| CLI 未導入、未認証 | `gpu-ai doctor` で導入状況と OAuth を確認します。WSL2 では実行シェルが Linux キーリングに接続できる必要があります。 [WSL2 と OAuth](setup-wsl2.md) |
| `rate_limited` / `allocation_failed` | Colab の利用状況とランタイム一覧を確認します。空きや利用制限は変動します。失敗したジョブの自動再実行は避けます。 [Colab FAQ](https://research.google.com/colaboratory/faq.html#resource-limits) |
| `input_validation_failed` | ファイルが許可されたルート内の通常ファイルか、形式、サイズ、動画長、FFmpeg・FFprobe を確認します。 |
| `upload:command_failed` / `download:invalid_response` | 台帳の状態履歴で転送段階を確認します。ハッシュや ZIP 検証で失敗した成果物は公開されません。 |
| `worker_dependencies` / `incompatible_cuda` | ワーカー資料で依存バージョンと必要な CUDA ライブラリを確認します。PaddleOCR の Linux x86_64・Python 3.13 用 GPU 環境には初回の wheel 取得時間がかかります。 |
| `wall_clock_timeout` | 台帳で止まった段階を確認します。時間上限は入力検証から成果物保存までで、停止には別の時間枠が適用されます。 |
| `cleanup_pending` | `reconcile` で保留を確認し、同じ設定で `reconcile --retry` を実行します。これは台帳に記録された所有セッションの停止だけを試みます。 |

- `confirmed` は停止を確認できた状態です。`failed` は停止操作が失敗し、`unknown` は応答やセッションの確認が取れない状態です。
- `workspace/.gpu-ai-jobs/<job-id>.json` の `history`、`session_id`、`cleanup_status` を確認します。成果物は `results_dir/<job-id>/gpu-ai-manifest.json` のサイズと SHA-256 で確認できます。
- 既に割当が成功したセッションを調査するときは、該当ジョブのセッション名、専用の CLI 状態ファイルを指定して `colab --auth=oauth2 --config "$WORKSPACE/.colab-cli/$SESSION_NAME/sessions.json" status -s "$SESSION_NAME"` を使います。Colab 側の一覧は `colab sessions` で確認します。専用設定ファイルには資格情報が含まれるため、その内容は共有しないでください。
- 割当直後の中断で CLI がセッション情報を保存できなかった場合は、自動処理で所有を確定できません。Colab のランタイム管理画面で該当ジョブの残存を確認してください。別のセッションは操作しないでください。
