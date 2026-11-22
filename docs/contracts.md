# 共通契約

`TaskRequest` はワーカーに渡す操作、ワークスペースからの相対パス、操作固有のパラメータ、GPU、ジョブ ID を保持します。GPU を省略すると T4、ジョブ ID を省略すると UUID が割り当てられます。

```json
{
  "operation": "transcribe_audio",
  "inputs": ["audio/sample.wav"],
  "parameters": {"language": "ja"},
  "gpu": "T4",
  "job_id": "e1049acb-26dd-44bb-9d01-adc38cb5cb4a"
}
```

`JobResult` は状態、成果物、計測値、エラー、GPU の停止確認状態を返します。停止前の `cleanup_status` は `unknown` です。

```json
{
  "schema_version": "1",
  "job_id": "e1049acb-26dd-44bb-9d01-adc38cb5cb4a",
  "status": "completed",
  "artifacts": [
    {
      "path": "transcript.json",
      "mime_type": "application/json",
      "sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
      "size": 2
    }
  ],
  "metrics": {},
  "error": null,
  "cleanup_status": "confirmed"
}
```

## 設定

`gpu_ai.config.load_settings(Path("settings.toml"))` で TOML を読み込みます。対応する `GPU_AI_` 環境変数はファイルの値を上書きします。たとえば `GPU_AI_MAX_INPUT_BYTES` で入力容量の上限を変更できます。相対ディレクトリは実行時の作業ディレクトリを基準に扱います。

```toml
workspace = "workspace"
results_dir = "results"
gpu = "T4"
max_concurrent_jobs = 1
max_input_bytes = 104857600
max_runtime_seconds = 1800
```

共通スキーマは相対パスの形式を検証します。実ファイルの存在、ワークスペース内に収まること、シンボリックリンク、サイズと MIME の検証はファイルを扱う層の責務です。
