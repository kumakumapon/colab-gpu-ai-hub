# 動画分析

`analyze_video` は MP4 を1本受け取り、音声、代表フレーム、相対深度、任意の対象マスクを再生時刻でまとめます。ワーカーは1台の Colab ランタイムで別々の子プロセスとして順次実行し、各プロセスの終了時にモデル用メモリを解放します。ローカルと Colab の両方に FFmpeg と FFprobe が必要です。

```sh
uv run gpu-ai run analyze_video --input clips/sample.mp4 --parameters '{"frame_interval_seconds":5,"max_frames":12}'
```

MCP では `analyze_video` に `{"request":{"inputs":["clips/sample.mp4"],"frame_interval_seconds":5}}` を渡し、返された ID で `get_job_status` と `get_job_result` を呼びます。GPU を使うスモークは、許可された短い動画で実行し、成果物と `cleanup_status=confirmed` を確認してから `colab sessions` で残存を確認します。

- 動画長は最大600秒です。設定の `max_media_duration_seconds` がそれより小さい場合はその値が上限になります。既定は300秒です。入力容量とジョブの実行時間にも共通の上限が適用されます。
- 代表フレームの間隔は1〜60秒、上限は1〜24枚です。既定は5秒間隔で最大12枚です。上限に達した後の映像には深度の推定を行いません。
- 音声トラックがある場合に Whisper で文字起こしします。音声がなければ空の SRT と空のセグメント配列を保存します。
- `objects` に [SAM 2.1 の点または矩形](sam2.md) を明示した場合、冒頭の最大7秒か48フレームのうち先に達する範囲を追跡します。点・矩形は映像の表示フレーム座標です。追跡には SAM 2.1 の寸法・画素数制限も適用されます。
- 各段階の状態と処理秒数を記録します。失敗した段階があっても回収できた成果物を保存し、ジョブの計測値に `partial=true` を返します。実行時間上限やGPU停止失敗にはジョブランナーの共通処理が適用されます。

## 成果物と時刻

| 成果物 | 内容 |
|---|---|
| `transcript.json` / `transcript.srt` | 音声の推定文字起こしと区間 |
| `timeline.json` | 動画長、平均フレームレート、音声有無、時刻順イベント、各段階の結果 |
| `evidence.zip` | サンプルフレーム、深度配列・PNG・メタデータ、任意の対象マスクと追跡結果。イベントから ZIP 内パスで参照できます。 |
| `report.md` | Qwen に渡した短い根拠抜粋に基づくモデル生成の要約。生成失敗時は処理結果への案内を保存します。 |

FFmpeg の `showinfo` で間引き直後の元フレーム PTS と時刻基準を取得します。動画の再生開始を0秒とし、音声の出力も同じ開始時刻に合わせます。可変フレームレートのテストでは、サンプルと追跡結果の時刻を FFprobe が返す元動画の各フレーム時刻と比較します。

`source_media` は抽出フレーム、`model_transcription` は文字起こし、`model_estimate` は相対深度と追跡マスクを表します。生成要約には誤りがあり得ます。対象マスクは推定結果で、深度には物理距離の単位がありません。GPU 費用は Colab アカウント側の利用状況で確認できます。ジョブの計測値には確保したランタイム数と実行秒数が含まれます。

- [FFmpeg フィルタの公式仕様](https://ffmpeg.org/ffmpeg-filters.html)
- [FFprobe の公式仕様](https://ffmpeg.org/ffprobe.html)

## T4 実機確認

2026-11-23（JST）に、WSL2 のローカル stdio MCP から2秒の合成 MP4 を送信し、Colab T4 1台で Whisper、深度推定、SAM 2.1、Qwen 要約の全段階が完了しました。ジョブ ID は `594f82f2-dbb1-4f2a-8cff-0f740966b3e3` です。

- MCP で成果物5件のパスを取得し、各ファイルの SHA-256 がマニフェストと一致することを確認しました。
- 自動停止は `cleanup_status=confirmed` で、停止後の Colab ランタイム一覧は0件でした。
- ジョブの処理時間は約151秒、試験全体は約161秒でした。
- 入力音声は合成トーンで、文字起こし段階は完了し、発話区間0件の空の SRT が保存されました。
