# WSL2 Ubuntu で Colab CLI を使う

WSL2 の Ubuntu シェルでインストールします。Windows 側の Python や CLI と区別するため、`command -v colab` で Ubuntu 側の実行ファイルを確認します。コマンドが見つからない場合は `uv tool update-shell` を実行してシェルを開き直します。

```sh
uv tool install google-colab-cli
colab version
colab --help
colab new --help
colab exec --help
```

アダプターは公式 CLI 0.8.0 のヘルプと実装に基づきます。更新時は `colab <command> --help` で各コマンドの引数を確認してください。

## 初回 OAuth ログイン

一時的な設定ファイルを指定し、既存セッションのローカル設定と分けて認証します。

```sh
auth_dir=$(mktemp -d)
colab --auth=oauth2 --config "$auth_dir/sessions.json" sessions
uv run python -m gpu_ai.doctor
```

最初のコマンドが表示する URL をブラウザーで開き、Google へのログインと同意を済ませ、認可コードを Ubuntu のプロンプトに貼り付けます。認可コードや認証情報は共有しないでください。WSL2 の Ubuntu 側で OS キーリングが利用でき、ロック解除されていることを確認してください。認証が次回コマンドに引き継がれない場合は Ubuntu 側の Secret Service と D-Bus セッションを確認します。

診断は CLI 未導入、未認証、認証済み、接続などの診断失敗を区別します。読み取りのセッション一覧で認証を確かめます。GPU の割当資格は Colab のアカウントプランを確認し、下記のスモークで割当を試してください。

## T4 スモーク（手動）

GPU 枠やコンピューティングユニットを消費する可能性があります。実行する場合は、以下を WSL2 の Ubuntu で実行します。専用の設定と名前を作成し、終了時にそのセッションを停止します。

```bash
(
    set -eu
    smoke_dir=$(mktemp -d)
    smoke_name="gpu-ai-smoke-$(cat /proc/sys/kernel/random/uuid)"
    config="$smoke_dir/sessions.json"
    printf 'Session: %s\nConfig: %s\n' "$smoke_name" "$config"
    trap 'colab --auth=oauth2 --config "$config" stop -s "$smoke_name"' EXIT
    printf '%s\n' 'import subprocess' 'subprocess.run(["nvidia-smi", "-L"], check=True)' > "$smoke_dir/check.py"
    colab --auth=oauth2 --config "$config" new -s "$smoke_name" --gpu T4
    colab --auth=oauth2 --config "$config" status -s "$smoke_name"
    colab --auth=oauth2 --config "$config" exec -s "$smoke_name" -f "$smoke_dir/check.py"
)
```

`nvidia-smi` の出力に T4 が表示され、停止時に `Session terminated.` が表示されることを確認します。停止を確認できない場合は、表示された設定ファイルとセッション名で `stop -s` を再試行し、Colab のランタイム管理画面で該当ランタイムの残存を確認します。無料枠の GPU 種別、容量、利用時間は変動し、T4 の割当は保証されません。自動テストは模擬 subprocess を使います。

## ワーカーの接続

`ColabCLIBackend` はアプリケーションが管理する `Operation → WorkerScript` を受け取ります。ワーカーは `/content` を作業ディレクトリとし、`gpu-ai-request.json` と相対パスで展開された入力を読み、宣言した成果物の相対パスに書き込みます。スクリプトと成果物一覧はアプリケーション側で登録し、タスクのパラメータは JSON のデータとして渡されます。入力が空の場合も `upload(session_id, {})` がリクエストを転送します。アダプターは成果物を回収し、取得したバイト列からハッシュを計算します。

ジョブの状態ファイルはワークスペースの `.colab-cli` に作成されます。未確認の停止を調べる際は、該当ジョブ専用の `sessions.json` とセッション名を使います。このファイルにはプロキシ資格情報が含まれるため共有しないでください。公式 CLI 自身は Ubuntu ユーザーの `~/.config/colab-cli` にログと履歴を保存するため、ワーカーのコードや出力にも秘密情報を含めないでください。

- [Google 公式 Colab CLI](https://github.com/googlecolab/google-colab-cli)
- [Colab FAQ と利用制限](https://research.google.com/colaboratory/faq.html)
