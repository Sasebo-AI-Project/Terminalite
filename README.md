# Terminalite

自然言語の依頼（日本語・英語）から bash のコマンドを 1 本提案する、CPU だけで動く小さなモデル。

```console
$ tl "logs フォルダで一番大きいファイルを 5 つ表示して"
ls -l /testbed/logs | awk 'NR>1 {print $5, $9}' | sort -rn | head -n 5

$ tl -n 3 "compress the reports directory into reports.tar.gz"
  -3.501  tar -czf reports.tar.gz ./reports
  -3.614  tar -czf reports.tar.gz reports
  -3.787  tar -czf reports.tar.gz -C reports reports
```

- モデルは非埋め込みパラメータ 5.78M の GRU（2 本）。GPU も外部 API も使わない
- 1 回の依頼は CPU で 0.1〜0.2 秒、起動は 0.3 秒ほど（AMD Ryzen 5 5600 で測定）
- コマンドは**表示するだけで実行しない**

## インストール

Python 3.10 以上が必要。

```console
pip install git+https://github.com/OWNER/terminalite
```

[uv](https://docs.astral.sh/uv/) を使う場合は `uv tool install git+https://github.com/OWNER/terminalite`。

初回の実行で重み（約 79 MB）を GitHub Releases からダウンロードし、ハッシュを照合して次の場所に保存する。

- Windows：`%LOCALAPPDATA%\terminalite`
- macOS・Linux：`~/.cache/terminalite`（`XDG_CACHE_HOME` があればその下）

保存先は環境変数 `TERMINALITE_HOME` で変えられる。

## 使い方

```console
tl "依頼"          # コマンドを 1 本表示
tl -n 3 "依頼"     # 候補を 3 本、スコア付きで表示（大きいほど自信がある）
tl                 # 対話モード：1 行に 1 つ依頼を入力し、Ctrl+C で終了
tl --timing "依頼" # 起動と推論にかかった時間を表示
```

`tl` の代わりに `terminalite` や `python -m terminalite` でも動く。

## 精度と注意

正答率（実行結果と LLM の判定による機能的同値性）は、評価セットによって 25〜60% 程度である。

| 評価セット | 正答率 |
|---|---|
| 合成テスト（英語 / 日本語） | 56.9% / 58.4% |
| NL2SH テスト（人手で作成） | 38.0% |
| InterCode-Bash | 25.4% |

- **実行する前に必ず中身を確認すること。** 特に削除や上書きをするコマンドに注意。確認のオプション（`rm -i` など）を落とすことがある
- 学習時の作業ディレクトリが `/testbed` だったため、依頼にパスがないと `/testbed/...` を補うことが多い。自分の環境のパスに読み替えるか、依頼にパスをはっきり書く
- 対象は Linux の bash。よく使われるコマンドラインツール（ffmpeg、sqlite3、jq など）と `for`/`while`/`if` も少し扱えるが、データの少ないツールは弱い

## 仕組み

GRU のエンコーダ・デコーダ（注意機構とコピー機構付き）でビームサーチして候補を 8 本出し、別に学習した逆向きのモデル p(依頼 | コマンド) のスコアを足して選び直す（noisy channel）。学習データは LLM が生成し、Docker のサンドボックスで実行または構文検査して確かめたものである。詳しくは論文を参照。

- 論文：（Sasebo AI Project のサイトの URL）

## ライセンス

[Apache License 2.0](LICENSE)。重みの学習に使ったデータの出所は [NOTICE](NOTICE) を参照。

---

## English

Terminalite suggests one bash command for a natural-language request in English or Japanese. It runs on CPU
only (two GRU models, 5.78M non-embedding parameters each; about 0.1–0.2 s per request) and never executes
the command. Install with `pip install git+https://github.com/OWNER/terminalite`, then run `tl "your request"`.
The weights (about 79 MB) are downloaded on first use. Always read a suggested command before running it.
