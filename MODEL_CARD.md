# Terminalite v0.1.0 モデルカード

## 概要

自然言語の依頼（日本語・英語）から bash のコマンドを 1 本提案するモデル。GPU も外部 API も使わず、CPU だけで動く。

- 開発：Sasebo AI Project
- 公開：2026 年 10 月、v0.1.0
- ライセンス：Apache License 2.0（[NOTICE](NOTICE) も参照）
- 論文：（Sasebo AI Project のサイトの URL）

## 構成

2 つのモデルを組み合わせて使う。

| ファイル | 役割 | 学習データ |
|---|---|---|
| `gru-copy-v1-13.npz` | 順方向 p(コマンド \| 依頼) | 全データ |
| `gru-rev-v1-12.npz` | 逆向き p(依頼 \| コマンド) | v12 の第 2 ラウンドまで |

- どちらも GRU のエンコーダ・デコーダで、注意機構とコピー機構（pointer-generator）を持つ。エンコーダは 2 層の双方向 GRU（各方向 256 次元）、デコーダは 2 層の GRU（512 次元）
- パラメータは 1 モデルあたり 9.88M、埋め込みを除くと 5.78M。重みは 2 モデル合わせて fp32 で約 79 MB
- 語彙は依頼とコマンドで共有する 16k のバイトレベル BPE（`bpe16k.json`）
- 順方向モデルのビームサーチ（幅 8）で候補を出し、逆向きモデルのスコアを足して選び直す（noisy channel、β = 4.0）
- 事前学習済みのモデルは使っていない。一から学習した

## 用途

- 想定する使い方：端末で、やりたいことからコマンドを思い出す補助。提案されたコマンドは、人が読んで確かめてから実行する
- 想定しない使い方：提案されたコマンドを確かめずに自動で実行すること。エージェントの行動として直接使うこと。Linux の bash 以外のシェル（PowerShell、cmd、zsh 固有の構文など）

## 学習データ

英語と日本語の（依頼, コマンド）の組。78,471 種類のコマンドに対して 788,621 組ある。

- 依頼文と大半のコマンドは LLM（Claude Haiku 5.5、Qwen3.8 Max）が生成した
- 一部のコマンドは Nemotron-Terminal-Corpus（NVIDIA、CC BY 4.0）の端末エージェントの行動ログから抜き出し、依頼文を LLM で付けた
- すべてのコマンドを Docker のサンドボックス（Ubuntu 24.04、作業ディレクトリ `/testbed`）で実行するか、構文を検査した。依頼文とコマンドが合っているかは DeepSeek V4.1 Flash で照合した
- 学習データは公開しない

## 評価

正答は、正解との文字列の一致、サンドボックスでの実行結果の一致、LLM（DeepSeek V4.1 Flash）による機能的同値の判定のどれかで数えた（judge）。Terminalite の数値は 2 シードの平均で、Qwen は greedy 復号。

| 評価セット | 件数 | Terminalite | Qwen2.5-Coder-0.5B + LoRA |
|---|---:|---:|---:|
| 合成テスト（英語） | 547 | 56.9% | 54.1% |
| 合成テスト（日本語） | 547 | 58.4% | 54.1% |
| NL2SH テスト（人手で作成） | 300 | 38.0% | 24.3% |
| InterCode-Bash | 193 | 25.4% | 15.0% |
| 外部ツールと制御構文 | 372 | 46.5% | — |

- 合成テストは学習データと同じ手順で作ったもので、コマンドは学習に出てこないが分布は近い。実際の使い心地の目安には、人手で作った NL2SH と InterCode の方が近い
- 判定に 2 つ目の LLM（Claude Haiku 5.5）も加え、両方が同値と認めたものだけを数えると、それぞれ 1〜4 ポイント下がる
- 推論は 1 依頼あたり中央値 123 ms、90 パーセンタイル 215 ms、起動は 0.24 秒（AMD Ryzen 5 5600、4 スレッド、numpy）

## 限界と注意

- 正答率は評価セットによって 25〜60% 程度である。提案の多くは誤っていることを前提に使うこと
- 誤りで最も多いのは、オプションと引数の選び方である。使うユーティリティは合っていることが多い
- 削除や上書きをするコマンドも出し、確認のオプション（`rm -i` など）を落とすことがある
- 学習時の作業ディレクトリが `/testbed` だったため、依頼にパスがないと `/testbed/...` を補うことが多い
- 利用者の環境（ファイル一覧、インストール済みのツール）は知らない
- データの少ないツールは弱い
- 学習データは LLM が書いたものなので、人の実際の頼み方とはずれがある。評価も LLM の判定に頼っている

## 重みのハッシュ（SHA-256）

```
8a9776e610da4f3aec3f4c5023d4021fb6f156ebb11eaa237358daa565d2d69d  gru-copy-v1-13.npz
be20122d610ea57f02d4d31ed085bd05801c78823dde09b69013c15f5f2140c5  gru-rev-v1-12.npz
```

重みは [Releases](https://github.com/Sasebo-AI-Project/Terminalite/releases/tag/v0.1.0) にある。`tl` コマンドは初回の実行でダウンロードし、このハッシュで照合する。

---

## English summary

Terminalite suggests one bash command for an English or Japanese request, on CPU only. It combines two GRU
encoder-decoders with attention and copying (5.78M non-embedding parameters each, trained from scratch): a forward
model generates 8 beam candidates and a reverse model p(request | command) reranks them (noisy channel).

- **Training data:** 788,621 English/Japanese (request, command) pairs over 78,471 distinct commands, written by
  LLMs or mined from Nemotron-Terminal-Corpus (CC BY 4.0), executed or syntax-checked in a Docker sandbox and
  cross-checked by a second LLM. The data are not released.
- **Accuracy** (execution or LLM-judged functional equivalence, mean of 2 seeds): 56.9% / 58.4% on held-out
  synthetic English / Japanese tests, 38.0% on NL2SH, 25.4% on InterCode-Bash. A LoRA-tuned Qwen2.5-Coder-0.5B
  scores 54.1% / 54.1% / 24.3% / 15.0%.
- **Speed:** median 123 ms per request on a Ryzen 5 5600 (4 threads).
- **Limitations:** 40–75% of suggestions are wrong, depending on the test set. The model may drop safety flags such as `rm -i`, and it
  tends to assume the working directory `/testbed`. Always read a command before running it.
- **License:** Apache 2.0.
