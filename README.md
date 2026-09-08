# gene_disease_prediction01

ローカルの生物医学 LLM を使って、疾患に関連する遺伝子候補をランキングするためのリポジトリ。

中心にあるのは Claude Code スキル `gene-disease-ranking`（`.claude/skills/` に配置）。
このスキルは、**LLM に遺伝子記号を自由生成させない**という一点を守るための設計と
スクリプト群をまとめたもの。

## なぜ「生成させない」のか

`GPR52` を答えさせたいのに `GPR56` が返る、という失敗は知識不足ではなく
デコード（文字生成）の問題。どのトークナイザも `GPR52` を 1 トークンでは持たず、
`G` / `PR` / `5` / `2` のように分割する。`GPR` まで出力した時点で、次の文字は
学習コーパスでの出現頻度に強く引っ張られる。PubMed で頻出の `GPR56` が勝つ。
プロンプトに候補リストを入れても、それは文脈であって制約ではない。

対策はプロンプト調整ではなく構成の変更：候補記号を「採点する」か、
`A`〜`E` の 1 トークンのラベルを出させてコード側で記号に戻すか、どちらかにする。

## 使い方（疾患名と遺伝子リストを変数で渡す）

```python
from rank import GeneRanker

ranker = GeneRanker("EPFLiGHT/Gemma-3-27B-MeditronFO")
result = ranker.rank(
    disease="Cystic fibrosis",                       # 疾患名（変数）
    genes=["CFTR", "HBB", "GPR52", "GPR56", "APOE"], # 遺伝子リスト（変数）
)
print(result["call"], result["margin"], result["rank_stability"])
```

遺伝子リストからプロンプトへの変換は `scripts/prompts.py` が自動で行う。
ここで注意すべきは、**リストを 1 つのプロンプトにまとめて「選ばせる」のではない**
という点。リストは (1) 遺伝子ごとの強制継続スコアリング（段階 1）と、
(2) 順序をローテーションした A〜E の選択肢ブロック（段階 2）に展開される。
どちらもモデルの出力側に遺伝子記号が現れないので、`GPR52` が `GPR56` に
滑る余地がない。

生成されるプロンプトは、モデルも torch も無しで確認できる：

```bash
cd .claude/skills/gene-disease-ranking
python scripts/rank.py --disease "Cystic fibrosis" --genes CFTR HBB GPR52 --dry-run
```

ファイル入力も従来どおり使え、変数指定と混ぜられる：

```bash
python scripts/rank.py --model <model-id> \
  --diseases-file examples/diseases.txt \
  --genes-file examples/candidates.txt --out result.jsonl
```

複数疾患を同じ遺伝子リストで回すときは `GeneRanker` を 1 つ使い回す。
疾患に依存しない中和項（neutral）を 1 回だけ計算して再利用するため。

## ノートブック

用途の違う 2 つがあります。

| ファイル | 用途 |
|---|---|
| [`notebooks/score_genes.ipynb`](notebooks/score_genes.ipynb) | **使うためのもの。** 疾患名と遺伝子リストを入れて点数を出す |
| [`notebooks/gene_disease_ranking.ipynb`](notebooks/gene_disease_ranking.ipynb) | 仕組みの説明、プロンプトの確認、評価と対照実験 |

```bash
pip install jupyterlab
jupyter lab notebooks/score_genes.ipynb
```

### score_genes.ipynb（スコアリング）

編集するのは「1. 入力」セルだけ（`DISEASE` / `GENES` / `BACKEND` / `MODEL`）。
上から実行すると、全遺伝子の点数表・判定・CSV 保存まで出ます。
`score(疾患名, 遺伝子リスト)` として何度でも呼べます。

### gene_disease_ranking.ipynb（解説と評価）

`MODEL = None` のままでも 3 章まで動き、**実際にモデルへ送られるプロンプトを目で確認できます**。
torch も GPU も要りません。まずここを見て、遺伝子記号がモデルの出力側に無いことを
確かめてから GPU のある環境へ持っていくのが安全です。
対照実験（ランダム／頻度のみ／疾患シャッフル）も 8 章にあります。

## バックエンド ─ ローカルの Ollama を使う場合

**先に結論。Ollama では段階 1（PMI）が動きません。**

このパイプラインはモデルに 2 つの違うことを頼みます。

| | 何を頼むか | どこで使うか | 難易度 |
|---|---|---|---|
| A | **こちらが渡した文字列**の確率 | 段階 1（PMI） | 対応していない実行環境がある |
| B | 次の 1 文字（A〜E）の確率 | 段階 2 | だいたいどこでもできる |

Ollama の `logprobs` / `top_logprobs`（v0.12.11 以降）は **モデルが生成した**
トークンの確率です。OpenAI の `echo` に相当する機能も採点用エンドポイントも無いため、
こちらが渡した `GPR52` という文字列の確率を読み出せません。

| バックエンド | 段階 1 | 段階 2 | 備考 |
|---|---|---|---|
| `transformers` | ○ | ○ | 基準となる実装 |
| `llamacpp` | ○ | ○ | GGUF。**Ollama が持っている重みをそのまま読める** |
| `ollama` | **×** | ○ | 渡した記号を採点できない |

まず実機で確認してください（推測しないこと）：

```bash
python .claude/skills/gene-disease-ranking/scripts/check_backend.py \
  --backend ollama --model gemma3:27b
```

### 推奨：同じ重みを llama.cpp で読む

Ollama は GGUF を `~/.ollama/models/blobs/` に置いています。llama.cpp から同じ
ファイルを指せば、**再ダウンロードなしで**段階 1 も動きます。

```bash
pip install llama-cpp-python
python .claude/skills/gene-disease-ranking/scripts/rank.py \
  --backend llamacpp --model gemma3:27b --disease "Cystic fibrosis" --genes CFTR HBB GPR52
```

`ollama` のまま進めることもできます。その場合は段階 2 だけを全候補に回します。
遺伝子記号を生成しない点は保たれますが、**出現頻度の偏りを打ち消すものが無くなります**。
警告が出るので、「頻度のみ」対照で必ず確認してください。

### Docker で動かしている場合

**接続**：`-p 11434:11434` で公開されていれば `http://localhost:11434` で届きます。
このコードは localhost → `host.docker.internal` → `172.17.0.1` の順に自動で探します
（呼び出す側もコンテナ内だと `localhost` は自分自身を指すため）。

**重みの読み出し**：ここが Docker 特有の問題です。よく推奨される名前付きボリューム
（`-v ollama:/root/.ollama`）だと、GGUF はホストから素直には見えません
（Linux では root 権限が必要、Docker Desktop では VM の中で**ホストからは見えない**）。
つまり上の「llama.cpp で同じ重みを読む」が成立しません。

| | 方法 | 代償 |
|---|---|---|
| 1 | `-v ~/.ollama:/root/.ollama` でバインドマウントし直す | 再起動のみ。**複製なし**。基本はこちら |
| 2 | `docker cp` でコピー | ディスク二重使用（27B の Q4 で約 17GB） |

`check_backend.py` がコンテナ名と digest を調べて、そのまま実行できる
`docker cp` のコマンドを表示します。

## パイプライン

| 段階 | 内容 | スクリプト |
|---|---|---|
| — | 疾患名と遺伝子リストを受け取り、以下を自動で実行 | `rank.py` |
| — | 実行環境が何をできるか確認（最初にこれ） | `check_backend.py` |
| 0 | HGNC で記号を正規化（別名 → 承認記号、非遺伝子は除去） | `normalize_genes.py` |
| 1 | PMI スコアリング（候補を強制継続で採点し、疾患なしの対照を引く） | `score_pmi.py` |
| 2 | 上位 10〜20 件を A〜E ラベルの選択式で再ランキング | `score_labels.py` |
| 3 | 文献・遺伝学的エビデンスで裏取り（Open Targets / PubTator3） | — |

評価は `evaluate.py`。ランダム・頻度のみ・疾患シャッフルの 3 対照を必ず併記する。
データベース単体（Open Targets）の対照も別途取り、LLM を足して改善しないなら
その事実をそのまま報告する。

詳細は [`.claude/skills/gene-disease-ranking/SKILL.md`](.claude/skills/gene-disease-ranking/SKILL.md) と
`references/` を参照。

## セットアップ

```bash
pip install -r requirements.txt
```

`evaluate.py` と `normalize_genes.py` は numpy だけで動く（torch 不要）。
モデルを使う `score_pmi.py` / `score_labels.py` のみ torch + transformers が必要。

## 最初に実行するもの

```bash
cd .claude/skills/gene-disease-ranking
python scripts/check_tokenizer.py --model <model-id> --genes GPR52 GPR56 HBB HBA1
```

そのモデルで遺伝子記号がどう分割されるか、`A`〜`E` が 1 トークンかを確認する。
数秒で終わり、静かに壊れる失敗を一群まとめて防げる。

## テスト

モデルも GPU も使わない回帰テスト（numpy のみ必要）：

```bash
python .claude/skills/gene-disease-ranking/tests/test_offline.py
python .claude/skills/gene-disease-ranking/tests/test_backends.py
```

静かに壊れやすい箇所を押さえてある：プロンプト構築、PMI による頻度補正、
棄却（abstain）判定、そして段階 2 で遺伝子が黙って落ちる 2 つの経路。

`test_backends.py` は偽の Ollama サーバを実際のソケットで立てて、HTTP の
リクエスト形状・応答の解釈・ラベル欠落・Docker 経路の診断まで通します。

## ライセンス上の注意

モデル重み・学習コーパス・データベースはそれぞれ別の条件を持つ。
商用利用の予定があるなら、ファインチューニングに投資する前に確認すること。
MeditronFO のコーパスは研究用途で、モデルは臨床利用向けに承認されていない。
現状は `.claude/skills/gene-disease-ranking/references/data-sources.md` にまとめてあるが、
条件は変わるので必ず一次情報で確認する。
