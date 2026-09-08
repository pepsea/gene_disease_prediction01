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

## パイプライン

| 段階 | 内容 | スクリプト |
|---|---|---|
| — | 疾患名と遺伝子リストを受け取り、以下を自動で実行 | `rank.py` |
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
```

静かに壊れやすい箇所を押さえてある：プロンプト構築、PMI による頻度補正、
棄却（abstain）判定、そして段階 2 で遺伝子が黙って落ちる 2 つの経路。

## ライセンス上の注意

モデル重み・学習コーパス・データベースはそれぞれ別の条件を持つ。
商用利用の予定があるなら、ファインチューニングに投資する前に確認すること。
MeditronFO のコーパスは研究用途で、モデルは臨床利用向けに承認されていない。
現状は `.claude/skills/gene-disease-ranking/references/data-sources.md` にまとめてあるが、
条件は変わるので必ず一次情報で確認する。
