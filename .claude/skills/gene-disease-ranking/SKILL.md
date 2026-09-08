---
name: gene-disease-ranking
description: Rank or select disease-associated genes with a local biomedical LLM (Meditron, MeditronFO, Gemma, OLMo, Apertus) without letting the model hallucinate gene symbols. Use this whenever the user wants to score, rank, prioritize, shortlist, or pick candidate genes for a disease or phenotype using an LLM — and especially if they report the model confusing similar symbols (GPR52 vs GPR56, SLC6A4 vs SLC6A3, HLA-DRB1 vs HLA-DRB5), inventing genes that do not exist, or returning the same famous genes (TP53, EGFR, TNF) for every disease. Also use for building the evaluation harness for such a system, for choosing which open medical LLM to use, and for wiring in Open Targets / HGNC / PubTator as candidate sources. Trigger even when the user only says "the model keeps picking the wrong gene" or "gene prioritization pipeline".
---

# Gene-Disease Ranking with Local LLMs

Build systems that rank candidate genes by association with a disease, using a
local open-weight LLM, without the model ever emitting a gene symbol as free text.

## The failure this skill exists to prevent

A user asks a model for the gene linked to a disease. The answer should be
`GPR52`. The model returns `GPR56`.

This is not a knowledge failure. It is a decoding failure. No tokenizer holds
`GPR52` as one token — it is split into pieces like `G` / `PR` / `5` / `2`. Once
the model has emitted `GPR`, the next token is chosen largely by how often each
continuation appeared in the training corpus. `GPR56` is more frequent in PubMed,
so it wins. The candidate list in the prompt is context, not a constraint.

**The fix is architectural: never let the model generate the symbol.** Every
method in this skill either scores a fixed string or emits a single-token label
that is mapped back to a symbol in code. Get this wrong and no amount of prompt
engineering, temperature tuning, or model upgrading will save the pipeline.

## Decide the method by candidate count

Read the count first — it determines everything downstream.

| Candidates | Method | Script |
|---|---|---|
| 2–10 | Single-token label MCQ, order-shuffled | `scripts/score_labels.py` |
| 11–~100 | PMI scoring, then label MCQ on the top 10 | both |
| 100+ | PMI scoring alone, then label MCQ on the top ~20 | both |

For 100+ candidates, do not attempt to put the full list in the prompt and ask
for a choice. Models attend weakly to the middle of long lists, so genes in the
center are effectively invisible regardless of context length.

`scripts/rank.py` applies this table for you. Disease name and gene list go in
as ordinary variables, and `scripts/prompts.py` converts the list into prompts
automatically:

```python
from rank import GeneRanker

ranker = GeneRanker("EPFLiGHT/Gemma-3-27B-MeditronFO")
result = ranker.rank(disease="Cystic fibrosis",
                     genes=["CFTR", "HBB", "GPR52", "GPR56", "APOE"])
result["call"], result["margin"], result["rank_stability"]
```

```bash
python scripts/rank.py --model <model-id> \
  --disease "Cystic fibrosis" --genes CFTR HBB GPR52 GPR56 APOE
```

Be precise about what "convert the list into a prompt" means here, because the
obvious reading is the one that breaks the pipeline. The list is **not** pasted
into a single prompt for the model to choose from. It is expanded into one
forced-continuation scoring per gene (Stage 1) and into rotated lettered option
blocks (Stage 2). Both keep the symbol out of the model's output.

`--dry-run` prints the generated prompts without loading a model or importing
torch. Run it once against a new candidate list — it is the cheapest way to see
that no gene symbol sits where the model would have to generate it.

```bash
python scripts/rank.py --disease "Cystic fibrosis" --genes CFTR HBB GPR52 --dry-run
```

Ranking several diseases against the same gene list should reuse one
`GeneRanker`: the neutral term is disease-independent, so `rank_many` computes
it once rather than once per disease.

A notebook walking the same path top to bottom, with the prompt inspection
reachable before any model is loaded, is at `notebooks/gene_disease_ranking.ipynb`
in this repository.

## Stage 1 — PMI scoring (the workhorse)

Score each candidate independently by forced-teacher log-likelihood. Because the
symbol is supplied rather than generated, `GPR52` and `GPR56` receive separate,
independent scores and cannot be confused.

Raw likelihood alone fails: it ranks by corpus frequency, so TP53, EGFR and TNF
top every disease. Subtract a disease-free control:

```
score(gene) = logP(gene | "Gene most strongly associated with {disease}: ")
            - logP(gene | "Gene: ")
```

This is pointwise mutual information — "how much did naming the disease raise
this gene's probability". Frequency cancels in the subtraction. The neutral term
does not depend on the disease, so compute it once and reuse it across every
disease in the run.

```bash
python scripts/score_pmi.py \
  --model EPFLiGHT/Gemma-3-27B-MeditronFO \
  --genes candidates.txt \
  --diseases diseases.txt \
  --out stage1_scores.jsonl \
  --neutral-cache neutral.npz
```

Report `Recall@K` for this stage, not accuracy. Stage 1's job is to keep the true
gene inside the shortlist; getting the exact order right is Stage 2's job.

## Stage 2 — single-token label MCQ

Take the Stage 1 shortlist (10–20 genes) and run 4–5 at a time as lettered
options. The model answers `A`–`E`, which is one token with no near-neighbours.

```
Disease: Cystic fibrosis
Options:
A. HBB
B. CFTR
C. GPR52
D. APOE
E. None of the above
Answer: B
```

Three details carry most of the value:

- **Read the logprobs of ` A`…` E` at the answer position instead of generating.**
  Fully deterministic, one forward pass, no parsing.
- **Always include "None of the above."** Without an exit, the model is forced to
  pick something, and guesses get scored as if they were knowledge. With five
  labels this caps a group at four genes; `build_mcq_rounds` clamps larger group
  sizes rather than letting the exit option fall off the end.
- **Rotate the option order and average.** If the ranking moves when the order
  moves, the model is not reading the genes — it is following position bias. That
  instability is a useful signal, not noise: surface it as a confidence flag.

```bash
python scripts/score_labels.py \
  --model EPFLiGHT/Gemma-3-27B-MeditronFO \
  --shortlist stage1_scores.jsonl \
  --top-k 20 --group-size 4 --rotations 4 \
  --out stage2_scores.jsonl
```

Record the margin between rank 1 and rank 2 for every prediction. Small margins
correlate strongly with errors, and thresholding on margin is usually the
cheapest accuracy gain available. Prefer abstaining to guessing — a pipeline that
says "no confident call" on 30% of diseases is more useful than one that is
confidently wrong.

## Stage 3 — evidence gate

An LLM score is a prior, not evidence. Before anything reaches the user:

1. Normalize every symbol against HGNC. Drop anything that is not an approved
   symbol or a known alias; map aliases to the approved symbol.
2. Look for supporting literature or genetic evidence (PubTator3, Open Targets).
3. Attach the evidence to the output. Flag calls with zero supporting evidence
   rather than silently dropping them — an LLM-only hit is the interesting case
   worth human review, but it must be labelled as such.

`scripts/normalize_genes.py` handles step 1 offline from an HGNC dump.

## Evaluation — do this before believing anything

Build the harness before tuning. Without it there is no way to tell an
improvement from a reshuffle.

```bash
python scripts/evaluate.py --pred stage1_scores.jsonl --gold gold.jsonl --k 10 50
```

Metrics:

| Metric | What it tells you |
|---|---|
| Recall@K | Did the true gene survive to the shortlist |
| MRR | Overall ranking quality |
| Family Confusion Rate | Fraction of top-1 errors sharing a symbol prefix with truth — the bug this skill targets, should reach ~0 |
| Abstention rate | How often the system declines, once margin thresholding is on |

### Four controls that are not optional

Skipping these is the most common way these projects produce impressive-looking
numbers that mean nothing.

1. **Random.** Recall@50 from 1000 candidates is 5% by chance. Print it next to
   your number every time.
2. **Frequency-only.** Rank genes by raw PubMed mention count, ignoring the
   disease. If the LLM does not clearly beat this, the LLM is reproducing corpus
   frequency and the PMI correction is not working.
3. **Disease shuffle.** Re-score with disease names swapped at random. Rankings
   should collapse. If they barely move, the model is not reading the disease.
4. **Database-only.** Score with Open Targets association scores alone. If adding
   the LLM does not improve on this, say so plainly and recommend dropping the
   LLM. Open Targets already does large-scale gene-disease association well; the
   LLM's value is confined to associations implicit in text but absent from
   curated data. Report the delta honestly even when it is unflattering — a
   pipeline that adds cost without adding signal should not ship.

`scripts/evaluate.py --controls` computes 1–3 automatically.

### Adversarial sibling test

Build a held-out set where each item's distractors are deliberately the true
gene's siblings (`GPR52` → add `GPR55`, `GPR56`, `GPR35`). This separates real
knowledge from prefix pattern-matching better than any random-distractor set.
Expect a large accuracy drop; a system that does not drop is probably being
scored on a set that was too easy.

## Choosing a model

See `references/models.md` for verified names, benchmark deltas, licenses and
VRAM. Summary of what matters here:

- **`epfl-llm/meditron-70b` (2023) is a poor fit.** It is a continued-pretraining
  model with no instruction tuning, so it cannot reliably follow a "choose from
  this list" format at all. If a user is on it, the model swap and the
  architecture fix are two separate changes — measure them separately or you will
  not know which one worked.
- Medical fine-tuning targets clinical reasoning (USMLE-style), which is a
  different axis from gene knowledge. Always benchmark the plain base model
  alongside the medicalized one. On several MeditronFO variants the medical
  gain is only ~1 point over base, so the base model may be equally good here at
  lower licensing friction.
- Larger is not automatically better: at time of writing the 27B MeditronFO
  outscores the 70B one on standard medical benchmarks.

Whatever the user's plan, run `scripts/check_tokenizer.py` first. It shows how
the candidate symbols actually tokenize and confirms that ` A`…` E` are single
tokens for that tokenizer — a five-second check that prevents a whole class of
silent failures.

```bash
python scripts/check_tokenizer.py --model <model-id> --genes GPR52 GPR56 SLC6A4
```

## Commercial use

If the user has any commercial intent, raise licensing before they invest in
fine-tuning. Model weights and training corpora carry separate terms, and the
MeditronFO corpus is research-use with an explicit note that the models are not
approved for clinical deployment. Details and the current state of each source
are in `references/data-sources.md` — verify against the live pages rather than
trusting the notes, since terms change.

## Reference files

- `scripts/rank.py` — entry point taking disease and gene list as variables
- `scripts/prompts.py` — the one place a gene list becomes prompt text
- `references/models.md` — model comparison, licenses, VRAM, tokenizer notes
- `references/scoring.md` — PMI derivation, calibration, length normalization, batching
- `references/data-sources.md` — Open Targets, HGNC, PubTator3, GWAS Catalog; licenses and access
- `examples/` — a sibling-heavy smoke-test set (GPR52/55/56, HBB/HBA1/HBA2) plus the commands to run it
