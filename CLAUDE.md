# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The invariant everything is built around

**The model must never emit a gene symbol as free text.** `README.md` explains why
(symbols are multi-token, so decoding follows corpus frequency and `GPR56` beats
`GPR52`). Every method here does one of two things instead:

- score a string **we supply** (teacher-forced log-likelihood), or
- read the logprob of a **single-letter label** (`A`–`E`) and map it back to a
  symbol in code.

Any change that lets a symbol appear at a position the model generates is a
regression, however good the output looks. `--dry-run` on `rank.py` prints the
prompts without loading a model; use it to check this on new candidate lists.

## Two separate implementations — know which one you are in

They do not share code and are not meant to.

| | `.claude/skills/gene-disease-ranking/scripts/` | `notebooks/gene_disease_ranking01.ipynb` |
|---|---|---|
| Shape | importable modules, backend-abstracted | **one self-contained notebook, imports no project `.py`** |
| Pipeline | PMI scoring → label MCQ → evidence gate | binary screening → MCQ (group 5 + exit) → round-robin + Bradley-Terry |
| Scale | 2–100+ candidates | 20,000 → 1,000 → 100 → ranked |
| Backends | transformers / llama.cpp / Ollama | Ollama `/api/generate` only, `qwen3:14b` by default |
| Language | English | Japanese prose throughout |

The notebook is deliberately free of project imports — the user asked for that
explicitly. Do not "clean it up" by extracting helpers into `.py` files.
`notebooks/gene_disease_ranking.ipynb` (no `01`) is the older walkthrough that
*does* import from `scripts/`; both it and `repeat_ranking.ipynb` carry a latent
`import importlib` / `importlib.util` bug that has been left alone on purpose.

### The notebook's three stages differ on purpose

Stage 0 screens all candidates one at a time with a Yes/No question, ranked by
`logP(Yes) − logP(No)`. Stages 1–2 compare candidates against each other.
Question wording and option text are **not** shared between them — measured
choices that are optimal in one stage are harmful in the other (the mechanism
wording lifts the binary screen 1.85 → 8.54 separation but drops MCQ rejection
from 5/20 to 0/20). UniProt protein names are on from stage 1 onward and off in
stage 0, again by measurement. See `notebooks/EXPERIMENTS.md` before changing any
prompt string.

Stage 0 puts the gene symbol **last** in the prompt so the whole prefix is shared
across candidates and stays in the KV cache — 1.8× faster and, measured on 1,000
genes, better ranked too.

## Commands

Offline regression tests — no model, no GPU, numpy only:

```bash
python .claude/skills/gene-disease-ranking/tests/test_offline.py
python .claude/skills/gene-disease-ranking/tests/test_backends.py
```

Both are plain scripts, not pytest; run a single one by running its file.
`test_backends.py` stands up a fake Ollama server on a real socket.

Before building on a runtime or a model:

```bash
python .claude/skills/gene-disease-ranking/scripts/check_backend.py --backend ollama --model gemma3:27b
python .claude/skills/gene-disease-ranking/scripts/check_tokenizer.py --model <model-id> --genes GPR52 GPR56 SLC6A4
```

Inspect generated prompts without importing torch:

```bash
cd .claude/skills/gene-disease-ranking && python scripts/rank.py --disease "Cystic fibrosis" --genes CFTR HBB GPR52 --dry-run
```

Evaluate with the mandatory controls (random / frequency-only / disease-shuffle):

```bash
python .claude/skills/gene-disease-ranking/scripts/evaluate.py --pred stage1_scores.jsonl --gold gold.jsonl --k 10 50 --controls
```

## Backend capability is the first question, not an implementation detail

The two stages ask for different things, and not every runtime can do both.
`Backend.score_continuations()` raises `ScoringUnsupported` where it cannot; the
per-backend capability table is in `README.md` and `SKILL.md`.

Ollama's `logprobs` cover tokens the model *generated*; there is no `echo` and no
scoring endpoint. `rank.py` degrades to stage 2 alone with a warning rather than
failing — but nothing is then cancelling corpus frequency, so the frequency-only
control decides whether the result means anything.

`resolve_ollama_gguf()` locates the GGUF Ollama already downloaded so `llamacpp`
can read the same weights with no second copy. Under a Docker *named* volume that
shortcut breaks (Linux: root-owned under `/var/lib/docker/volumes/`; Docker
Desktop: inside a VM, not on the host); `check_backend.py` detects this and prints
the exact `docker cp` line.

## Gene list format

Candidate lists in `notebooks/` are TSV: `symbol<TAB>HGNC name<TAB>UniProt protein name`.
`#` comments and blank lines are skipped; the name columns are optional. Files are
named `genelist_<disease>_<N>.txt`. `genelist_cystinuria_slc50.txt` is the
deliberately hard case — 42 of 50 genes are SLC-family.

Cystinuria is the default working disease because the answer is unambiguous:
Open Targets gives SLC3A1 0.853 and SLC7A9 0.850, with third place (PREPL, 0.566)
far behind.

## Measurement discipline

`notebooks/EXPERIMENTS.md` is the record of what was actually measured, including a
"撤回した測定" section listing measurements that were retracted and why. Read it
before proposing a change to any prompt, group size, or threshold — several
obvious-sounding ideas are already measured and rejected there.

Two failure modes have bitten this project repeatedly and are worth guarding
against explicitly:

- **Matching the decoy pool.** Random decoys put every configuration at ceiling,
  so differences vanish; same-family decoys (SLC-prefix, low Open Targets score)
  are what separate methods. A comparison whose two arms used different decoy
  pools is invalid, not merely noisy.
- **Averages vs ranks.** Stage 0's mean separation favours one prompt layout while
  the actual rank of the true genes among 1,000 favours the other. Judge on the
  outcome that matters, not the summary statistic.

Temperature is 0 everywhere, so a repeated run returns an identical answer.
Repeats only mean something if the candidate order is reshuffled between them.

`notebooks/gene_disease_ranking02.ipynb` (the disease-mechanism format) has its own
section at the end of `EXPERIMENTS.md` ("ノートブック 02"), with three things to know
before touching it:

- **The top of the ranking is decided by how the mechanism is written.** Swapping one
  medically valid description for another moved C4A from 0.96 to 0.00 and DRD2 from
  0.27 to 0.89, and gemma3:27b reproduces the pattern (+0.83 rank agreement). Treat a
  single run as "genes that fit this description", not "genes for this disease".
  The swing lives in the comparative stages (0.5 onward); stage 0 barely cares — all
  11 schizophrenia answers stayed in the top 10% under all three descriptions.
- **Several fixes are already measured and rejected** there: a Yes/No/Unknown stage 0,
  protein names or aliases in stage 0, disease-name-only questions, AND-phrased steps,
  "indirect involvement" wording. Aliases are also a hazard in their own right: 590
  of them are another gene's official symbol.
- **What the model does not know cannot be prompted out of it.** CHRM4 (xanomeline's
  target) stays near the bottom under every wording, description and a second model.
  Use the output to check known targets, not as a primary screen for new ones.

## Ollama on this machine

VRAM is 17.8 GiB — `gemma3:27b` (~15GB) and `qwen3:14b` (10GB) cannot both be
resident. If another process touches the other model, Ollama evicts and reloads on
every call and a 0.3 s/gene loop becomes 30 s/gene. The notebook's stage 0 prints
resident models before starting and flags any cycle 3× slower than its best.
Check with `ollama ps`; free memory with `ollama stop <model>`.

Never kill `ollama serve` — other work on this machine uses it. Kill only the
specific Python process.

## Conventions

Comments inside the `01` notebook are Japanese and explain *why* a measured choice
was made, usually with the number that justifies it; match that when editing it.

Run outputs (`*_ranking.csv`, `gene_scores*`, `stage[12]*.jsonl`, `*.npz`) and HGNC
dumps are gitignored as regenerable.
