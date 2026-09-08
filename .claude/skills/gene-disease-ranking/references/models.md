# Model selection

Verified against Hugging Face model cards in September 2026. Model cards change —
re-check before committing to one, and treat every number here as a starting
point for your own measurement rather than a result.

## Candidates

| Model ID | Params | License | Notes |
|---|---|---|---|
| `EPFLiGHT/Gemma-3-27B-MeditronFO` | 29B | Gemma Terms | Best medical benchmark average in the family; fits one GPU at Q4 (~18GB) |
| `EPFLiGHT/Apertus-70B-MeditronFO` | 71B | Apache 2.0 | Cleanest license; ~140GB bf16, ~40GB Q4 |
| `EPFLiGHT/OLMo-2-32B-MeditronFO` | 32B | check card | Fully open lineage |
| `EPFLiGHT/EuroLLM-22B-MeditronFO` | 23B | check card | Multilingual base |
| `google/gemma-3-27b-it` | 27B | Gemma Terms | The control. Run it. |
| `aaditya/Llama3-OpenBioLLM-70B` | 70B | Llama 3 | April 2024, dated. Note the repo name — `Anshuman73/OpenBioLLM-70B` does not exist and appears in some LLM-generated guides. |
| `epfl-llm/meditron-70b` | 70B | Llama 2 | 2023, no instruction tuning. Legacy only. |

## Benchmark deltas from medicalization

From the MeditronFO model cards (accuracy %, average over MedMCQA / MedQA /
PubMedQA / MedXpertQA / HealthBench Hard):

| Model | Base | After medical FT | Delta |
|---|---|---|---|
| Apertus-70B | 44.90 | 51.43 | +6.53 |
| Gemma-3-27B | 55.20 | 56.45 | +1.25 |

Two things follow. First, the 27B beats the 70B in absolute terms — parameter
count is not the deciding variable. Second, the Gemma medical gain is small
enough that on a non-clinical task like gene ranking the base model may match the
medicalized one, which is exactly why the base model belongs in the benchmark as
a control rather than being assumed inferior.

Note also that these are clinical-reasoning benchmarks. None of them measures
gene-disease knowledge. Do not use them to predict performance on this task; use
them only to narrow the shortlist before measuring properly.

## Tokenizer

Vocabulary size varies widely across these models (Llama-2 32k, Llama-3 128k,
Gemma-3 ~262k), which changes how many tokens a symbol costs, but **no tokenizer
in this list stores a gene symbol as a single token.** A larger vocabulary
reduces sequence length; it does not fix family confusion. Only the architectural
fix does that.

What the tokenizer does determine:

- Whether ` A` … ` E` are single tokens (required for label MCQ). Verify per model.
- Whether digits are split individually, which affects PMI length effects.

`scripts/check_tokenizer.py` reports both.

## Inference stack

Hugging Face `transformers` is enough for scoring — logprob extraction is a plain
forward pass and needs no generation server. vLLM and SGLang help when you need
throughput across many diseases, but add setup cost; start with transformers and
move only if scoring time becomes the bottleneck.

For quantized local runs, confirm what the runtime can do before depending on
it. Checked in September 2026:

- **Ollama** exposes `logprobs` / `top_logprobs` (v0.12.11+) for tokens it
  *generates*, which covers stage 2. It has no `echo` and no scoring endpoint,
  so stage 1 is not possible through it. Ollama Cloud has been reported to
  return null logprobs even when asked.
- **llama.cpp** via `llama-cpp-python` does both: construct with
  `logits_all=True` and call `create_completion(..., echo=True, logprobs=N,
  max_tokens=0)` to score a supplied sequence. Passing token ids rather than
  text keeps the prefix/target boundary exact.

`scripts/check_backend.py` probes a live runtime instead of trusting this note,
which is the right habit — these APIs move quickly.
