# Scoring methodology

## Why PMI rather than raw likelihood

Raw `logP(gene | disease prompt)` conflates two things: how strongly the disease
predicts the gene, and how common the gene is in the pretraining corpus. The
second term dominates. A ranking built on raw likelihood returns roughly the same
list of famous genes for every disease, which looks plausible and is useless.

Subtracting a disease-free control removes the frequency term:

```
PMI(gene, disease) = logP(gene | "Gene most strongly associated with {disease}: ")
                   - logP(gene | "Gene: ")
```

Both terms score the *same* token sequence, so tokenization-length effects, digit
splitting, and symbol rarity largely cancel. This is why explicit length
normalization is usually unnecessary here and often harmful — dividing by token
count after the subtraction reintroduces a bias toward short symbols.

The neutral term is independent of the disease. Compute it once per (model, gene
set) and cache it. For 1000 genes across 200 diseases this turns 400,000 forward
passes into 201,000.

## Prompt templates

Keep the disease and neutral prompts as similar as possible except for the
disease mention, so the subtraction isolates the disease signal:

```python
DISEASE_TMPL = "Gene most strongly associated with {disease}: "
NEUTRAL_TMPL = "Gene: "
```

An alternative neutral that matches structure more closely:

```python
NEUTRAL_TMPL = "Gene most strongly associated with disease: "
```

Try both on your gold set. The tighter match sometimes over-subtracts, so this is
an empirical choice rather than a settled one.

## Tokenization boundary

Do not compute the prefix token count by tokenizing the prefix separately and
assuming the same boundary holds in the concatenated string. Subword merges can
straddle the join, shifting the boundary by a token and silently corrupting every
score.

Instead, tokenize prefix and target separately and concatenate the ID lists:

```python
prefix_ids = tok.encode(prefix, add_special_tokens=False)
target_ids = tok.encode(target, add_special_tokens=False)
input_ids  = bos + prefix_ids + target_ids
# target logprobs live at positions len(bos)+len(prefix_ids)-1 .. -2 in the
# shifted logits
```

This makes the boundary exact by construction.

## Batching

Right padding is correct for scoring with a causal LM: attention only looks
leftward, so padding on the right cannot influence real token positions, provided
the attention mask is set and padded positions are excluded from the sum.

Left padding is required for *generation*, not scoring. Mixing the two conventions
is a common source of quiet wrongness.

## Calibration and abstention

Record for every prediction:

- `top1_score`, `top2_score`, `margin = top1 - top2`
- `rank_stability` — how much the top-1 changes across option rotations in Stage 2

Fit the margin threshold on a validation split, not on the test set. Plot
accuracy against abstention rate and pick an operating point with the user
rather than for them; the right trade-off depends on whether a wrong call costs
a wasted experiment or merely a wasted glance.

## Combining with database scores

```
final = w1 * norm(open_targets_score) + w2 * norm(llm_pmi)
```

Normalize each to zero mean and unit variance across the candidate set for that
disease before combining, otherwise the weights are meaningless. Fit `w1, w2` on
known pairs by grid search — with two parameters, anything more elaborate is
overfitting.

Always report the three-way comparison: database alone, LLM alone, combined. If
the combination does not beat the database alone, the honest recommendation is to
drop the LLM, and that recommendation should be made explicitly rather than
buried.

## Known limitations

- PMI scores are not comparable across diseases without per-disease
  normalization. Rank within a disease; do not threshold globally on raw PMI.
- Aliases fragment the signal. `HTT` and `IT15` refer to the same gene and will
  score differently. Normalize to approved symbols before scoring, and consider
  scoring the approved symbol and top alias then taking the maximum.
- Genes named after the disease (`HTT` / Huntington) leak lexically rather than
  biologically. These inflate benchmark scores without reflecting real knowledge.
  Flag them in the gold set and report accuracy with and without them.
