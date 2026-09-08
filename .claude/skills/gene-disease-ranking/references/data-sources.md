# Data sources

Licence terms change. Treat this file as a starting map, and verify the current
terms on each source's own site before shipping anything commercial. When you
cannot confirm a term, say so rather than guessing.

## Candidate generation and evidence

| Source | Use here | Licence as of last check |
|---|---|---|
| Open Targets Platform | Disease-gene association scores; the strongest single baseline and a good candidate generator | CC0 |
| GWAS Catalog (EBI) | Genetic association evidence | Open, citation requested |
| PubTator3 (NCBI) | Gene mention normalization in literature; evidence for Stage 3 | US Government work |
| HGNC | Approved symbols and aliases; required for normalization | Freely distributed |
| DisGeNET | Disease-gene associations | Moved to a paid licence; commercial use requires a contract |

Open Targets is both a candidate source and a baseline. Use it for both — the
same download serves the Stage 0 shortlist and the database-only control.

## HGNC normalization

Download the complete approved symbol set with aliases and previous symbols.
Build three maps: approved symbol, alias to approved, previous to approved. Any
model output that hits none of the three is not a gene and should be dropped
before it reaches scoring or reporting.

This step alone eliminates hallucinated symbols. It does not eliminate family
confusion, because `GPR56` is a perfectly valid approved symbol — it is simply
the wrong one. Only the architectural fix addresses that.

## Local data

If the user has bioinformatics databases already on disk, check those before
downloading. Ask where they keep them rather than assuming a path. Note the
snapshot date in the output — a two-year-old Open Targets release will quietly
depress the database baseline and make the LLM look better than it is.

## Offline operation

Networks in lab environments are often restricted. The scoring scripts here need
no network access once the model weights and the HGNC dump are local. Keep the
literature-evidence step (Stage 3) optional and clearly separated so the core
pipeline still runs air-gapped.

## Model weight licences

Separate from data licences, and separately restrictive. Apache 2.0 weights
(Apertus-based) are the least encumbered. Gemma-derived weights carry the Gemma
Terms with a use-restriction policy. Llama-derived weights carry Meta's community
licence.

Note also that a permissive weight licence does not automatically extend to the
training corpus or to outputs in regulated settings. The MeditronFO corpus is
research-use, and the project states the models are not approved for clinical
deployment. Gene prioritization for research is a different context from clinical
decision support, but if the user's roadmap ends in a clinical product, flag this
early — it is much cheaper to raise before fine-tuning than after.
