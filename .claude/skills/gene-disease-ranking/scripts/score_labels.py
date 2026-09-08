#!/usr/bin/env python3
"""Stage 2: rerank a shortlist using single-token A-E labels.

The model answers with one letter, never a gene symbol, so similar symbols cannot
be confused. Option order is rotated and scores averaged, which both removes
position bias and exposes it: if the ranking swings across rotations, the model
is following position rather than reading the genes.

  python score_labels.py --model EPFLiGHT/Gemma-3-27B-MeditronFO \
      --shortlist stage1.jsonl --top-k 20 --group-size 4 --rotations 4 \
      --out stage2.jsonl
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

LABELS = list("ABCDE")
NONE_OPT = "None of the above"

FEWSHOT = """Disease: Cystic fibrosis
Options:
A. HBB
B. CFTR
C. GPR52
D. APOE
E. None of the above
Answer: B

Disease: Sickle cell disease
Options:
A. CFTR
B. APOE
C. HBB
D. GPR56
E. None of the above
Answer: C

"""


def build_prompt(disease, options, fewshot=True):
    lines = [FEWSHOT] if fewshot else []
    lines.append(f"Disease: {disease}\nOptions:")
    for lab, opt in zip(LABELS, options):
        lines.append(f"{lab}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


class LabelScorer:
    def __init__(self, model_id, dtype="bfloat16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="auto"
        ).eval()
        self.label_ids = []
        for lab in LABELS:
            ids = self.tok.encode(" " + lab, add_special_tokens=False)
            self.label_ids.append(ids[-1])
        if len(set(self.label_ids)) != len(LABELS):
            raise SystemExit(
                "Label tokens collide for this tokenizer. Run "
                "check_tokenizer.py and pick different labels.")

    @torch.no_grad()
    def score(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        return {lab: lp[i].item() for lab, i in zip(LABELS, self.label_ids)}


def rerank(scorer, disease, genes, group_size=4, rotations=4):
    """Score genes in groups, rotating option order. Returns {gene: mean logprob}."""
    acc = defaultdict(list)
    groups = [genes[i:i + group_size] for i in range(0, len(genes), group_size)]
    for group in groups:
        if len(group) < 2:
            continue
        for r in range(min(rotations, len(group))):
            rolled = group[r:] + group[:r]
            options = rolled + [NONE_OPT]
            options = options[:len(LABELS)]
            s = scorer.score(build_prompt(disease, options))
            for lab, opt in zip(LABELS, options):
                acc[opt].append(s[lab])
    return {g: float(np.mean(v)) for g, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shortlist", required=True, help="stage1 jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--margin-threshold", type=float, default=0.5,
                    help="abstain when top1 - top2 falls below this")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    scorer = LabelScorer(args.model, dtype=args.dtype)

    with open(args.shortlist) as fin, open(args.out, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            disease = rec["disease"]
            genes = [r["gene"] for r in rec["ranked"][:args.top_k]]
            scores = rerank(scorer, disease, genes,
                            args.group_size, args.rotations)

            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            gene_ranked = [(g, s) for g, s in ranked if g != NONE_OPT]
            none_score = scores.get(NONE_OPT)

            margin = (gene_ranked[0][1] - gene_ranked[1][1]
                      if len(gene_ranked) > 1 else float("inf"))

            if none_score is not None and none_score >= gene_ranked[0][1]:
                call, reason = None, "model preferred 'none of the above'"
            elif margin < args.margin_threshold:
                call, reason = None, f"margin {margin:.3f} below threshold"
            else:
                call, reason = gene_ranked[0][0], None

            fout.write(json.dumps({
                "disease": disease,
                "call": call,
                "abstain_reason": reason,
                "margin": round(float(margin), 4),
                "none_score": None if none_score is None else round(none_score, 4),
                "ranked": [{"gene": g, "score": round(s, 4)}
                           for g, s in gene_ranked],
            }, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  {disease}: {call or 'ABSTAIN'} (margin {margin:.3f})")

    print(f"\nwrote {args.out}")
    print("Tune --margin-threshold on a validation split, never on the test set.")


if __name__ == "__main__":
    main()
