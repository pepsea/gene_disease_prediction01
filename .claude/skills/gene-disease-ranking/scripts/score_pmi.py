#!/usr/bin/env python3
"""Stage 1: rank candidate genes by PMI against a disease.

Scores each gene as a forced continuation, so symbols are never generated and
cannot drift to a similar-looking sibling. Subtracts a disease-free control to
remove corpus-frequency bias.

  python score_pmi.py --model EPFLiGHT/Gemma-3-27B-MeditronFO \
      --genes candidates.txt --diseases diseases.txt \
      --out stage1.jsonl --neutral-cache neutral.npz

genes.txt / diseases.txt: one entry per line, blank lines and # comments ignored.
Output: one JSON object per disease, genes sorted by descending PMI.
"""
import argparse
import json
import os

import numpy as np
import torch

DISEASE_TMPL = "Gene most strongly associated with {disease}: "
NEUTRAL_TMPL = "Gene: "


def read_lines(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


class Scorer:
    def __init__(self, model_id, dtype="bfloat16", device_map="auto"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map=device_map
        ).eval()
        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tok.eos_token_id or 0
        bos = getattr(self.tok, "bos_token_id", None)
        self.bos = [bos] if bos is not None else []

    @torch.no_grad()
    def score(self, prefix, targets, batch_size=32):
        """Sum of log P(target tokens | prefix) for each target. Right padding.

        Prefix and target are tokenized separately and their id lists
        concatenated, so the boundary is exact and unaffected by subword merges
        that might otherwise straddle the join.
        """
        prefix_ids = self.tok.encode(prefix, add_special_tokens=False)
        head = self.bos + prefix_ids
        n_head = len(head)

        results = np.zeros(len(targets), dtype=np.float64)
        for i in range(0, len(targets), batch_size):
            chunk = targets[i:i + batch_size]
            seqs = [head + self.tok.encode(t, add_special_tokens=False)
                    for t in chunk]
            n_tgt = [len(s) - n_head for s in seqs]
            maxlen = max(len(s) for s in seqs)

            input_ids = torch.full((len(seqs), maxlen), self.pad_id,
                                   dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, :len(s)] = torch.tensor(s)
                attn[j, :len(s)] = 1

            input_ids = input_ids.to(self.model.device)
            attn = attn.to(self.model.device)

            logits = self.model(input_ids=input_ids,
                                attention_mask=attn).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = input_ids[:, 1:]
            tok_lp = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

            # position p in tok_lp scores input_ids[:, p+1]; target tokens start
            # at absolute index n_head, i.e. tok_lp index n_head - 1
            for j, nt in enumerate(n_tgt):
                start = n_head - 1
                results[i + j] = tok_lp[j, start:start + nt].sum().item()

        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--diseases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neutral-cache", default=None,
                    help="npz path; computed once and reused across runs")
    ap.add_argument("--disease-template", default=DISEASE_TMPL)
    ap.add_argument("--neutral-template", default=NEUTRAL_TMPL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--top-n", type=int, default=0,
                    help="keep only top N per disease (0 = keep all)")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    genes = read_lines(args.genes)
    diseases = read_lines(args.diseases)
    print(f"{len(genes)} genes x {len(diseases)} diseases")

    scorer = Scorer(args.model, dtype=args.dtype)

    # Neutral scores are disease-independent: compute once, cache, reuse.
    neutral = None
    if args.neutral_cache and os.path.exists(args.neutral_cache):
        z = np.load(args.neutral_cache, allow_pickle=True)
        if list(z["genes"]) == genes and str(z["model"]) == args.model:
            neutral = z["scores"]
            print("neutral scores loaded from cache")
        else:
            print("cache does not match this gene set/model; recomputing")
    if neutral is None:
        print("scoring neutral control...")
        neutral = scorer.score(args.neutral_template, genes, args.batch_size)
        if args.neutral_cache:
            np.savez(args.neutral_cache, genes=np.array(genes, dtype=object),
                     scores=neutral, model=args.model)

    with open(args.out, "w") as fout:
        for d in diseases:
            cond = scorer.score(args.disease_template.format(disease=d),
                                genes, args.batch_size)
            pmi = cond - neutral
            order = np.argsort(-pmi)
            if args.top_n:
                order = order[:args.top_n]
            ranked = [{"gene": genes[i],
                       "pmi": round(float(pmi[i]), 4),
                       "cond": round(float(cond[i]), 4),
                       "neutral": round(float(neutral[i]), 4)}
                      for i in order]
            fout.write(json.dumps({"disease": d, "ranked": ranked},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            top = ", ".join(r["gene"] for r in ranked[:5])
            print(f"  {d}: {top}")

    print(f"\nwrote {args.out}")
    print("Next: run evaluate.py --controls before drawing any conclusion.")


if __name__ == "__main__":
    main()
