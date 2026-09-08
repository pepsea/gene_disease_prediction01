#!/usr/bin/env python3
"""Entry point: disease name and gene list go in as variables.

    from rank import GeneRanker, preview

    ranker = GeneRanker("EPFLiGHT/Gemma-3-27B-MeditronFO")
    result = ranker.rank(disease="Cystic fibrosis",
                         genes=["CFTR", "HBB", "GPR52", "GPR56", "APOE"])
    print(result["call"], result["margin"])

The gene list is converted into prompts automatically by prompts.py. Note what
that conversion is *not*: the list is never pasted into one prompt for the model
to choose from. Stage 1 turns each gene into its own forced-continuation target,
and stage 2 turns groups of genes into lettered options the model answers with a
single token. A gene symbol is never generated, so GPR52 cannot drift to GPR56.

Inspect the generated prompts without a model, torch or weights:

    python rank.py --disease "Cystic fibrosis" --genes CFTR HBB GPR52 --dry-run

Score for real:

    python rank.py --model <model-id> \
        --disease "Cystic fibrosis" --genes CFTR HBB GPR52 GPR56 APOE \
        --out result.jsonl

Files still work, and mix freely with inline variables:

    python rank.py --model <model-id> \
        --diseases-file ../examples/diseases.txt \
        --genes-file ../examples/candidates.txt --out result.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import (DISEASE_TMPL, NEUTRAL_TMPL, LABELS, NONE_OPT,  # noqa: E402
                     build_mcq_rounds, build_pmi_pairs, clean_genes,
                     plan_stages)


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f
                if l.strip() and not l.strip().startswith("#")]


def _load_hgnc(path):
    from normalize_genes import load_hgnc
    return load_hgnc(path)


def preview(disease, genes, top_k=None, group_size=4, rotations=4,
            fewshot=True, disease_tmpl=DISEASE_TMPL,
            neutral_tmpl=NEUTRAL_TMPL):
    """Show exactly what the gene list becomes, without loading a model.

    Cheap enough to run before every real job, and the fastest way to confirm
    that a symbol appears only as a scored target or a lettered option.
    """
    genes, report = clean_genes(genes)
    plan = plan_stages(len(genes), top_k)
    pmi = (build_pmi_pairs(disease, genes, disease_tmpl, neutral_tmpl)
           if "pmi" in plan["stages"] else None)
    shortlist = genes[:plan["shortlist"]]
    rounds = build_mcq_rounds(disease, shortlist, group_size, rotations,
                              fewshot)
    return {"disease": disease, "genes": genes, "plan": plan,
            "normalization": report, "stage1": pmi, "stage2_rounds": rounds}


class GeneRanker:
    """Holds the model and the reusable neutral scores across diseases.

    Build one and call rank() per disease. The neutral term does not depend on
    the disease, so it is computed once per gene set and reused: for 1000 genes
    over 200 diseases that is 201,000 forward passes instead of 400,000.
    """

    def __init__(self, model, dtype="bfloat16", batch_size=32, hgnc=None,
                 disease_tmpl=DISEASE_TMPL, neutral_tmpl=NEUTRAL_TMPL):
        self.model_id = model
        self.dtype = dtype
        self.batch_size = batch_size
        self.disease_tmpl = disease_tmpl
        self.neutral_tmpl = neutral_tmpl
        self._pmi_scorer = None
        self._label_scorer = None
        self._neutral = {}
        self.approved = self.alias_to = None
        if hgnc:
            self.approved, self.alias_to = _load_hgnc(hgnc)

    # Both scorers load the same weights twice if used naively. Share one.
    def _load(self):
        if self._pmi_scorer is not None:
            return
        from score_pmi import Scorer
        from score_labels import LabelScorer
        self._pmi_scorer = Scorer(self.model_id, dtype=self.dtype)
        self._label_scorer = LabelScorer.__new__(LabelScorer)
        self._label_scorer.tok = self._pmi_scorer.tok
        self._label_scorer.model = self._pmi_scorer.model
        ids = [self._label_scorer.tok.encode(" " + lab,
                                             add_special_tokens=False)[-1]
               for lab in LABELS]
        if len(set(ids)) != len(LABELS):
            raise SystemExit("Label tokens collide for this tokenizer. Run "
                             "check_tokenizer.py and pick different labels.")
        self._label_scorer.label_ids = ids

    def neutral_scores(self, genes, cache_path=None):
        key = tuple(genes)
        if key in self._neutral:
            return self._neutral[key]
        import numpy as np
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            if list(z["genes"]) == list(genes) and str(z["model"]) == self.model_id:
                self._neutral[key] = z["scores"]
                return self._neutral[key]
        self._load()
        scores = self._pmi_scorer.score(self.neutral_tmpl, list(genes),
                                        self.batch_size)
        if cache_path:
            np.savez(cache_path, genes=np.array(list(genes), dtype=object),
                     scores=scores, model=self.model_id)
        self._neutral[key] = scores
        return scores

    def rank(self, disease, genes, top_k=None, group_size=4, rotations=4,
             margin_threshold=0.5, fewshot=True, neutral_cache=None):
        """Rank `genes` for `disease`. Both are plain variables.

        Returns a dict with the stage 1 ranking (when used), the stage 2
        scores, and either a call or an explicit abstention with its reason.
        Abstaining beats guessing: a pipeline that declines on 30% of diseases
        is more useful than one that is confidently wrong.
        """
        import numpy as np

        genes, report = clean_genes(genes, self.approved, self.alias_to)
        plan = plan_stages(len(genes), top_k)
        self._load()

        stage1 = None
        shortlist = genes
        if "pmi" in plan["stages"]:
            pairs = build_pmi_pairs(disease, genes, self.disease_tmpl,
                                    self.neutral_tmpl)
            cond = self._pmi_scorer.score(pairs["conditional_prefix"], genes,
                                          self.batch_size)
            neutral = self.neutral_scores(genes, neutral_cache)
            pmi = cond - neutral
            order = np.argsort(-pmi)
            stage1 = [{"gene": genes[i], "pmi": round(float(pmi[i]), 4),
                       "cond": round(float(cond[i]), 4),
                       "neutral": round(float(neutral[i]), 4)}
                      for i in order]
            shortlist = [r["gene"] for r in stage1[:plan["shortlist"]]]

        rounds = build_mcq_rounds(disease, shortlist, group_size, rotations,
                                  fewshot)
        per_gene, round_winners = {}, []
        for rnd in rounds:
            s = self._label_scorer.score(rnd["prompt"])
            best, best_lp = None, None
            for lab, opt in zip(LABELS, rnd["options"]):
                per_gene.setdefault(opt, []).append(s[lab])
                if best_lp is None or s[lab] > best_lp:
                    best, best_lp = opt, s[lab]
            round_winners.append(best)

        scores = {g: float(np.mean(v)) for g, v in per_gene.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        gene_ranked = [(g, s) for g, s in ranked if g != NONE_OPT]
        none_score = scores.get(NONE_OPT)

        if not gene_ranked:
            raise SystemExit("no genes scored; check the candidate list")

        margin = (gene_ranked[0][1] - gene_ranked[1][1]
                  if len(gene_ranked) > 1 else float("inf"))
        top1 = gene_ranked[0][0]

        # How often the eventual winner also won its own rotation. Low values
        # mean position, not the gene, is driving the answer.
        contested = [w for w in round_winners if w is not None]
        stability = (sum(1 for w in contested if w == top1) /
                     max(1, sum(1 for rnd, w in zip(rounds, contested)
                                if top1 in rnd["options"])))

        if none_score is not None and none_score >= gene_ranked[0][1]:
            call, reason = None, "model preferred 'none of the above'"
        elif margin < margin_threshold:
            call, reason = None, f"margin {margin:.3f} below threshold"
        else:
            call, reason = top1, None

        return {
            "disease": disease,
            "model": self.model_id,
            "n_candidates": len(genes),
            "plan": plan,
            "normalization": report,
            "stage1": stage1,
            "shortlist": shortlist,
            "stage2": [{"gene": g, "score": round(s, 4)} for g, s in gene_ranked],
            "call": call,
            "abstain_reason": reason,
            "margin": None if margin == float("inf") else round(float(margin), 4),
            "none_score": None if none_score is None else round(none_score, 4),
            "rank_stability": round(float(stability), 4),
            # evaluate.py reads "ranked"; keep it so results feed straight in.
            "ranked": [{"gene": g, "score": round(s, 4)} for g, s in gene_ranked],
        }

    def rank_many(self, diseases, genes, **kw):
        """Same gene list across several diseases, neutral term computed once."""
        return [self.rank(d, genes, **kw) for d in diseases]


def rank_genes(disease, genes, model, **kw):
    """One-shot convenience. Loads the model per call, so prefer GeneRanker
    when ranking more than one disease."""
    ctor = {k: kw.pop(k) for k in ("dtype", "batch_size", "hgnc")
            if k in kw}
    return GeneRanker(model, **ctor).rank(disease, genes, **kw)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--model", help="required unless --dry-run")
    ap.add_argument("--disease", action="append", default=[],
                    help="disease name; repeat for several")
    ap.add_argument("--diseases-file")
    ap.add_argument("--genes", nargs="+", default=[],
                    help="gene symbols inline")
    ap.add_argument("--genes-file")
    ap.add_argument("--out")
    ap.add_argument("--hgnc", help="HGNC dump; enables alias normalization")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--margin-threshold", type=float, default=0.5)
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--neutral-cache")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the generated prompts and exit; no model")
    ap.add_argument("--show-prompts", type=int, default=3,
                    help="how many stage 2 prompts to print in --dry-run")
    args = ap.parse_args()

    diseases = list(args.disease)
    if args.diseases_file:
        diseases += read_lines(args.diseases_file)
    genes = list(args.genes)
    if args.genes_file:
        genes += read_lines(args.genes_file)

    if not diseases:
        ap.error("give at least one --disease or a --diseases-file")
    if len(genes) < 2:
        ap.error("give at least 2 genes via --genes or --genes-file")

    if args.dry_run:
        for d in diseases:
            p = preview(d, genes, args.top_k, args.group_size, args.rotations,
                        not args.no_fewshot)
            print(f"\n=== {d} ===")
            print(f"  candidates : {len(p['genes'])}")
            print(f"  plan       : {' -> '.join(p['plan']['stages'])} "
                  f"({p['plan']['reason']})")
            if p["stage1"]:
                pre = p["stage1"]["conditional_prefix"]
                print(f"\n  stage 1: {len(p['genes'])} forced-continuation "
                      f"scorings, each of the form")
                print(f"    prefix : {pre!r}")
                print(f"    target : {p['genes'][0]!r}   "
                      f"(then {p['genes'][1]!r}, ... scored separately)")
                print(f"    minus  : {p['stage1']['neutral_prefix']!r} "
                      f"+ same target, computed once for all diseases")
            n_opt = (len(p["stage2_rounds"][0]["options"]) - 1
                     if p["stage2_rounds"] else 0)
            print(f"\n  stage 2: {len(p['stage2_rounds'])} MCQ prompts, "
                  f"{n_opt} genes + exit option each, order rotated")
            for rnd in p["stage2_rounds"][:args.show_prompts]:
                print(f"\n  --- group {rnd['group']} rotation {rnd['rotation']} ---")
                for line in rnd["prompt"].split("\n"):
                    print(f"    {line}")
            if len(p["stage2_rounds"]) > args.show_prompts:
                print(f"\n  ... {len(p['stage2_rounds']) - args.show_prompts} "
                      f"more prompts not shown (--show-prompts)")
        print("\nNo model was loaded. Drop --dry-run and pass --model to score.")
        return

    if not args.model:
        ap.error("--model is required unless --dry-run")

    ranker = GeneRanker(args.model, dtype=args.dtype,
                        batch_size=args.batch_size, hgnc=args.hgnc)
    results = []
    for d in diseases:
        r = ranker.rank(d, genes, top_k=args.top_k,
                        group_size=args.group_size, rotations=args.rotations,
                        margin_threshold=args.margin_threshold,
                        fewshot=not args.no_fewshot,
                        neutral_cache=args.neutral_cache)
        results.append(r)
        print(f"  {d}: {r['call'] or 'ABSTAIN'} "
              f"(margin {r['margin']}, stability {r['rank_stability']})")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")
    print("Run evaluate.py --controls before drawing any conclusion.")


if __name__ == "__main__":
    main()
