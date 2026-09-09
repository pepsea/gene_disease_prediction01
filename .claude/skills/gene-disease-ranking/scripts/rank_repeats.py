#!/usr/bin/env python3
"""Average the stage 2 ranking over repeats that reshuffle the candidate order.

  python rank_repeats.py --backend ollama --model M --disease D \
      --genes-file G --repeats 10 --out repeats.json

Repeating the pipeline verbatim is pointless: it decodes at temperature 0 and
reproduces itself exactly. The variance worth measuring is elsewhere.

Stage 2 never compares all candidates at once -- `build_mcq_rounds` cuts the list
into groups of four and each gene is scored only against its own group-mates.
Which three genes those are is decided by nothing but the input order, so a gene
grouped with three obscure ones scores well and the same gene grouped with DRD2
scores badly. Rotation controls the position of an option inside a group; it does
nothing about the composition of the group itself.

So each repeat reshuffles the candidate order, and the reported score is the mean
across repeats. The spread across repeats is reported alongside it, because it is
the honest measure of how much of a single run's ranking was an artefact of the
grouping. Repeat 0 keeps the original order, which makes it directly comparable
to a plain rank.py run.
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rank import GeneRanker  # noqa: E402


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def repeat_rank(ranker, disease, genes, repeats=10, group_size=4, rotations=4,
                seed=0, progress=None):
    """Run the stage 2 ranking `repeats` times, reshuffling the order each time.

    Repeat 0 keeps the original order, so it reproduces a plain rank.py run.
    `progress`, when given, is called with each run record as it completes --
    a 100-gene run is 100 model calls, so silence for ten minutes is unkind.
    """
    per_run = []
    for i in range(repeats):
        order = list(genes)
        if i > 0:
            random.Random(seed + i).shuffle(order)
        rec = ranker.rank(disease, order, group_size=group_size,
                          rotations=rotations)
        run = {"repeat": i,
               "scores": {r["gene"]: r["score"] for r in rec["ranked"]},
               "call": rec.get("call"),
               "abstain_reason": rec.get("abstain_reason"),
               "margin": rec.get("margin"),
               "top": rec["ranked"][0]["gene"]}
        per_run.append(run)
        if progress:
            progress(run)
    return per_run


def aggregate(per_run, genes):
    """Per-run scores -> one row per gene, ordered by mean score.

    `sd_score` and the rank range are not decoration: they say how much of any
    single run's ordering was decided by which three genes a gene happened to
    share its question with.
    """
    rows = []
    for g in genes:
        vals = [r["scores"][g] for r in per_run if g in r["scores"]]
        if not vals:
            continue
        ranks = []
        for r in per_run:
            if g in r["scores"]:
                order = sorted(r["scores"], key=lambda x: -r["scores"][x])
                ranks.append(order.index(g) + 1)
        rows.append({
            "gene": g, "n_runs": len(vals),
            "mean_score": statistics.mean(vals),
            "sd_score": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "mean_rank": statistics.mean(ranks),
            "best_rank": min(ranks), "worst_rank": max(ranks),
            "rank_sd": statistics.stdev(ranks) if len(ranks) > 1 else 0.0,
        })
    rows.sort(key=lambda r: -r["mean_score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def instability(rows):
    """Summarize how much a single run's ordering can be trusted."""
    if not rows:
        return {}
    spans = [r["worst_rank"] - r["best_rank"] for r in rows]
    return {
        "median_rank_sd": statistics.median(r["rank_sd"] for r in rows),
        "median_rank_span": statistics.median(spans),
        "n_span_over_50": sum(1 for s in spans if s > 50),
        "n_genes": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--genes-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--hgnc", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    genes = read_lines(args.genes_file)
    kw = {"host": args.host} if args.host else {}
    ranker = GeneRanker(args.model, backend=args.backend, hgnc=args.hgnc, **kw)

    def show(run):
        print(f"  repeat {run['repeat']}: top={run['top']:<10} "
              f"margin={run['margin']} call={run['call'] or 'ABSTAIN'}",
              flush=True)

    per_run = repeat_rank(ranker, args.disease, genes, args.repeats,
                          args.group_size, args.rotations, args.seed, show)
    rows = aggregate(per_run, genes)

    result = {"disease": args.disease, "model": args.model,
              "repeats": args.repeats, "n_genes": len(genes),
              "instability": instability(rows),
              "runs": per_run, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\n{json.dumps(result['instability'], indent=1)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
