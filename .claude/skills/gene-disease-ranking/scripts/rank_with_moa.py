#!/usr/bin/env python3
"""Join the stage 2 gene ranking to a per-gene inhibit/activate call.

  python rank.py --backend ollama --model M --disease D --genes-file G --out r.jsonl
  python rank_with_moa.py --backend ollama --model M --disease D \
      --ranking r.jsonl --genes-file G --out moa.json

Ranking answers "which gene", `direction_prompts` answers "which way". Reporting
one without the other is how a pipeline ends up recommending the right gene and
the wrong intervention.

The direction call reuses form B (`build_choice_rounds`) unchanged, but reads it
differently. Form B scores gene and direction jointly and asks who wins overall;
here the gene is already fixed by the ranking, so each rotation is reduced to the
one comparison that matters -- the mass on (inhibit, gene) against (activate,
gene) -- and the four rotations become four independent votes. Stability is the
share of rotations agreeing with the modal direction: 1.00 means the option order
never moved the answer, 0.50 means it decided it.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import direction_prompts as dp  # noqa: E402
from rank_direction import make_backend, softmax_over  # noqa: E402


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def direction_for_gene(be, disease, gene, distractor, rotations=4, fewshot=True):
    """Four rotations -> a direction, its stability, and the per-rotation trace."""
    votes = collections.Counter()
    mass = collections.Counter()
    trace = []
    for rd in dp.build_choice_rounds(disease, gene, distractor,
                                     rotations=rotations, fewshot=fewshot):
        p = softmax_over(be.label_logprobs(rd["prompt"], dp.LABELS))
        if not p:
            trace.append({"rotation": rd["rotation"], "pick": None})
            continue
        # collapse this rotation onto the target gene only
        own = {d: 0.0 for d in dp.DIRECTIONS}
        for lab, prob in p.items():
            opt = rd["label_map"][lab]
            if opt and opt[1] == gene:
                own[opt[0]] += prob
        pick = max(own, key=own.get)
        total = sum(own.values())
        votes[pick] += 1
        for d, v in own.items():
            mass[d] += v / rotations
        trace.append({"rotation": rd["rotation"], "pick": pick,
                      "p_inhibit": round(own["inhibit"], 4),
                      "p_activate": round(own["activate"], 4),
                      "share_on_this_gene": round(total, 4)})
    if not votes:
        return {"direction": None, "stability": 0.0, "trace": trace}
    direction, n = votes.most_common(1)[0]
    return {
        "direction": direction,
        "stability": n / rotations,
        "votes": dict(votes),
        "mean_p_inhibit": round(mass["inhibit"], 4),
        "mean_p_activate": round(mass["activate"], 4),
        "margin": round(abs(mass["inhibit"] - mass["activate"]), 4),
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--disease", required=True)
    ap.add_argument("--genes-file", required=True)
    ap.add_argument("--ranking", help="rank.py output jsonl, to join on")
    ap.add_argument("--gold", help='JSON {"GENE": {"verdict": "inhibit"}}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    genes = read_lines(args.genes_file)
    be = make_backend(args.backend, args.model, args.host)

    ranking = {}
    abstained = None
    if args.ranking:
        for line in open(args.ranking):
            rec = json.loads(line)
            if rec["disease"].lower() != args.disease.lower():
                continue
            ranking = {r["gene"]: r["score"] for r in rec["ranked"]}
            abstained = {"call": rec.get("call"),
                         "reason": rec.get("abstain_reason"),
                         "margin": rec.get("margin"),
                         "rank_stability": rec.get("rank_stability")}

    gold = json.load(open(args.gold)) if args.gold else {}

    rng = random.Random(args.seed)
    rows = []
    for g in genes:
        distractor = rng.choice([x for x in genes if x != g])
        d = direction_for_gene(be, args.disease, g, distractor,
                               args.rotations)
        gv = (gold.get(g) or {}).get("verdict")
        rows.append({
            "gene": g,
            "rank_score": ranking.get(g),
            "direction": d["direction"],
            "stability": d["stability"],
            "margin": d.get("margin"),
            "distractor": distractor,
            "gold_direction": gv,
            "agrees_with_gold": (None if gv in (None, "MIXED")
                                 else d["direction"] == gv),
            "detail": d,
        })

    rows.sort(key=lambda r: (r["rank_score"] is None, -(r["rank_score"] or 0)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    checked = [r for r in rows if r["agrees_with_gold"] is not None]
    result = {
        "disease": args.disease, "model": args.model, "backend": args.backend,
        "n_genes": len(rows), "ranking_call": abstained,
        "gold_agreement": {
            "n_with_gold": len(checked),
            "n_agree": sum(r["agrees_with_gold"] for r in checked),
            "accuracy": (sum(r["agrees_with_gold"] for r in checked) / len(checked))
                        if checked else None,
        },
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\n{args.disease}: {len(rows)} genes, "
          f"ranking call = {abstained['call'] if abstained else 'n/a'}")
    print(f"{'#':<4}{'gene':<10}{'rank score':>11}  {'MOA':<9}{'stab':>6}"
          f"{'margin':>8}  {'gold':<9}{'agree'}")
    for r in rows:
        s = "-" if r["rank_score"] is None else f"{r['rank_score']:.4f}"
        a = {True: "yes", False: "NO", None: "-"}[r["agrees_with_gold"]]
        print(f"{r['rank']:<4}{r['gene']:<10}{s:>11}  {str(r['direction']):<9}"
              f"{r['stability']:>6.2f}{(r['margin'] or 0):>8.3f}  "
              f"{str(r['gold_direction'] or '-'):<9}{a}")
    ga = result["gold_agreement"]
    print(f"\ndirection vs drug-derived gold: {ga['n_agree']}/{ga['n_with_gold']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
