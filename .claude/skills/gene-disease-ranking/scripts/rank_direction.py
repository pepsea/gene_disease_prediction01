#!/usr/bin/env python3
"""Score the directional prompts and report them against their controls.

  python rank_direction.py --backend ollama --model qwen3:14b \
      --gold ../examples/direction_gold.jsonl --out direction.jsonl

Two forms are run (see direction_prompts.py):

  A  outcome   asked once per direction; the pair is compared, so no gold label
               is invented for the non-therapeutic direction.
  B  choice    gene and direction scored jointly, rotated to expose position bias.

The controls decide whether any of it means anything:

  majority-direction   always answer the commoner direction in the gold set.
                       Drug targets skew heavily to inhibition, so this baseline
                       is high; a model that cannot beat it has learned the skew,
                       not the biology.
  disease-shuffled     re-ask with the disease names permuted. A model reading
                       the disease should collapse toward chance.
  dual-direction       the subset of genes that appear with BOTH directions under
                       different diseases. Gene identity alone cannot answer
                       these, so they are the only items that separate reading
                       the disease from recalling the gene.
"""
import argparse
import collections
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import direction_prompts as dp  # noqa: E402
from backends import match_labels  # noqa: E402


def load_gold(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def make_backend(name, model, host=None):
    if name == "ollama":
        from backends import OllamaBackend, discover_ollama_host
        return OllamaBackend(model, host=host or discover_ollama_host())
    if name == "transformers":
        from backends import TransformersBackend
        return TransformersBackend(model)
    if name == "llamacpp":
        from backends import LlamaCppBackend
        return LlamaCppBackend(model)
    raise SystemExit(f"unknown backend {name!r}")


def softmax_over(scores):
    """Normalize the labels that came back, dropping any the runtime never
    produced. Returns {} when nothing usable arrived, so callers can abstain
    rather than score a fabricated distribution."""
    import math
    got = {k: v for k, v in scores.items() if v is not None}
    if not got:
        return {}
    m = max(got.values())
    ex = {k: math.exp(v - m) for k, v in got.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def run_outcome(be, rows, fewshot=True):
    """Form A, both directions per row."""
    out = []
    for r in rows:
        rec = {"gene": r["gene"], "disease": r["disease"],
               "gold_direction": r["direction"], "probs": {}}
        for direction in dp.DIRECTIONS:
            prompt = dp.build_outcome_prompt(r["disease"], r["gene"], direction,
                                             fewshot=fewshot)
            raw = be.label_logprobs(prompt, dp.LABELS[:len(dp.OUTCOMES)])
            p = softmax_over(raw)
            rec["probs"][direction] = {
                "p_improves": p.get("A"), "p_worsens": p.get("B"),
                "p_unclear": p.get("C"),
                "top": max(p, key=p.get) if p else None,
            }
        out.append(rec)
    return out


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p). Exact, stdlib only -- the sample sizes
    here are far too small for a normal approximation to be honest."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def bootstrap_ci(hits, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a mean of 0/1 outcomes."""
    if not hits:
        return None
    rng = random.Random(seed)
    n = len(hits)
    means = sorted(sum(rng.choice(hits) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


def accuracy_stats(hits, chance, label, seed=0):
    """Point estimate, bootstrap CI, and an exact one-sided test against chance."""
    n, k = len(hits), sum(hits)
    if n == 0:
        return None
    return {
        "label": label, "k": k, "n": n, "accuracy": k / n,
        "ci95_bootstrap": bootstrap_ci(hits, seed=seed),
        "chance": chance,
        "p_value_vs_chance_exact": binom_sf(k, n, chance),
        "significant_at_0.05": binom_sf(k, n, chance) < 0.05,
    }


def load_pinned(path):
    """(gene, disease) -> distractor, from a previous run's output.

    Re-running an extended gold set changes the distractor pool, so a plain
    re-run mixes two effects: the model's variance and a different question.
    Pinning the old items' distractors makes the old subset an actual
    replication rather than a new experiment wearing the same name.
    """
    prev = json.load(open(path))
    return {(c["gene"], c["disease"]): c["distractor"]
            for c in prev.get("choice", [])}


def run_choice(be, rows, rotations=4, seed=0, fewshot=True, pinned=None):
    """Form B, rotated. The distractor is another gold gene drawn from a
    different disease, so it is a real gene with real associations -- an easier
    distractor would flatter the result."""
    rng = random.Random(seed)
    pinned = pinned or {}
    out = []
    for i, r in enumerate(rows):
        pool = [x for x in rows if x["gene"] != r["gene"]
                and x["disease"] != r["disease"]]
        drawn = rng.choice(pool)["gene"]          # drawn regardless, to keep the
        key = (r["gene"], r["disease"])           # RNG stream independent of pins
        distractor = pinned.get(key, drawn)
        gold = (r["direction"], r["gene"])

        votes = collections.Counter()
        mass = collections.defaultdict(float)
        per_rotation = []
        for rd in dp.build_choice_rounds(r["disease"], r["gene"], distractor,
                                         rotations=rotations, fewshot=fewshot):
            raw = be.label_logprobs(rd["prompt"], dp.LABELS)
            p = softmax_over(raw)
            if not p:
                per_rotation.append({"rotation": rd["rotation"], "pick": None})
                continue
            top = max(p, key=p.get)
            pick = rd["label_map"][top]
            votes[pick] += 1
            for lab, prob in p.items():
                mass[rd["label_map"][lab]] += prob / rotations
            per_rotation.append({"rotation": rd["rotation"],
                                 "pick": pick, "p": round(p[top], 4)})

        winner = max(mass, key=mass.get) if mass else None
        out.append({
            "gene": r["gene"], "disease": r["disease"], "distractor": distractor,
            "distractor_pinned": key in pinned,
            "gold": gold, "winner": winner,
            "correct_joint": winner == gold,
            "correct_gene": bool(winner) and winner[1] == r["gene"],
            "correct_direction_given_gene":
                (winner[0] == r["direction"]) if winner and winner[1] == r["gene"] else None,
            "stability": (votes[winner] / rotations) if winner else 0.0,
            "mass": {f"{d} {g}" if x else "none": round(v, 4)
                     for x, v in mass.items() for d, g in [x or ("", "")]},
            "rotations": per_rotation,
        })
    return out


def summarize(gold, outcome, choice, subsets=None):
    n = len(gold)
    dirs = collections.Counter(r["direction"] for r in gold)
    majority, maj_n = dirs.most_common(1)[0]

    # Form A: absolute, and the paired comparison that needs no invented gold.
    a_top = sum(1 for r in outcome
                if r["probs"][r["gold_direction"]]["top"] == "A")
    paired, paired_ok = 0, 0
    for r in outcome:
        gd = r["gold_direction"]
        wrong = "activate" if gd == "inhibit" else "inhibit"
        a, b = (r["probs"][gd]["p_improves"], r["probs"][wrong]["p_improves"])
        if a is None or b is None:
            continue
        paired += 1
        paired_ok += a > b

    # Form B
    joint = sum(c["correct_joint"] for c in choice)
    gene_ok = sum(c["correct_gene"] for c in choice)
    dir_given = [c["correct_direction_given_gene"] for c in choice
                 if c["correct_direction_given_gene"] is not None]

    dual_genes = {g for g, c in collections.Counter(
        r["gene"] for r in gold).items() if c > 1}
    dual_idx = [i for i, r in enumerate(gold) if r["gene"] in dual_genes]
    dual_joint = sum(choice[i]["correct_joint"] for i in dual_idx)

    # Statistics on the joint task. Chance is 1/5: four interventions plus the
    # exit option. The direction sub-task has its own chance of 1/2.
    joint_hits = [1 if c["correct_joint"] else 0 for c in choice]
    dir_hits = [1 if c["correct_direction_given_gene"] else 0 for c in choice
                if c["correct_direction_given_gene"] is not None]
    stats = {
        "joint_vs_chance": accuracy_stats(joint_hits, 0.2, "joint (all rows)"),
        "direction_given_gene_vs_coinflip":
            accuracy_stats(dir_hits, 0.5, "direction | gene correct"),
        "direction_vs_majority_baseline":
            accuracy_stats(dir_hits, maj_n / n, "direction | gene correct"),
    }
    if subsets:
        stats["subsets"] = {}
        for name, idx in subsets.items():
            hits = [1 if choice[i]["correct_joint"] else 0 for i in idx]
            stats["subsets"][name] = accuracy_stats(hits, 0.2, name)

    return {
        "n_pairs": n,
        "gold_direction_counts": dict(dirs),
        "baseline_majority_direction": {"direction": majority,
                                        "accuracy": maj_n / n},
        "form_a_outcome": {
            "improves_on_correct_direction": a_top / n,
            "paired_discrimination": (paired_ok / paired) if paired else None,
            "n_paired": paired,
        },
        "form_b_choice": {
            "joint_accuracy": joint / n,
            "gene_accuracy": gene_ok / n,
            "direction_accuracy_given_gene":
                (sum(dir_given) / len(dir_given)) if dir_given else None,
            "chance_joint": 1 / 5,
            "mean_rotation_stability":
                sum(c["stability"] for c in choice) / n,
        },
        "dual_direction_subset": {
            "genes": sorted(dual_genes), "n": len(dual_idx),
            "joint_accuracy": (dual_joint / len(dual_idx)) if dual_idx else None,
        },
        "statistics": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--no-fewshot", action="store_true")
    ap.add_argument("--pin-distractors", default=None,
                    help="previous run JSON; reuse its distractors where the "
                         "(gene, disease) matches, making that subset a "
                         "replication rather than a new question")
    ap.add_argument("--subset-genes", default=None,
                    help="comma-separated genes to report as the 'previous' "
                         "subset, the rest as 'new'")
    ap.add_argument("--shuffle-control", action="store_true",
                    help="also run form A with disease names permuted")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    be = make_backend(args.backend, args.model, args.host)
    fewshot = not args.no_fewshot
    print(f"{len(gold)} gold pairs, backend={args.backend}, model={args.model}")

    print("form A (outcome) ...", flush=True)
    outcome = run_outcome(be, gold, fewshot)
    print("form B (choice) ...", flush=True)
    pinned = load_pinned(args.pin_distractors) if args.pin_distractors else None
    if pinned:
        n_hit = sum((r["gene"], r["disease"]) in pinned for r in gold)
        print(f"  pinned distractors reused for {n_hit}/{len(gold)} rows")
    choice = run_choice(be, gold, args.rotations, args.seed, fewshot, pinned)

    subsets = None
    if args.subset_genes:
        prev = {g.strip() for g in args.subset_genes.split(",") if g.strip()}
        subsets = {
            "previous_genes": [i for i, r in enumerate(gold) if r["gene"] in prev],
            "new_genes": [i for i, r in enumerate(gold) if r["gene"] not in prev],
        }

    result = {"model": args.model, "backend": args.backend,
              "n_pairs": len(gold), "fewshot": fewshot,
              "summary": summarize(gold, outcome, choice, subsets),
              "outcome": outcome, "choice": choice}

    if args.shuffle_control:
        print("control: disease-shuffled form A ...", flush=True)
        rng = random.Random(args.seed)
        idx = list(range(len(gold)))
        while True:
            rng.shuffle(idx)
            if all(i != j for i, j in enumerate(idx)):
                break
        shuffled = [dict(gold[i], disease=gold[j]["disease"])
                    for i, j in enumerate(idx)]
        sh = run_outcome(be, shuffled, fewshot)
        paired, ok = 0, 0
        for r in sh:
            gd = r["gold_direction"]
            wrong = "activate" if gd == "inhibit" else "inhibit"
            a, b = r["probs"][gd]["p_improves"], r["probs"][wrong]["p_improves"]
            if a is None or b is None:
                continue
            paired += 1
            ok += a > b
        result["summary"]["control_disease_shuffled"] = {
            "paired_discrimination": (ok / paired) if paired else None,
            "n_paired": paired,
        }
        result["outcome_shuffled"] = sh

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
