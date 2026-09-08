#!/usr/bin/env python3
"""Evaluate a ranking against gold pairs, with the controls that make the
numbers interpretable.

  python evaluate.py --pred stage1.jsonl --gold gold.jsonl --k 10 50 --controls

gold.jsonl: {"disease": "...", "genes": ["HTT"]}  (list = any is correct)
pred.jsonl: output of score_pmi.py or score_labels.py
"""
import argparse
import json
import re

import numpy as np

FAMILY_RE = re.compile(r"^([A-Za-z\-]+)")


def family(sym):
    """Leading alphabetic run: GPR52 -> GPR, SLC6A4 -> SLC, HLA-DRB1 -> HLA-DRB."""
    m = FAMILY_RE.match(sym)
    return m.group(1).upper() if m else sym.upper()


def confusable(a, b, min_prefix=3):
    """Would free generation plausibly drift from `a` to `b`?

    Two rules, because neither alone is enough. Identical alphabetic stems catch
    GPR52/GPR56 and SLC6A4/SLC6A3. A shared character prefix catches families
    whose stems differ only in a trailing letter, like HBA1/HBB.

    Heuristic, and deliberately conservative: it undercounts families with stems
    shorter than `min_prefix`. Treat the number as a floor, and spot-check the
    actual error list rather than trusting it alone.
    """
    a, b = a.upper(), b.upper()
    if a == b:
        return False
    if family(a) == family(b):
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= min_prefix or (n >= 2 and min(len(a), len(b)) <= 4)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def ranked_genes(rec):
    return [r["gene"] for r in rec["ranked"]]


def evaluate(preds, gold, ks):
    gold_map = {g["disease"]: set(g["genes"]) for g in gold}
    hits = {k: 0 for k in ks}
    rr, n = [], 0
    fam_conf = top1_wrong = 0
    n_cand = []

    for rec in preds:
        truth = gold_map.get(rec["disease"])
        if not truth:
            continue
        n += 1
        order = ranked_genes(rec)
        n_cand.append(len(order))

        pos = next((i for i, g in enumerate(order) if g in truth), None)
        rr.append(1.0 / (pos + 1) if pos is not None else 0.0)
        for k in ks:
            if pos is not None and pos < k:
                hits[k] += 1

        if order and order[0] not in truth:
            top1_wrong += 1
            if any(confusable(order[0], t) for t in truth):
                fam_conf += 1

    if n == 0:
        raise SystemExit("no overlap between predictions and gold")

    return {
        "n_diseases": n,
        "mean_candidates": float(np.mean(n_cand)),
        "recall": {f"@{k}": hits[k] / n for k in ks},
        "mrr": float(np.mean(rr)),
        "top1_accuracy": 1.0 - top1_wrong / n,
        "family_confusion_rate": fam_conf / top1_wrong if top1_wrong else 0.0,
        "family_confusion_count": fam_conf,
        "top1_wrong_count": top1_wrong,
    }


def random_baseline(n_candidates, ks):
    """Expected recall@k for a uniformly random ranking with one true gene."""
    return {f"@{k}": min(1.0, k / n_candidates) for k in ks}


def frequency_baseline(preds, gold, ks, freq_path):
    """Rank by disease-independent frequency. If the model cannot beat this,
    it is reproducing corpus frequency rather than reading the disease."""
    with open(freq_path) as f:
        freq = json.load(f)
    fake = []
    for rec in preds:
        genes = ranked_genes(rec)
        order = sorted(genes, key=lambda g: -freq.get(g, 0))
        fake.append({"disease": rec["disease"],
                     "ranked": [{"gene": g} for g in order]})
    return evaluate(fake, gold, ks)


def shuffle_control(preds, gold, ks, seed=0):
    """Reassign each disease's ranking to a different disease. Scores should
    collapse toward random; if they do not, the ranking is disease-blind."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(preds))
    if len(preds) > 1:
        while any(i == j for i, j in enumerate(idx)):
            idx = rng.permutation(len(preds))
    shuffled = [{"disease": preds[i]["disease"], "ranked": preds[j]["ranked"]}
                for i, j in enumerate(idx)]
    return evaluate(shuffled, gold, ks)


def show(name, m, ks):
    rec = "  ".join(f"R{k}={m['recall'][f'@{k}']:.3f}" for k in ks)
    print(f"  {name:<22} {rec}  MRR={m['mrr']:.3f}  top1={m['top1_accuracy']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 10, 50])
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--freq", default=None,
                    help='JSON {"GENE": count} for the frequency baseline')
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    preds = load_jsonl(args.pred)
    gold = load_jsonl(args.gold)
    main_m = evaluate(preds, gold, args.k)

    print(f"\nevaluated {main_m['n_diseases']} diseases, "
          f"mean {main_m['mean_candidates']:.0f} candidates each\n")

    print("=== main ===")
    show("system", main_m, args.k)
    print(f"\n  Family Confusion Rate: {main_m['family_confusion_rate']:.3f}  "
          f"({main_m['family_confusion_count']}/{main_m['top1_wrong_count']} "
          f"top-1 errors share a symbol prefix with the true gene)")
    if main_m["family_confusion_rate"] > 0.1:
        print("  -> Still high. Confirm symbols are scored, never generated.")

    out = {"main": main_m}

    if args.controls:
        print("\n=== controls ===")
        rnd = random_baseline(main_m["mean_candidates"], args.k)
        print("  " + f"{'random':<22} " +
              "  ".join(f"R{k}={rnd[f'@{k}']:.3f}" for k in args.k))
        out["random"] = rnd

        sh = shuffle_control(preds, gold, args.k)
        show("disease-shuffled", sh, args.k)
        out["disease_shuffled"] = sh
        if sh["mrr"] > 0.5 * main_m["mrr"]:
            print("  -> Shuffling barely hurt. The ranking is largely "
                  "disease-independent; check the PMI subtraction.")

        if args.freq:
            fq = frequency_baseline(preds, gold, args.k, args.freq)
            show("frequency-only", fq, args.k)
            out["frequency_only"] = fq
            if fq["mrr"] >= main_m["mrr"]:
                print("  -> Frequency alone matches or beats the model. "
                      "The LLM is adding nothing here.")
        else:
            print("  (frequency baseline skipped: pass --freq)")

        print("\n  Not computed here: the database-only baseline. Score the same")
        print("  gold set with Open Targets association scores and compare. If")
        print("  the LLM does not beat it, report that plainly.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
