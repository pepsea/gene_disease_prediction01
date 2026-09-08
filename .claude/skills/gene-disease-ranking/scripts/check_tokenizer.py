#!/usr/bin/env python3
"""Show how gene symbols tokenize and verify single-token MCQ labels.

Run this before anything else. It takes seconds and catches two silent failures:
symbols sharing long token prefixes (the family-confusion bug), and MCQ labels
that are not single tokens for this tokenizer.

  python check_tokenizer.py --model google/gemma-3-27b-it --genes GPR52 GPR56
"""
import argparse
from itertools import combinations


def shared_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--genes", nargs="+",
                    default=["GPR52", "GPR56", "SLC6A4", "SLC6A3", "TP53"])
    ap.add_argument("--labels", nargs="+", default=list("ABCDE"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"model      : {args.model}")
    print(f"vocab size : {len(tok)}")
    print()

    print("=== gene symbol tokenization ===")
    enc = {}
    for g in args.genes:
        ids = tok.encode(g, add_special_tokens=False)
        enc[g] = ids
        print(f"  {g:<12} {len(ids)} tok  {tok.convert_ids_to_tokens(ids)}")

    print()
    print("=== shared prefixes (higher = more confusable when generating) ===")
    risky = 0
    for a, b in combinations(args.genes, 2):
        n = shared_prefix_len(enc[a], enc[b])
        if n == 0:
            continue
        risky += 1
        print(f"  {a} / {b}: first {n} token(s) identical "
              f"-> {tok.convert_ids_to_tokens(enc[a][:n])}")
    if risky == 0:
        print("  none among the symbols given")
    else:
        print()
        print("  Any nonzero value means free generation can drift between these")
        print("  symbols. Score them independently instead (score_pmi.py).")

    print()
    print("=== MCQ label check (leading space form, as used after 'Answer:') ===")
    ok = True
    for lab in args.labels:
        ids = tok.encode(" " + lab, add_special_tokens=False)
        flag = "OK" if len(ids) == 1 else "NOT SINGLE TOKEN"
        if len(ids) != 1:
            ok = False
        print(f"  ' {lab}' -> {ids} {tok.convert_ids_to_tokens(ids)}  [{flag}]")

    if not ok:
        print()
        print("  Some labels are multi-token. score_labels.py uses the LAST token")
        print("  id, which still works, but check that the last tokens differ")
        print("  across labels before trusting the scores.")


if __name__ == "__main__":
    main()
