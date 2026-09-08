#!/usr/bin/env python3
"""Normalize gene symbols against an HGNC dump. Offline, no network.

Removes symbols that are not genes and collapses aliases onto approved symbols,
so the same gene is not scored twice under two names.

This does NOT fix family confusion: GPR56 is a perfectly valid approved symbol,
just the wrong one. Only scoring symbols rather than generating them fixes that.

  python normalize_genes.py --hgnc hgnc_complete_set.txt \
      --input raw_genes.txt --out clean_genes.txt --report report.json

HGNC dump: the tab-delimited complete set from genenames.org, which includes the
symbol, alias_symbol and prev_symbol columns.
"""
import argparse
import csv
import json
import sys


def load_hgnc(path):
    approved, alias_to = set(), {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        if "symbol" not in cols:
            sys.exit(f"no 'symbol' column in {path}; columns: {cols[:10]}")
        for row in reader:
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            approved.add(sym.upper())
            for col in ("alias_symbol", "prev_symbol"):
                raw = (row.get(col) or "").strip().strip('"')
                for a in filter(None, (x.strip() for x in raw.split("|"))):
                    alias_to.setdefault(a.upper(), sym)
    return approved, alias_to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hgnc", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--keep-unknown", action="store_true",
                    help="keep unrecognized symbols (default: drop)")
    args = ap.parse_args()

    approved, alias_to = load_hgnc(args.hgnc)
    print(f"HGNC: {len(approved)} approved symbols, {len(alias_to)} aliases")

    with open(args.input) as f:
        raw = [l.strip() for l in f
               if l.strip() and not l.strip().startswith("#")]

    kept, seen = [], set()
    mapped, unknown, dupes = [], [], []
    for g in raw:
        u = g.upper()
        if u in approved:
            canon = u
        elif u in alias_to:
            canon = alias_to[u]
            mapped.append({"input": g, "approved": canon})
        else:
            unknown.append(g)
            if not args.keep_unknown:
                continue
            canon = g
        if canon in seen:
            dupes.append(g)
            continue
        seen.add(canon)
        kept.append(canon)

    with open(args.out, "w") as f:
        f.write("\n".join(kept) + "\n")

    print(f"in {len(raw)} -> out {len(kept)}")
    print(f"  aliases mapped : {len(mapped)}")
    print(f"  unknown        : {len(unknown)}"
          f"{' (kept)' if args.keep_unknown else ' (dropped)'}")
    print(f"  duplicates     : {len(dupes)}")
    if unknown[:5]:
        print(f"  e.g. unknown   : {', '.join(unknown[:5])}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"n_input": len(raw), "n_output": len(kept),
                       "mapped": mapped, "unknown": unknown,
                       "duplicates": dupes}, f, indent=2)
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
