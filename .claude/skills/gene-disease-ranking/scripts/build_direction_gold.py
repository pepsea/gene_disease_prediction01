#!/usr/bin/env python3
"""Build the directional gold set from Open Targets / ChEMBL mechanism-of-action
records. Every row's direction comes from a curated `actionType`, never from the
author's memory.

Two sets come out:

  direction_gold.jsonl           the base set, one row per (gene, disease)
  direction_gold_reversal.jsonl  only genes that appear with BOTH directions
                                 under DIFFERENT diseases

The reversal set is the one that carries information. Everywhere else a model can
answer from the gene alone -- oncogenes are inhibited, hormone deficiencies are
replaced -- so a high score there is consistent with pure gene recall. A gene
that is activated for one disease and inhibited for another cannot be answered
without reading the disease.

Rows are rejected, never patched, when:
  - the designated gene is not among the drug's on-targets
  - the drug's action types mix inhibition and activation
  - the disease is absent from that drug's own indication table
  - (reversal set) the gene lacks two directions, or its two directions share a
    disease, which makes it a mechanism disagreement rather than a reversal

Multi-target drugs are flagged rather than dropped: attributing one indication to
one member of a receptor family is a judgement call, and the reader should see
which rows rest on one.
"""
import argparse
import collections
import json

INHIBIT = {"INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR",
           "RNAI INHIBITOR", "ANTISENSE INHIBITOR", "DEGRADER", "INVERSE AGONIST",
           "ALLOSTERIC ANTAGONIST"}
ACTIVATE = {"AGONIST", "ACTIVATOR", "POSITIVE MODULATOR",
            "POSITIVE ALLOSTERIC MODULATOR", "PARTIAL AGONIST", "OPENER"}

# (drug, designated gene, disease id). The disease id is checked against the
# drug's own indication table below; nothing here is trusted on its own.
BASE = [
    ("ivacaftor",     "CFTR",   "MONDO_0009061"),
    ("semaglutide",   "GLP1R",  "MONDO_0005148"),
    ("salbutamol",    "ADRB2",  "MONDO_0004979"),
    ("morphine",      "OPRM1",  "HP_0012531"),
    ("dexamethasone", "NR3C1",  "MONDO_0004979"),
    ("pioglitazone",  "PPARG",  "MONDO_0005148"),
    ("setmelanotide", "MC4R",   "MONDO_0011122"),
    ("pramipexole",   "DRD2",   "MONDO_0005180"),
    ("diazepam",      "GABRA1", "MONDO_0005027"),
    ("estradiol",     "ESR1",   "GO_0042697"),
    ("nusinersen",    "SMN2",   "MONDO_0001516"),
    ("osimertinib",   "EGFR",   "MONDO_0005233"),
    ("trastuzumab",   "ERBB2",  "MONDO_0007254"),
    ("adalimumab",    "TNF",    "MONDO_0008383"),
    ("evolocumab",    "PCSK9",  "MONDO_0005439"),
    ("atorvastatin",  "HMGCR",  "HP_0003124"),
    ("fluoxetine",    "SLC6A4", "MONDO_0002009"),
    ("dapagliflozin", "SLC5A2", "MONDO_0005148"),
    ("allopurinol",   "XDH",    "MONDO_0005393"),
    ("patisiran",     "TTR",    "MONDO_0018634"),
    ("haloperidol",   "DRD2",   "MONDO_0005090"),
    ("naltrexone",    "OPRM1",  "MONDO_0007079"),
    ("ibrutinib",     "BTK",    "MONDO_0004948"),
    ("olaparib",      "PARP1",  "MONDO_0008170"),
    ("metoprolol",    "ADRB1",  "HP_0000822"),
    ("tominersen",    "HTT",    "MONDO_0007739"),
    ("vemurafenib",   "BRAF",   "MONDO_0005105"),
    ("fulvestrant",   "ESR1",   "MONDO_0007254"),
]

# Added to complete a reversal: each supplies the direction its gene was missing.
REVERSAL_EXTRA = [
    ("mifepristone",  "NR3C1",  "MONDO_0018912"),  # Cushing syndrome
    ("testosterone",  "AR",     "MONDO_0002146"),  # hypogonadism
    ("enzalutamide",  "AR",     "MONDO_0008315"),  # prostate cancer
    ("dobutamine",    "ADRB1",  "MONDO_0005252"),  # heart failure
    ("midodrine",     "ADRA1A", "MONDO_0005469"),  # orthostatic hypotension
    ("tamsulosin",    "ADRA1A", "MONDO_0010811"),  # benign prostatic hyperplasia
    ("diazoxide",     "KCNJ11", "MONDO_0002177"),  # hyperinsulinism
    ("glibenclamide", "KCNJ11", "MONDO_0005148"),  # type 2 diabetes
    ("pilocarpine",   "CHRM3",  "EFO_0009869"),    # xerostomia
    ("solifenacin",   "CHRM3",  "MONDO_0006624"),  # overactive bladder
    ("timolol",       "ADRB2",  "MONDO_0005041"),  # glaucoma
    ("desmopressin",  "AVPR2",  "MONDO_0004782"),  # diabetes insipidus
    ("tolvaptan",     "AVPR2",  "HP_0002902"),     # hyponatremia
]

# Candidates checked and NOT used, kept here so the rejections stay visible.
REJECTED_CANDIDATES = {
    "PGR": "mifepristone is the only approved PGR antagonist here, and its "
           "APPROVAL indication (Cushing syndrome) is a glucocorticoid-receptor "
           "effect, not a progesterone-receptor one. Attributing it to PGR "
           "would be a fabricated direction.",
    "GABRA1": "flumazenil (ALLOSTERIC ANTAGONIST) carries no APPROVAL-stage "
              "indication in the source; its only row is 'sedation' at UNKNOWN "
              "stage.",
    "ADORA2A": "regadenoson (AGONIST) is a diagnostic stress agent; its only "
               "indication row is PHASE_1 and it treats no disease.",
    "GNRHR": "leuprolide is typed AGONIST but works by receptor "
             "downregulation, and both directions target prostate cancer. The "
             "action type and the therapeutic effect point opposite ways.",
    "VEGFA": "no approved VEGFA activator exists; inhibition only.",
    "TNF": "no approved TNF activator exists; inhibition only.",
}


def direction_of(action_types):
    inh, act = action_types & INHIBIT, action_types & ACTIVATE
    if inh and act:
        return None, f"mixed action types {sorted(action_types)}"
    if not (inh or act):
        return None, f"unmapped action types {sorted(action_types)}"
    return ("inhibit" if inh else "activate"), None


def build_rows(raw, pairs):
    rows, problems = [], []
    for drug, gene, disease_id in pairs:
        v = raw.get(drug)
        if v is None:
            problems.append(f"{drug}: not in the mechanism cache")
            continue
        types = set(v["moa"]["uniqueActionTypes"])
        on_targets = sorted({t["approvedSymbol"]
                             for r in v["moa"]["rows"] for t in r["targets"]})
        if gene not in on_targets:
            problems.append(f"{drug}: {gene} not among on-targets {on_targets}")
            continue
        direction, err = direction_of(types)
        if err:
            problems.append(f"{drug}: {err}")
            continue
        ind = {i: (st, n) for st, i, n in v["indications"]}
        if disease_id not in ind:
            problems.append(f"{drug}: {disease_id} not in indications")
            continue
        stage, disease_name = ind[disease_id]
        rows.append({
            "gene": gene, "disease": disease_name, "disease_id": disease_id,
            "direction": direction,
            "evidence": {
                "drug": v["name"], "chembl_id": v["chembl"],
                "drug_type": v["drugType"], "action_types": sorted(types),
                "indication_max_stage": stage, "drug_max_stage": v["stage"],
                "on_targets": on_targets,
                "target_attribution": "single" if len(on_targets) == 1
                                      else "designated",
                "source": "Open Targets Platform / ChEMBL mechanism of action",
            },
        })
    return rows, problems


def reversal_subset(rows):
    """Keep only genes carrying both directions under different diseases."""
    by_gene = collections.defaultdict(list)
    for r in rows:
        by_gene[r["gene"]].append(r)
    keep, dropped = [], []
    for gene, rs in by_gene.items():
        dirs = {r["direction"] for r in rs}
        if len(dirs) < 2:
            continue
        # one row per direction; if a direction has several, take the first
        chosen = {}
        for r in rs:
            chosen.setdefault(r["direction"], r)
        a, b = chosen["activate"], chosen["inhibit"]
        if a["disease_id"] == b["disease_id"]:
            dropped.append(f"{gene}: both directions target {a['disease']}")
            continue
        keep.extend([a, b])
    return keep, dropped


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def report(name, rows, problems):
    print(f"\n=== {name}: {len(rows)} rows, {len(problems)} rejected ===")
    for p in problems:
        print("  REJECTED:", p)
    print("  direction:", dict(collections.Counter(r["direction"] for r in rows)))
    stages = collections.Counter(r["evidence"]["indication_max_stage"] for r in rows)
    print("  indication stage:", dict(stages))
    attr = collections.Counter(r["evidence"]["target_attribution"] for r in rows)
    print("  target attribution:", dict(attr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moa", default="moa_raw.json")
    ap.add_argument("--out-base", default="direction_gold.jsonl")
    ap.add_argument("--out-reversal", default="direction_gold_reversal.jsonl")
    args = ap.parse_args()

    raw = json.load(open(args.moa))

    base, base_problems = build_rows(raw, BASE)
    write(args.out_base, base)
    report("base set", base, base_problems)

    allrows, all_problems = build_rows(raw, BASE + REVERSAL_EXTRA)
    rev, rev_dropped = reversal_subset(allrows)
    rev.sort(key=lambda r: (r["gene"], r["direction"]))
    write(args.out_reversal, rev)
    report("reversal set", rev, all_problems + rev_dropped)

    genes = sorted({r["gene"] for r in rev})
    print(f"\n  reversal genes ({len(genes)}): {genes}")
    print("\n  candidates checked and not used:")
    for g, why in REJECTED_CANDIDATES.items():
        print(f"    {g}: {why}")


if __name__ == "__main__":
    main()
