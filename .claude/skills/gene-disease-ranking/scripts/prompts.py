#!/usr/bin/env python3
"""Turn a disease name and a gene list into the exact prompts the model sees.

Two variables in, prompts out. This module is the single place where a gene
list becomes prompt text, so the dry-run view and the scoring path can never
drift apart.

Nothing here ever asks the model to produce a gene symbol:

  Stage 1  supplies each symbol as a forced continuation and scores it.
  Stage 2  replaces symbols with single-token A-E labels, mapped back to
           symbols in code.

Stdlib only, on purpose: prompts can be inspected on a laptop with no torch,
numpy or model weights installed.
"""

LABELS = list("ABCDE")
NONE_OPT = "None of the above"

# The last label slot is reserved for the exit option, so a group holds at most
# len(LABELS) - 1 genes. Without the exit the model must pick something, and a
# forced guess gets scored as if it were knowledge.
MAX_GENES_PER_GROUP = len(LABELS) - 1

DISEASE_TMPL = "Gene most strongly associated with {disease}: "
NEUTRAL_TMPL = "Gene: "

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


def clean_genes(genes, approved=None, alias_to=None, keep_unknown=True):
    """Normalize a gene list variable: strip, drop blanks and comments, dedupe
    while preserving order, and optionally map aliases onto approved symbols.

    Pass `approved` / `alias_to` from normalize_genes.load_hgnc() to switch the
    HGNC mapping on. Without them the list is only tidied, never validated.

    Returns (clean_list, report_dict).
    """
    kept, seen = [], set()
    mapped, unknown, dupes = [], [], []
    for raw in genes:
        g = (raw or "").strip()
        if not g or g.startswith("#"):
            continue
        canon = g
        if approved is not None:
            u = g.upper()
            if u in approved:
                canon = u
            elif alias_to and u in alias_to:
                canon = alias_to[u]
                mapped.append({"input": g, "approved": canon})
            else:
                unknown.append(g)
                if not keep_unknown:
                    continue
        if canon in seen:
            dupes.append(g)
            continue
        seen.add(canon)
        kept.append(canon)
    return kept, {"n_input": len(genes), "n_output": len(kept),
                  "mapped": mapped, "unknown": unknown, "duplicates": dupes}


def plan_stages(n_candidates, top_k=None):
    """Pick the method from the candidate count, per the skill's decision table.

    2-10    single-token label MCQ alone; PMI adds nothing at this size.
    11-100  PMI first, then label MCQ over the top 10.
    100+    PMI first, then label MCQ over the top 20. Never put a list this
            long in one prompt: models attend weakly to the middle, so genes
            in the centre are effectively invisible however long the context.
    """
    if n_candidates < 2:
        raise ValueError("need at least 2 candidate genes")
    if n_candidates <= 10:
        return {"stages": ["labels"], "shortlist": n_candidates,
                "reason": "few enough candidates to rank directly by label MCQ"}
    shortlist = top_k if top_k else (10 if n_candidates <= 100 else 20)
    shortlist = min(shortlist, n_candidates)
    return {"stages": ["pmi", "labels"], "shortlist": shortlist,
            "reason": f"{n_candidates} candidates: PMI shortlist to "
                      f"{shortlist}, then label MCQ"}


def build_pmi_pairs(disease, genes, disease_tmpl=DISEASE_TMPL,
                    neutral_tmpl=NEUTRAL_TMPL):
    """Gene list -> the (prefix, target) pairs scored in stage 1.

    The gene is the *target*, never part of the generated text, which is what
    keeps GPR52 and GPR56 on separate independent scores.

    The neutral prefix carries no disease, so its scores are reusable across
    every disease in a run. Compute them once.
    """
    return {
        "conditional_prefix": disease_tmpl.format(disease=disease),
        "neutral_prefix": neutral_tmpl,
        "targets": list(genes),
        "pairs": [(disease_tmpl.format(disease=disease), g) for g in genes],
        "neutral_pairs": [(neutral_tmpl, g) for g in genes],
    }


def build_mcq_prompt(disease, options, fewshot=True):
    """One lettered multiple-choice prompt ending at 'Answer:'."""
    lines = [FEWSHOT] if fewshot else []
    lines.append(f"Disease: {disease}\nOptions:")
    for lab, opt in zip(LABELS, options):
        lines.append(f"{lab}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_mcq_rounds(disease, genes, group_size=4, rotations=4, fewshot=True):
    """Gene list -> every stage 2 prompt, grouped and order-rotated.

    Rotation both removes position bias and exposes it: if the ranking swings
    across rotations the model is following position, not reading the genes.

    Two edge cases are handled rather than left to silently lose genes:
      - group_size above MAX_GENES_PER_GROUP is clamped, so the exit option
        always survives.
      - a trailing group of one borrows the previous group's last gene instead
        of being dropped; the shared gene is simply averaged over more rounds.
    """
    genes = list(genes)
    group_size = max(2, min(group_size, MAX_GENES_PER_GROUP))

    groups = [genes[i:i + group_size] for i in range(0, len(genes), group_size)]
    if len(groups) > 1 and len(groups[-1]) < 2:
        groups[-1] = groups[-2][-1:] + groups[-1]

    rounds = []
    for gi, group in enumerate(groups):
        if len(group) < 2:
            continue
        for r in range(min(rotations, len(group))):
            rolled = group[r:] + group[:r]
            options = rolled + [NONE_OPT]
            rounds.append({
                "group": gi,
                "rotation": r,
                "options": options,
                "prompt": build_mcq_prompt(disease, options, fewshot),
            })
    return rounds
