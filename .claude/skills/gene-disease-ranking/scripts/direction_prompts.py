#!/usr/bin/env python3
"""Directional prompts: not just *which* gene, but *which way* to move it.

`prompts.py` asks "which gene is associated with this disease". That question has
no direction, so a system can be right about CFTR and still recommend the wrong
intervention. These two prompt forms add the axis.

  Form A (outcome)  Fix the intervention, ask for the consequence.
                    "Inhibit CFTR. Effect on cystic fibrosis: improves/worsens?"
                    Run once per direction and compare the two — the paired form
                    needs no gold label for the wrong direction, which matters
                    because "the reverse of therapeutic" is often "no benefit",
                    not "actively worse".

  Form B (choice)   Fix the disease, ask for the intervention, with both
                    directions of two genes on the ballot.
                    "To treat X: A. Inhibit CFTR  B. Activate CFTR  C. ..."
                    Gene identity and direction are scored jointly, so a system
                    that knows the gene but not the direction is visibly wrong.

Both keep the invariant from `prompts.py`: the model emits a single A-E label and
never a gene symbol, so GPR52 and GPR56 can never be confused by the decoder.

Stdlib only, like `prompts.py`.
"""

LABELS = list("ABCDE")

DIRECTIONS = ("inhibit", "activate")
OUTCOMES = ("Improves", "Worsens", "Unclear")
NONE_OPT = "None of the above"

# Few-shot pairs are deliberately NOT in examples/direction_gold.jsonl, and carry
# one direction each, so the shots teach the format without leaking a test answer
# or an inhibit/activate prior.
#   psoriasis / IL17A / inhibit   secukinumab, ChEMBL1743068, approved
#   anemia    / EPOR  / activate  epoetin alfa, ChEMBL1201565, approved
FEWSHOT_OUTCOME = """Disease: psoriasis
Intervention: inhibit IL17A
Effect on the disease:
A. Improves
B. Worsens
C. Unclear
Answer: A

Disease: anemia
Intervention: activate EPOR
Effect on the disease:
A. Improves
B. Worsens
C. Unclear
Answer: A

"""

FEWSHOT_CHOICE = """Disease: psoriasis
Which intervention would treat this disease?
A. Activate IL17A
B. Inhibit IL17A
C. Inhibit EPOR
D. Activate EPOR
E. None of the above
Answer: B

Disease: anemia
Which intervention would treat this disease?
A. Inhibit EPOR
B. Inhibit IL17A
C. Activate EPOR
D. Activate IL17A
E. None of the above
Answer: C

"""


def build_outcome_prompt(disease, gene, direction, fewshot=True):
    """Form A. One (disease, gene, direction) -> a prompt ending at 'Answer:'.

    Labels map to OUTCOMES positionally: A=Improves, B=Worsens, C=Unclear.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    head = FEWSHOT_OUTCOME if fewshot else ""
    opts = "\n".join(f"{lab}. {o}" for lab, o in zip(LABELS, OUTCOMES))
    return (f"{head}Disease: {disease}\n"
            f"Intervention: {direction} {gene}\n"
            f"Effect on the disease:\n{opts}\nAnswer:")


def build_choice_options(true_gene, distractor_gene):
    """The four real options of form B, in canonical (unrotated) order."""
    return [(d, g) for g in (true_gene, distractor_gene) for d in DIRECTIONS]


def build_choice_prompt(disease, options, fewshot=True):
    """Form B. `options` is a list of (direction, gene); the exit option is added.

    Returns (prompt, label_map) where label_map[label] is the (direction, gene)
    that label stands for, or None for the exit option. Never assume A is the
    answer slot -- read it back through label_map.
    """
    if len(options) > len(LABELS) - 1:
        raise ValueError(f"at most {len(LABELS) - 1} options, plus the exit")
    head = FEWSHOT_CHOICE if fewshot else ""
    lines, label_map = [], {}
    for lab, (direction, gene) in zip(LABELS, options):
        lines.append(f"{lab}. {direction.capitalize()} {gene}")
        label_map[lab] = (direction, gene)
    exit_label = LABELS[len(options)]
    lines.append(f"{exit_label}. {NONE_OPT}")
    label_map[exit_label] = None
    prompt = (f"{head}Disease: {disease}\n"
              f"Which intervention would treat this disease?\n"
              + "\n".join(lines) + "\nAnswer:")
    return prompt, label_map


def build_choice_rounds(disease, true_gene, distractor_gene, rotations=4,
                        fewshot=True):
    """Every form-B prompt for one pair, rotated to expose position bias.

    Four options rotate through four starting offsets, so each option occupies
    each of the four slots exactly once. A model following position rather than
    content scores the same for every gene, which shows up as a flat result
    across rotations rather than as a plausible-looking answer.
    """
    base = build_choice_options(true_gene, distractor_gene)
    rounds = []
    for r in range(min(rotations, len(base))):
        rolled = base[r:] + base[:r]
        prompt, label_map = build_choice_prompt(disease, rolled, fewshot)
        rounds.append({"rotation": r, "prompt": prompt, "label_map": label_map})
    return rounds
