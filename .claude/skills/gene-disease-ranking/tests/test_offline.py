#!/usr/bin/env python3
"""Offline regression tests: fake scorers, no model weights, no GPU.

Covers the parts that fail silently — prompt construction, the PMI frequency
correction, abstention, and the two ways a gene used to disappear from stage 2.
Needs numpy only; torch is stubbed if absent, since nothing here runs a model.

    python tests/test_offline.py
"""
import importlib.util
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

if importlib.util.find_spec("torch") is None:
    stub = types.ModuleType("torch")
    stub.no_grad = lambda: (lambda fn: fn)
    sys.modules["torch"] = stub

import numpy as np  # noqa: E402

import score_labels as SL  # noqa: E402
import score_pmi as SP  # noqa: E402
from prompts import (LABELS, NONE_OPT, build_mcq_rounds,  # noqa: E402
                     build_mcq_prompt, plan_stages)
from rank import GeneRanker, preview  # noqa: E402

EXAMPLES = os.path.join(HERE, "..", "examples")


def read(path):
    with open(path) as f:
        return [l.strip() for l in f
                if l.strip() and not l.strip().startswith("#")]


def options_of(prompt):
    block = prompt.rsplit("Options:", 1)[1]
    out = {}
    for line in block.split("\n"):
        m = re.match(r"\s*([A-E])\.\s+(.*\S)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class FakePMI:
    """cond = frequency + disease affinity; neutral = frequency alone.

    PMI must cancel the frequency term, so a famous gene with no affinity has
    to fall down the ranking.
    """

    def __init__(self, affinity, freq):
        self.affinity, self.freq = affinity, freq

    def score(self, prefix, targets, batch_size=32):
        conditioned = "associated with" in prefix
        return np.array([self.freq.get(t, 0.0) +
                         (self.affinity.get(t, 0.0) if conditioned else 0.0)
                         for t in targets], dtype=np.float64)


class FakeLabels:
    def __init__(self, pref):
        self.pref, self.prompts = pref, []

    def score(self, prompt):
        self.prompts.append(prompt)
        opts = options_of(prompt)
        return {lab: self.pref.get(opts.get(lab, ""), -3.0) for lab in LABELS}


def ranker(affinity, freq, pref):
    r = GeneRanker("fake/model")
    r._pmi_scorer = FakePMI(affinity, freq)
    r._label_scorer = FakeLabels(pref)
    return r


GENES = read(os.path.join(EXAMPLES, "candidates.txt"))
FREQ = dict.fromkeys(GENES, 0.0)
FREQ["TP53"] = 50.0          # famous, and irrelevant to cystic fibrosis
AFF = {"CFTR": 9.0, "GPR56": 3.0, "HBB": 2.0}
PREF = {"CFTR": -0.1, NONE_OPT: -6.0}

failures = []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        failures.append(f"{name}: {e}")
        print(f"  FAIL  {name}: {e}")
    else:
        print(f"  ok    {name}")


def test_plan_table():
    assert plan_stages(5)["stages"] == ["labels"]
    assert plan_stages(50)["stages"] == ["pmi", "labels"]
    assert plan_stages(50)["shortlist"] == 10
    assert plan_stages(500)["shortlist"] == 20
    try:
        plan_stages(1)
    except ValueError:
        pass
    else:
        raise AssertionError("a single candidate should be rejected")


def test_symbol_never_generated():
    """Every prompt must stop at 'Answer:', with symbols only as options."""
    for rnd in build_mcq_rounds("Cystic fibrosis", GENES[:9]):
        assert rnd["prompt"].rstrip().endswith("Answer:")
        assert NONE_OPT in rnd["options"]


def test_pmi_cancels_frequency():
    r = ranker(AFF, FREQ, PREF).rank("Cystic fibrosis", GENES)
    order = [x["gene"] for x in r["stage1"]]
    assert order[0] == "CFTR", order[:3]
    assert order.index("TP53") > 5, f"frequency leaked: TP53 at {order.index('TP53')}"
    assert "TP53" not in r["shortlist"]
    assert len(r["shortlist"]) == 10
    assert r["call"] == "CFTR"
    assert r["ranked"][0]["gene"] == "CFTR"   # evaluate.py reads this key


def test_no_gene_dropped():
    five = ["CFTR", "HBB", "GPR52", "GPR56", "APOE"]
    r = ranker(AFF, FREQ, PREF).rank("Cystic fibrosis", five)
    assert r["stage1"] is None, "5 candidates should skip PMI"
    assert {x["gene"] for x in r["stage2"]} == set(five)

    # A trailing group of one used to be skipped entirely.
    nine = GENES[:9]
    scored = SL.rerank(FakeLabels(PREF), "Cystic fibrosis", nine)
    assert set(scored) - {NONE_OPT} == set(nine), set(nine) - set(scored)


def test_exit_option_survives_large_groups():
    for gs in (4, 5, 9):
        for rnd in build_mcq_rounds("Cystic fibrosis", GENES[:9], group_size=gs):
            assert NONE_OPT in rnd["options"], f"exit lost at group_size={gs}"
            assert len(rnd["options"]) <= len(LABELS)


def test_abstains_rather_than_guesses():
    flat = ranker(AFF, FREQ, {NONE_OPT: -6.0}).rank("Cystic fibrosis", GENES)
    assert flat["call"] is None and "margin" in flat["abstain_reason"]

    exit_wins = ranker(AFF, FREQ, {NONE_OPT: 0.0, "CFTR": -0.1}).rank(
        "Cystic fibrosis", GENES)
    assert exit_wins["call"] is None
    assert "none of the above" in exit_wins["abstain_reason"]


def test_alias_and_dedup():
    r = ranker(AFF, FREQ, PREF)
    r.approved, r.alias_to = {"HTT", "CFTR", "HBB"}, {"IT15": "HTT"}
    out = r.rank("Cystic fibrosis", ["CFTR", "cftr ", "IT15", "NOTAGENE", "HBB"])
    rep = out["normalization"]
    assert rep["mapped"] == [{"input": "IT15", "approved": "HTT"}]
    assert rep["unknown"] == ["NOTAGENE"] and rep["duplicates"] == ["cftr"]


def test_preview_needs_no_model():
    p = preview("Cystic fibrosis", GENES)
    assert p["plan"]["stages"] == ["pmi", "labels"]
    assert p["stage1"]["conditional_prefix"].endswith("Cystic fibrosis: ")
    assert p["stage1"]["neutral_prefix"] == "Gene: "
    assert p["stage2_rounds"]


def test_templates_shared():
    assert SP.DISEASE_TMPL == "Gene most strongly associated with {disease}: "
    assert SP.NEUTRAL_TMPL == "Gene: "
    assert SL.build_prompt is build_mcq_prompt


if __name__ == "__main__":
    print("offline tests (no model, no GPU)")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    print()
    if failures:
        print(f"{len(failures)} FAILED")
        sys.exit(1)
    print("all passed")
