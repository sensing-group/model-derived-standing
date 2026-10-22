"""One test per defect found in the review of 22 October 2026.

Each test is the snippet that confirmed the defect, turned into an assertion
of the behaviour the fix establishes.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.continuous import sigma_local
from standing.divergence import bhattacharyya_gaussian, hellinger2_gaussian, jeffreys_gaussian
from standing.drift import bracket
from standing.model import Conditions, GaussianLaw
from standing.rules import Anchor, Rule, decide
from standing.sigma import sigma


def test_closed_form_divergence_does_not_depend_on_the_units():
    # Two Gaussians, means 0 and 0.5, spread 0.5, in units of 1 down to 1e-12.
    reference = hellinger2_gaussian(0.0, 0.5, 0.5, 0.5)
    for k in (1e-3, 1e-6, 1e-9, 1e-12):
        assert hellinger2_gaussian(0.0, 0.5 * k, 0.5 * k, 0.5 * k) == pytest.approx(reference, rel=1e-9)
        assert bhattacharyya_gaussian(0.0, 0.5 * k, 0.5 * k, 0.5 * k) == pytest.approx(1.0 - reference, rel=1e-9)
        assert jeffreys_gaussian(0.0, 0.5 * k, 0.5 * k, 0.5 * k) == pytest.approx(jeffreys_gaussian(0.0, 0.5, 0.5, 0.5), rel=1e-9)
    # Point masses keep their semantics, decided by an exact zero spread.
    assert hellinger2_gaussian(1e-9, 0.0, 1e-9, 0.0) == 0.0
    assert hellinger2_gaussian(0.0, 0.0, 1e-9, 0.0) == 1.0
    assert hellinger2_gaussian(0.0, 0.0, 0.0, 1e-9) == 1.0


def test_a_bracket_that_touches_zero_is_carried_not_refused():
    answers = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    b = bracket(0.10, 0.20)
    assert b.low == 0.0
    ruling = decide(b, 0.25, answers=answers)
    assert ruling.rule is Rule.CARRY and not ruling.exact and ruling.sigma == 0.0


def test_recruitment_on_a_two_answer_set_does_not_raise():
    C = Conditions({})

    class Close:
        def law(self, a, c):
            return GaussianLaw({"x": 0.0, "y": 0.1}[a.name], 1.0)

    class Far:
        def law(self, a, c):
            return GaussianLaw({"p": 0.0, "q": 5.0}[a.name], 1.0)

    pair = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    weak = sigma(pair, Close(), C)
    strong = sigma(AnswerSet((Answer("p", "P"), Answer("q", "Q"))), Far(), C)
    anchor = Anchor(name="other", rules_out="y", standing=strong, floor=0.25)
    ruling = decide(weak, 0.25, answers=pair, anchors=[anchor], recompute=lambda s: sigma(s, Close(), C))
    assert ruling.rule is Rule.CARRY and ruling.recruited is None


def test_a_generator_seeded_result_reproduces_from_its_own_record():
    C = Conditions({})

    class Sampled:
        def law(self, a, c):
            law = GaussianLaw({"x": 0.0, "y": 0.4}[a.name], 1.0)
            return type("NoClosedForm", (), {"sample": law.sample})()

    answers = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    first = sigma(answers, Sampled(), C, rng=np.random.default_rng(5), draws=500)
    recorded = first.provenance["seed"]
    assert isinstance(recorded, int)
    again = sigma(answers, Sampled(), C, rng=recorded, draws=500)
    assert dict(again.per_pair) == dict(first.per_pair)


def test_continuous_insensitivity_does_not_depend_on_the_units():
    C = Conditions({})

    class Tiny:
        def __init__(self, k):
            self.k = k

        def mean(self, t, c):
            return self.k * (1.0 + 0.5 * t)

        def sd(self, t, c):
            return self.k * 0.1

    reference = sigma_local(Tiny(1.0), 0.5, 0.1, C)
    assert not reference.insensitive and reference.value > 0.0
    for k in (1e-6, 1e-9, 1e-12):
        r = sigma_local(Tiny(k), 0.5, 0.1, C)
        assert not r.insensitive
        assert r.value == pytest.approx(reference.value, rel=1e-9)

    class Flat:
        def mean(self, t, c):
            return 14.1e-9

        def sd(self, t, c):
            return 1e-10

    assert sigma_local(Flat(), 0.5, 0.1, C).insensitive


def test_the_answer_fingerprint_is_stable_across_hash_seeds(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from standing.answers import AnswerSet\n"
        "s = AnswerSet.from_actions([frozenset({'a','b','c'}), frozenset({'d','e'})],\n"
        "                           lambda st: 'act' if 'a' in st else 'hold')\n"
        "print(s.origin.action)\n"
    )
    import os
    from pathlib import Path

    import standing

    # The probe runs in a fresh interpreter, which must find the package
    # whether it is installed or only on the test process's path.
    src = str(Path(standing.__file__).resolve().parents[1])
    seen = set()
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1
