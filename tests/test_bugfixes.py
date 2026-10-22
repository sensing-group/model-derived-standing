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


# ---- second batch, from the independent review of 22 October 2026 ----------

from standing.audit import check_stability
from standing.checks import null_calibration, quotient_audit
from standing.continuous import resolution
from standing.errors import InvalidInput
from standing.numerics import h2_bernoulli
from standing.record import Record
from standing.sigma import SigmaResult


class _NoClosedForm:
    """A law with a sampler only, so sigma must estimate."""

    def __init__(self, mean, sd):
        self.inner = GaussianLaw(mean, sd)

    def sample(self, size, rng):
        return self.inner.sample(size, rng)


def test_a_spread_that_moves_with_theta_is_not_a_structural_zero():
    C = Conditions({})

    class SpreadOnly:
        def mean(self, t, c):
            return 0.0

        def sd(self, t, c):
            return 1.0 + t

    r = sigma_local(SpreadOnly(), 0.0, 1.0, C)
    assert not r.insensitive
    assert r.value == pytest.approx(hellinger2_gaussian(0.0, 1.0, 0.0, 2.0), rel=1e-9)
    assert resolution(SpreadOnly(), 0.0, C) < float("inf")


def test_an_exact_zero_abstains_even_at_floor_zero():
    zero = SigmaResult(value=0.0, binding_pair=("x", "y"), per_pair={("x", "y"): 0.0}, answers=("x", "y"), exact=True)
    assert decide(zero, floor=0.0).rule is Rule.ABSTAIN
    assert not decide(zero, floor=0.0).admitted


def test_stability_judges_the_base_the_way_it_judges_perturbations():
    C = Conditions({})

    class Same:
        def law(self, a, c):
            return _NoClosedForm(0.0, 1.0)

    answers = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    res = sigma(answers, Same(), C, draws=4000, rng=1)
    assert 0.0 < res.value < 0.05 and not res.exact
    ruling = decide(res, 0.25)
    check = check_stability(ruling, answers, [Same()], C, result=res, draws=4000, rng=2)
    assert check.status == "passed", check.reason


def test_quotient_audit_refuses_an_exact_result_when_forced_to_sample():
    C = Conditions({})

    class G:
        def law(self, a, c):
            return GaussianLaw({"X": 0.0, "Y": 1.0}[a.state], 1.0)

    answers = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    exact = sigma(answers, G(), C)
    assert exact.exact
    with pytest.raises(InvalidInput):
        quotient_audit(answers, G(), C, floor=0.25, delta=0.5, states={"x": {"x1": "X"}, "y": {"y1": "Y"}}, result=exact, force_samples=True, draws=50)


def test_conditions_key_and_record_do_not_depend_on_the_hash_seed(tmp_path):
    import os
    from pathlib import Path

    import standing

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from standing.model import Conditions\n"
        "print(Conditions({'links': frozenset({'ap1', 'ap2', 'ap3'})}).key())\n"
    )
    src = str(Path(standing.__file__).resolve().parents[1])
    seen = set()
    for seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
        out = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1


def test_local_result_labels_are_distinct_for_a_small_delta():
    C = Conditions({})

    class Lin:
        def mean(self, t, c):
            return t

        def sd(self, t, c):
            return 1.0

    r = sigma_local(Lin(), 1.0, 1e-9, C).as_sigma_result()
    assert r.answers[0] != r.answers[1]
    assert r.binding_pair[0] != r.binding_pair[1]


def test_record_build_accepts_a_conditions_object():
    zero = SigmaResult(value=0.5, binding_pair=("x", "y"), per_pair={("x", "y"): 0.5}, answers=("x", "y"), exact=True)
    ruling = decide(zero, 0.25)
    record = Record.build(claim="c", interval="i", ruling=ruling, result=zero, conditions=Conditions({"k": 1}))
    assert record.conditions == [["k", "1"]]
    with pytest.raises(InvalidInput):
        Record.build(claim="c", interval="i", ruling=ruling, result=zero, conditions=42)


def test_bernoulli_divergence_does_not_cancel_to_zero():
    assert h2_bernoulli(0.5, 0.5 + 1e-9) > 0.0
    assert h2_bernoulli(0.2, 0.7) == pytest.approx(1.0 - (np.sqrt(0.14) + np.sqrt(0.24)), rel=1e-12)


def test_null_calibration_refuses_one_empty_side():
    class Half:
        def law(self, a, c):
            silent = bool(c.get("silent"))
            return type("L", (), {"sample": (lambda self, n, rng: np.empty(0)) if silent else (lambda self, n, rng: rng.normal(0.0, 1.0, n))})()

    answers = AnswerSet((Answer("x", "X"), Answer("y", "Y")))
    with pytest.raises(InvalidInput):
        null_calibration(Half(), answers, Conditions({"silent": False}), against=Conditions({"silent": True}), answer="x", draws=50, rng=0)


def test_recruitment_is_admitted_on_the_same_terms_as_direct_admission():
    from standing.certify import certify, split
    from standing.errors import StandingError

    C2 = Conditions({"spread": 1.0})

    class Shifted:
        def __init__(self, offsets):
            self.offsets = offsets

        def law(self, answer, conditions):
            return GaussianLaw(self.offsets[answer.name], 1.0)

    offsets = {"absent": 0.0, "present": 3.0, "confounded": 0.05}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    model = Shifted(offsets)
    before = sigma(answers, model, C2)
    anchor_set = AnswerSet((Answer("seen", ("seen",)), Answer("unseen", ("unseen",))))
    anchor = Anchor("clock", "confounded", standing=sigma(anchor_set, Shifted({"seen": 0.0, "unseen": 4.0}), C2), floor=0.25)
    recompute = lambda n: sigma(n, model, C2)

    bare = decide(before, floor=0.25, answers=answers, anchors=[anchor], recompute=recompute)
    assert bare.rule is Rule.RECRUIT and not bare.admitted and bare.certificate is None
    with pytest.raises(StandingError):
        Anchor.from_ruling("chained", "present", bare)

    rng = np.random.default_rng(0)
    selection, scoring = {}, {}
    for a in answers.answers:
        selection[a.name], scoring[a.name] = split(model.law(a, C2).sample(2000, rng), rng)
    _, readings = certify(answers, selection, scoring, source="readings")
    good = decide(before, floor=0.25, answers=answers, anchors=[anchor], recompute=recompute, certificates=readings)
    assert good.rule is Rule.RECRUIT and good.admitted and good.certificate >= 0.25
    _, simulated = certify(answers, selection, scoring, source="model")
    unverified = decide(before, floor=0.25, answers=answers, anchors=[anchor], recompute=recompute, certificates=simulated)
    assert unverified.rule is Rule.RECRUIT and not unverified.admitted
