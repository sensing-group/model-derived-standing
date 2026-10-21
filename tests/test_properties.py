"""Property-based tests: the search, rather than examples someone thought of.

Every bug found by hand so far was found because I guessed where to look. That
method has no stopping condition and gives no reason to believe the next pass is
clean. These tests state the invariants and let Hypothesis look for a
counterexample, including the metamorphic ones (scale, shift, permutation,
relabelling) which are the class that caught the worst defect: an absolute
degeneracy epsilon that sent sigma to zero at small magnitudes.

If these pass at high example counts, that is evidence of a different kind from
"I could not think of another probe".
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from standing.answers import Answer, AnswerSet
from standing.certify import certify_pair, split
from standing.divergence import hellinger2_gaussian, hellinger2_samples
from standing.drift import bracket
from standing.errors import InvalidInput, StandingError
from standing.model import Conditions, GaussianLaw
from standing.numerics import (
    combine_heterogeneous,
    combine_replicates,
    h2_bernoulli,
    independence_inflation,
    n_eff,
    saturation_count,
)
from standing.record import Record, verify
from standing.rules import Rule, decide
from standing.sigma import sigma

SLOW = settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
counts = st.integers(min_value=1, max_value=200)
finite = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)


# --- the combination laws ---------------------------------------------------


@given(unit, unit)
def test_h2_bernoulli_is_symmetric_bounded_and_zero_on_the_diagonal(a, b):
    value = h2_bernoulli(a, b)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(h2_bernoulli(b, a), abs=1e-12)
    assert h2_bernoulli(a, a) == pytest.approx(0.0, abs=1e-12)


@given(st.lists(unit, min_size=1, max_size=40))
def test_heterogeneous_combination_is_bounded_and_dominates_each_member(hs):
    value = combine_heterogeneous(hs, dependence='independent')
    assert 0.0 <= value <= 1.0
    assert value >= max(hs) - 1e-9


@given(st.lists(unit, min_size=1, max_size=25))
def test_combination_is_invariant_to_the_order_of_its_views(hs):
    rotated = hs[1:] + hs[:1]
    assert combine_heterogeneous(hs, dependence='independent') == pytest.approx(combine_heterogeneous(rotated, dependence='independent'), abs=1e-12)


@given(st.lists(unit, min_size=2, max_size=12))
def test_the_two_wrong_substitutions_err_in_the_directions_claimed(hs):
    truth = combine_heterogeneous(hs, dependence='independent')
    n = len(hs)
    assert combine_replicates(float(np.mean(hs)), n) <= truth + 1e-9
    assert combine_replicates(float(np.max(hs)), n) >= truth - 1e-9


@given(unit, counts)
def test_replicates_is_monotone_in_the_count(h, n):
    assert combine_replicates(h, n) <= combine_replicates(h, n + 1) + 1e-12


@given(unit, counts, st.floats(min_value=0.0, max_value=0.999))
def test_dependence_never_raises_a_combination(h, n, rho):
    assert combine_replicates(h, n, rho) <= combine_replicates(h, n) + 1e-12


@given(counts, st.floats(min_value=0.0, max_value=0.999))
def test_n_eff_is_between_one_and_n_and_under_the_ceiling(n, rho):
    value = n_eff(n, rho)
    assert 1.0 - 1e-9 <= value <= n + 1e-9
    if rho > 0:
        assert value <= 1.0 / rho + 1e-9


@given(unit, counts)
def test_inflation_is_between_one_and_n(h, n):
    value = independence_inflation(h, n)
    assert 1.0 - 1e-9 <= value <= n + 1e-9


@given(st.floats(min_value=1e-6, max_value=0.999), st.floats(min_value=1e-6, max_value=0.5))
def test_saturation_count_inverts_the_law_it_describes(h, eps):
    n = saturation_count(h, eps)
    assert (1.0 - h) ** n == pytest.approx(eps, rel=1e-6)


# --- the estimator, and the metamorphic relations that caught the worst bug --


@given(
    st.floats(min_value=-3.0, max_value=3.0),
    st.floats(min_value=0.2, max_value=3.0),
    st.floats(min_value=-3.0, max_value=3.0),
    st.floats(min_value=0.2, max_value=3.0),
)
def test_closed_form_is_symmetric_bounded_and_zero_only_when_equal(m1, s1, m2, s2):
    value = hellinger2_gaussian(m1, s1, m2, s2)
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(hellinger2_gaussian(m2, s2, m1, s1), abs=1e-12)
    if (m1, s1) == (m2, s2):
        assert value == pytest.approx(0.0, abs=1e-12)


@given(
    st.integers(min_value=0, max_value=2**31),
    st.floats(min_value=0.0, max_value=4.0),
    st.sampled_from([1e-18, 1e-9, 1e-3, 1.0, 1e3, 1e9, 1e18]),
)
@SLOW
def test_the_estimator_is_scale_invariant(seed, delta, scale):
    """T5, as a metamorphic relation. An absolute epsilon broke this at 1e-15."""
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0.0, 1.0, 800), rng.normal(delta, 1.0, 800)
    assert hellinger2_samples(p * scale, q * scale) == pytest.approx(
        hellinger2_samples(p, q), abs=1e-9
    )


@given(
    st.integers(min_value=0, max_value=2**31),
    st.floats(min_value=0.0, max_value=4.0),
    st.floats(min_value=-1e6, max_value=1e6),
)
@SLOW
def test_the_estimator_is_shift_invariant(seed, delta, shift):
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0.0, 1.0, 800), rng.normal(delta, 1.0, 800)
    assert hellinger2_samples(p + shift, q + shift) == pytest.approx(
        hellinger2_samples(p, q), abs=1e-6
    )


@given(st.integers(min_value=0, max_value=2**31))
@SLOW
def test_the_estimator_is_invariant_to_the_order_of_the_draws(seed):
    rng = np.random.default_rng(seed)
    p, q = rng.normal(0.0, 1.0, 600), rng.normal(1.0, 1.0, 600)
    assert hellinger2_samples(rng.permutation(p), rng.permutation(q)) == pytest.approx(
        hellinger2_samples(p, q), abs=1e-12
    )


@given(st.integers(min_value=0, max_value=2**31), st.floats(min_value=0.0, max_value=6.0))
@SLOW
def test_the_estimator_never_leaves_the_unit_interval(seed, delta):
    rng = np.random.default_rng(seed)
    value = hellinger2_samples(rng.normal(0.0, 1.0, 500), rng.normal(delta, 1.0, 500))
    assert 0.0 <= value <= 1.0


# --- the drift bracket ------------------------------------------------------


@given(unit, unit)
def test_the_bracket_always_contains_a_consistent_range(measured, null):
    band = bracket(measured, null)
    assert 0.0 <= band.low <= band.high <= 1.0


@given(unit, unit)
def test_a_bracket_with_a_zero_null_is_a_point(measured, null):
    assume(null == 0.0)
    band = bracket(measured, null)
    assert band.low == pytest.approx(measured, abs=1e-12)
    assert band.high == pytest.approx(measured, abs=1e-12)


@given(
    st.floats(min_value=0.0, max_value=3.0),
    st.floats(min_value=0.0, max_value=3.0),
)
def test_the_bracket_contains_the_truth_under_composed_drift(separation, drift):
    """H is a metric, so the triangle inequality must hold for any two gaps."""
    truth = hellinger2_gaussian(0.0, 1.0, separation, 1.0)
    null = hellinger2_gaussian(0.0, 1.0, drift, 1.0)
    measured = hellinger2_gaussian(0.0, 1.0, separation + drift, 1.0)
    band = bracket(measured, null)
    assert band.low - 1e-9 <= truth <= band.high + 1e-9


# --- sigma and the rules ----------------------------------------------------


@given(
    st.lists(finite, min_size=2, max_size=5, unique=True),
    st.floats(min_value=0.2, max_value=3.0),
)
@SLOW
def test_sigma_is_the_minimum_over_pairs_and_antitone_in_the_set(offsets, spread):
    names = [f"a{i}" for i in range(len(offsets))]
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in names))

    class Model:
        def law(self, answer, conditions):
            return GaussianLaw(offsets[names.index(answer.name)], spread)

    conditions = Conditions({"spread": spread})
    result = sigma(answers, Model(), conditions)
    assert result.value == min(result.per_pair.values())
    assert 0.0 <= result.value <= 1.0
    if len(names) > 2:
        for dropped in names:
            narrowed = sigma(answers.narrowed(ruled_out=dropped), Model(), conditions)
            assert narrowed.value >= result.value - 1e-12


@given(st.lists(finite, min_size=2, max_size=4, unique=True), st.floats(min_value=0.2, max_value=2.0))
@SLOW
def test_sigma_is_invariant_to_renaming_the_answers(offsets, spread):
    """A relabelling is not a change of channel, so it must not move sigma."""
    names = [f"a{i}" for i in range(len(offsets))]
    aliases = [f"z{i}" for i in range(len(offsets))]

    def build(labels):
        answers = AnswerSet(tuple(Answer(n, (n,)) for n in labels))

        class Model:
            def law(self, answer, conditions):
                return GaussianLaw(offsets[labels.index(answer.name)], spread)

        return sigma(answers, Model(), Conditions({"spread": spread})).value

    assert build(names) == pytest.approx(build(aliases), abs=1e-12)


@given(unit, st.floats(min_value=0.0, max_value=1.0))
def test_every_value_and_floor_lands_in_exactly_one_disposition(value, floor):
    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))

    class Model:
        def law(self, answer, conditions):
            return GaussianLaw(0.0 if answer.name == "a" else _offset_for(value), 1.0)

    def _offset_for(target):
        # invert the Gaussian Hellinger to place the pair at `value`
        if target <= 0.0:
            return 0.0
        if target >= 1.0:
            return 40.0
        return math.sqrt(-8.0 * math.log(1.0 - target))

    ruling = decide(sigma(answers, Model(), Conditions({"w": 1})), floor=floor)
    assert ruling.rule in set(Rule)
    assert ruling.admitted == (ruling.rule in (Rule.ADMITTED, Rule.RECRUIT))
    assert ruling.rule is not Rule.ADMITTED  # no certificates were offered
    if ruling.rule is Rule.ABSTAIN:
        assert ruling.sigma == 0.0


# --- certificates -----------------------------------------------------------


@given(st.integers(min_value=0, max_value=2**31), st.floats(min_value=0.0, max_value=5.0))
@SLOW
def test_a_certificate_never_exceeds_the_truth_it_bounds(seed, delta):
    rng = np.random.default_rng(seed)
    truth = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
    ls, lz = split(rng.normal(0.0, 1.0, 2000), rng)
    rs, rz = split(rng.normal(delta, 1.0, 2000), rng)
    value = certify_pair("a", ls, lz, "b", rs, rz, source="readings").value
    assert 0.0 <= value <= truth + 0.06


@given(st.integers(min_value=0, max_value=2**31))
@SLOW
def test_a_certificate_is_symmetric_in_the_pair(seed):
    rng = np.random.default_rng(seed)
    ls, lz = split(rng.normal(0.0, 1.0, 1200), rng)
    rs, rz = split(rng.normal(2.0, 1.0, 1200), rng)
    forward = certify_pair("a", ls, lz, "b", rs, rz, source="readings").value
    backward = certify_pair("b", rs, rz, "a", ls, lz, source="readings").value
    assert forward == pytest.approx(backward, abs=1e-12)


# --- the verifier must never raise on anything it is handed -----------------


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True), st.text()),
        max_size=6,
    )
)
def test_the_verifier_reports_on_arbitrary_junk_and_never_raises(junk):
    """A verifier is handed records it did not write. Crashing is not an option."""
    payload = {
        "schema": "standing/record/1",
        "claim": "c",
        "interval": "i",
        "answers": ["a", "b"],
        "floor": 0.25,
        "rule": "admit",
        "sigma": junk,
        "bracket": None,
        "certificates": [],
        "conditions": [],
        "provenance": {},
    }
    import json

    try:
        record = Record.from_json(json.dumps(payload))
    except (InvalidInput, ValueError):
        return  # refusing to parse is a valid outcome
    problems = verify(record)
    assert isinstance(problems, list)


# --- the audit ---------------------------------------------------------------


class _Offsets:
    def __init__(self, offsets):
        self.offsets = offsets

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], 1.0)


def _ruling(delta: float, floor: float):
    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    model = _Offsets({"a": 0.0, "b": delta})
    return answers, model, decide(sigma(answers, model, Conditions({"w": 1})), floor=floor)


@given(st.floats(min_value=0.0, max_value=6.0), st.floats(min_value=0.01, max_value=0.99))
def test_an_audit_with_no_inputs_runs_nothing_and_grades_the_rule_alone(delta, floor):
    """The absence of a check is visible, and it never raises."""
    from standing.audit import NOT_RUN, audit, grade_of

    _, _, ruling = _ruling(delta, floor)
    report = audit(ruling)
    assert all(c.status == NOT_RUN for c in report)
    assert report.grade == grade_of(ruling.rule, [])
    if ruling.rule is Rule.ABSTAIN:
        assert report.grade == "refused, structural"
    else:
        assert "contested" not in report.grade


@given(
    st.sampled_from(list(Rule)),
    st.lists(st.sampled_from(["passed", "failed", "not_run"]), max_size=8),
)
def test_a_check_that_did_not_run_never_changes_whether_a_grade_is_contested(rule, statuses):
    from standing.audit import Check, grade_of

    checks = [Check(name=f"c{i}", step=1, status=s_, reason="") for i, s_ in enumerate(statuses)]
    before = "contested" in grade_of(rule, checks)
    after = "contested" in grade_of(rule, checks + [Check(name="extra", step=5, status="not_run", reason="")])
    assert before == after
    if rule is not Rule.ABSTAIN:
        assert "contested" in grade_of(rule, checks + [Check(name="x", step=2, status="failed", reason="")])


@given(st.floats(min_value=0.0, max_value=6.0), st.floats(min_value=0.6, max_value=6.0))
def test_a_refinement_that_only_adds_a_far_state_never_revokes_an_admission(delta, far):
    """Adding a state far from every answer adds only large pairs, so the minimum stands."""
    from standing.audit import FAILED, audit

    answers, model, ruling = _ruling(delta, 0.25)
    assume(ruling.rule is not Rule.ABSTAIN)
    fine = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",)), Answer("c", ("c",))))
    finer = _Offsets({"a": 0.0, "b": delta, "c": delta + 40.0 + far})
    check = audit(ruling, fine=(fine, finer), conditions=Conditions({"w": 1}))["vocabulary"]
    assert check.status != FAILED
    assert check.values["sigma_fine"] <= check.values["sigma_coarse"] + 1e-9


@given(st.lists(st.sampled_from(["a", "b", "ghost"]), min_size=1, max_size=12), st.integers(min_value=0, max_value=2**16))
def test_the_scope_check_does_not_depend_on_the_order_of_the_misses(states_, seed):
    from standing.audit import audit

    answers, model, ruling = _ruling(3.0, 0.25)
    misses = [{"state": s_} for s_ in states_]
    shuffled = list(misses)
    np.random.default_rng(seed).shuffle(shuffled)
    one = audit(ruling, misses=misses, states=("a", "b"))["scope"]
    two = audit(ruling, misses=shuffled, states=("a", "b"))["scope"]
    assert one.status == two.status
    assert one.values["unknown_states"] == two.values["unknown_states"]


# --- the admission gate ------------------------------------------------------


def _certified(delta: float, seed: int, n: int = 1200):
    from standing.certify import certify

    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    model = _Offsets({"a": 0.0, "b": delta})
    rng = np.random.default_rng(seed)
    selection, scoring = {}, {}
    for answer in answers.answers:
        selection[answer.name], scoring[answer.name] = split(rng.normal(model.offsets[answer.name], 1.0, n), rng)
    return answers, model, certify(answers, selection, scoring, source="readings")[1]


@given(st.floats(min_value=0.0, max_value=5.0), st.integers(min_value=0, max_value=2**20))
@SLOW
def test_a_certificate_never_exceeds_the_sigma_it_certifies(delta, seed):
    """T10, through decide: the certified bound is a lower bound on sigma."""
    answers, model, certs = _certified(delta, seed)
    result = sigma(answers, model, Conditions({"w": 1}))
    ruling = decide(result, floor=0.01, certificates=certs)
    if ruling.certificate is not None:
        assert ruling.certificate <= result.value + 0.06


@given(
    st.floats(min_value=0.3, max_value=5.0),
    st.integers(min_value=0, max_value=2**20),
    st.floats(min_value=0.01, max_value=0.99),
    st.floats(min_value=0.01, max_value=0.99),
)
@SLOW
def test_admission_is_monotone_in_the_floor(delta, seed, f1, f2):
    """Admitted at a floor means admitted at every lower floor, and the same for admissible."""
    low, high = min(f1, f2), max(f1, f2)
    answers, model, certs = _certified(delta, seed)
    result = sigma(answers, model, Conditions({"w": 1}))
    at_low = decide(result, floor=low, certificates=certs)
    at_high = decide(result, floor=high, certificates=certs)
    rank = {Rule.ABSTAIN: 0, Rule.CARRY: 1, Rule.ADMISSIBLE: 2, Rule.ADMITTED: 3}
    assert rank[at_low.rule] >= rank[at_high.rule]
    if at_high.rule is Rule.ADMITTED:
        assert at_low.rule is Rule.ADMITTED


@given(st.floats(min_value=0.0, max_value=5.0), st.integers(min_value=0, max_value=2**20), st.floats(min_value=0.01, max_value=0.99))
@SLOW
def test_the_same_certificates_in_either_pair_order_give_the_same_ruling(delta, seed, floor):
    answers, model, certs = _certified(delta, seed)
    result = sigma(answers, model, Conditions({"w": 1}))
    forward = decide(result, floor=floor, certificates=certs)
    backward = decide(result, floor=floor, certificates={(b, a): c for (a, b), c in certs.items()})
    assert forward.rule is backward.rule and forward.certificate == backward.certificate
