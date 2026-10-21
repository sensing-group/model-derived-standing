"""The diagnostics, including the two that caught real defects in the field."""

from __future__ import annotations

import math

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.checks import (
    answer_set_ladder,
    null_calibration,
    redundancy_report,
    seed_stability,
)
from standing.errors import InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.sigma import SigmaResult, sigma


class Shifted:
    def __init__(self, offsets, spread=1.0):
        self.offsets, self.spread = offsets, spread

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], self.spread)


C = Conditions({"spread": 1.0})
OFFSETS = {"absent": 0.0, "present": 3.0, "confounded": 0.6}
ANSWERS = AnswerSet(tuple(Answer(n, (n,)) for n in OFFSETS))


def test_the_null_is_small_when_the_law_does_not_move():
    calibration = null_calibration(Shifted(OFFSETS), ANSWERS, C, rng=0)
    assert calibration.answer == "absent"
    assert calibration.value < 0.02


def test_the_null_becomes_the_budget_for_a_measurement():
    calibration = null_calibration(Shifted(OFFSETS), ANSWERS, C, rng=0)
    band = calibration.budget_for(0.9019)
    assert band.low <= 0.9019 <= band.high


def test_the_budget_reproduces_the_deployment_bracket():
    """A null of 0.1710 against a measured 0.9019 certifies 0.2875."""
    from standing.checks import NullCalibration

    band = NullCalibration(value=0.1710, answer="empty", draws=4000).budget_for(0.9019)
    assert band.low == pytest.approx(0.2875, abs=0.001)


def test_a_null_on_an_unknown_answer_is_refused():
    with pytest.raises(InvalidInput, match="not in the answer set"):
        null_calibration(Shifted(OFFSETS), ANSWERS, C, answer="nonexistent")


def test_the_ladder_is_monotone_because_sigma_is_antitone():
    coarse = ANSWERS.narrowed(ruled_out="confounded")
    rungs = answer_set_ladder(
        [ANSWERS, coarse], lambda s: sigma(s, Shifted(OFFSETS), C).value
    )
    assert [len(r.answers) for r in rungs] == [3, 2]
    assert rungs[1].value >= rungs[0].value


def test_an_empty_ladder_is_refused():
    with pytest.raises(InvalidInput, match="at least one answer set"):
        answer_set_ladder([], lambda s: 0.0)


def test_stability_reports_the_spread_and_flags_a_floor_it_straddles():
    stability = seed_stability(
        lambda seed: sigma(
            ANSWERS, Shifted(OFFSETS), C, draws=500, rng=seed, force_samples=True
        ).value,
        seeds=(0, 1, 2, 3, 4),
    )
    assert stability.spread >= 0.0
    assert stability.low <= stability.mean <= stability.high
    assert stability.threatens((stability.low + stability.high) / 2)
    assert not stability.threatens(0.999)


def test_stability_needs_more_than_one_seed():
    with pytest.raises(InvalidInput, match="at least two seeds"):
        seed_stability(lambda seed: 0.5, seeds=(0,))


def test_redundancy_report_prices_dependence_and_saturation():
    report = redundancy_report({f"v{i}": 0.17 for i in range(30)}, rho=0.0)
    assert report.views == 30
    assert report.effective == 30.0
    assert report.ceiling == math.inf
    assert report.saturates_at == pytest.approx(24.7, abs=0.2)
    assert not report.informative  # 30 effective views past a 24.7 threshold
    assert report.combined > 0.99


def test_dependence_pulls_the_effective_count_under_the_ceiling():
    """The wireless deployment's regime: 45 nearly independent per-link views.

    Its report quotes N_eff = 17.95 at rho = 0.034. The formula gives 18.03 at
    exactly 0.034; 17.95 corresponds to rho = 0.03425, which is what 0.034 was
    rounded from. The assertion below is on the formula, not on the rounding.
    """
    report = redundancy_report({f"v{i}": 0.17 for i in range(45)}, rho=0.034)
    assert report.effective == pytest.approx(45.0 / (1.0 + 44.0 * 0.034), abs=1e-9)
    assert report.effective == pytest.approx(18.03, abs=0.01)
    assert redundancy_report({f"v{i}": 0.17 for i in range(45)}, rho=0.03425).effective == (
        pytest.approx(17.95, abs=0.01)
    )
    assert report.ceiling == pytest.approx(29.41, abs=0.05)
    assert report.informative  # under saturation once dependence is priced


def test_redundancy_needs_a_view():
    with pytest.raises(InvalidInput):
        redundancy_report({})


# ---------------------------------------------------------------------------
# quotient_audit: what the answer-level sigma is resting on, one level down.
# ---------------------------------------------------------------------------


class _Mixture:
    """The law of a GROUP, which is what a quotiented answer actually predicts."""

    def __init__(self, means, spread, weights):
        self.means, self.spread, self.weights = means, spread, weights

    def sample(self, n, rng):
        w = np.asarray(self.weights, dtype=float)
        pick = rng.choice(len(self.means), size=n, p=w / w.sum())
        return np.asarray(self.means, dtype=float)[pick] + rng.normal(0.0, self.spread, n)


class Statewise:
    """Predicts for one state, and for a group predicts the mixture over it.

    This is the contract quotient_audit requires: a model that can be asked
    about a single state, not only about the named groups.
    """

    def __init__(self, means, spread=1.0, weights=None):
        self.means, self.spread = means, spread
        self.weights = weights or {}

    def _members(self, answer):
        if isinstance(answer.state, frozenset):
            return sorted(answer.state)
        return [answer.state]

    def law(self, answer, conditions):
        members = self._members(answer)
        for member in members:
            if member not in self.means:
                raise KeyError(f"no law for state {member!r}")
        if len(members) == 1:
            return GaussianLaw(self.means[members[0]], self.spread)
        return _Mixture(
            [self.means[m] for m in members],
            self.spread,
            [self.weights.get(m, 1.0) for m in members],
        )


# A motion-only view: a still occupant and an empty room look identical to it.
MOTION = {"empty": 0.0, "occupied_still": 0.0, "occupied_moving": 4.0}
OCCUPANCY = AnswerSet.from_actions(
    ["empty", "occupied_still", "occupied_moving"],
    lambda s: "no-alarm" if s == "empty" else "alarm",
)


def _states_of(answers):
    """The members of each answer, labelled.

    The audit will not guess these: a tuple is legal both as a region and as one
    opaque state, so guessing silently scored nothing on one reading and tore a
    real state apart on the other.
    """
    out = {}
    for answer in answers.answers:
        if isinstance(answer.state, frozenset):
            out[answer.name] = {str(m): m for m in sorted(answer.state, key=str)}
        else:
            out[answer.name] = {str(answer.state): answer.state}
    return out


def _audit(model=None, answers=OCCUPANCY, **kwargs):
    from standing.checks import quotient_audit

    options = {
        "floor": 0.05,
        "delta": 0.1,
        "draws": 20_000,
        "rng": 0,
        "states": _states_of(answers),
    }
    options.update(kwargs)
    return quotient_audit(answers, model or Statewise(MOTION), C, **options)


def test_the_audit_catches_the_claim_that_sigma_admits():
    """The motivating defect: admitted at the answer level, unentitled below it.

    Sigma is a minimum over ANSWER pairs, so the separable member of a group
    carries the group's law away from the other group and the claim clears the
    floor, while the pair that decides whether the room is reported empty is
    indistinguishable.
    """
    result = sigma(OCCUPANCY, Statewise(MOTION), C, draws=20_000, rng=0)
    assert result.clears(0.05)

    audit = _audit(result=result)
    assert audit.laundered is True
    assert audit.sound is False
    worst = audit.worst
    assert {worst.left, worst.right} == {"empty", "occupied_still"}
    assert worst.left_answer != worst.right_answer


def test_a_collapse_between_two_closed_forms_is_structural():
    """Exactly zero, obtained without estimation, is a theorem rather than a policy."""
    assert _audit().worst.structural is True


def test_a_view_that_separates_every_state_is_sound():
    audit = _audit(Statewise({"empty": 0.0, "occupied_still": 3.0, "occupied_moving": 6.0}))
    assert audit.sound is True
    assert audit.laundered is False
    assert audit.worst is None
    assert "all separate at or above the floor" in audit.reason()


def test_states_grouped_into_the_same_answer_are_not_a_collapse():
    """Two states inside one answer being alike is what the quotient is FOR."""
    answers = AnswerSet.from_actions(
        ["a1", "a2", "b"], lambda s: "hold" if s.startswith("a") else "act"
    )
    # b is kept close enough that sigma stays under 1 - delta, so the pairs are
    # actually scored rather than short-circuited by the mixture bound.
    audit = _audit(Statewise({"a1": 0.0, "a2": 0.0, "b": 1.6}), answers=answers)
    assert audit.sound is True
    assert audit.pairs_scored == 2  # a1|b and a2|b, never a1|a2


def test_an_audit_over_singletons_says_it_adds_nothing():
    answers = AnswerSet(tuple(Answer(n, n) for n in ("empty", "occupied_moving")))
    audit = _audit(answers=answers)
    assert audit.refines is False
    assert "repeats sigma" in audit.reason()


def test_a_collapse_is_reported_even_when_the_quotient_groups_nothing():
    """reason() once hid a collapse behind the repeats-sigma message."""
    answers = AnswerSet(tuple(Answer(n, n) for n in ("empty", "occupied_still")))
    audit = _audit(answers=answers)
    assert audit.refines is False and audit.worst is not None
    assert "indistinguishable" in audit.reason()
    assert "repeats sigma" in audit.reason()


def test_a_collapse_below_the_floor_but_not_zero_is_not_structural():
    audit = _audit(Statewise({"empty": 0.0, "occupied_still": 0.4, "occupied_moving": 4.0}))
    assert audit.sound is False
    assert audit.worst.structural is False
    assert 0.0 < audit.worst.value < 0.05


def test_a_state_pair_scores_what_the_same_pair_scores_as_a_plain_sigma():
    """Otherwise the audit's number is not comparable to the sigma it indicts."""
    audit = _audit()
    collapse = audit.worst
    pair_set = AnswerSet((Answer("empty", "empty"), Answer("occupied_still", "occupied_still")))
    direct = sigma(pair_set, Statewise(MOTION), C, draws=20_000, rng=0)
    assert collapse.value == pytest.approx(direct.value, abs=1e-12)


def test_the_report_is_deterministic():
    first, second = _audit(), _audit()
    assert first.collapses == second.collapses


def test_the_model_must_be_answerable_about_one_state():
    class GroupsOnly(Statewise):
        def law(self, answer, conditions):
            if not isinstance(answer.state, frozenset):
                raise KeyError("I only know the groups")
            return super().law(answer, conditions)

    with pytest.raises(InvalidInput, match="one state at a time"):
        _audit(GroupsOnly(MOTION))


def test_max_pairs_refuses_rather_than_running_away():
    with pytest.raises(InvalidInput, match="max_pairs=1"):
        _audit(max_pairs=1)


def test_max_pairs_admits_exactly_its_own_count():
    assert _audit(max_pairs=2).pairs_scored == 2


def test_a_sigma_from_another_question_cannot_be_compared():
    other = AnswerSet((Answer("empty", "empty"), Answer("occupied_moving", "occupied_moving")))
    result = sigma(other, Statewise(MOTION), C, draws=20_000, rng=0)
    with pytest.raises(InvalidInput, match="says nothing"):
        _audit(result=result)


def test_a_weight_is_not_a_sigma_result():
    with pytest.raises(InvalidInput, match="must be a SigmaResult"):
        _audit(result=0.42)


def test_an_answer_with_no_states_supplied_is_refused():
    """An audit that skips an answer reports sound for a view nobody examined."""
    answers = AnswerSet((Answer("none", "n"), Answer("some", "s")))
    states = {"some": {"s": "s"}}
    with pytest.raises(InvalidInput, match="no states given"):
        _audit(Statewise({"n": 0.0, "s": 3.0}), answers=answers, states=states)


def test_an_empty_member_mapping_is_refused():
    answers = AnswerSet((Answer("none", "n"), Answer("some", "s")))
    with pytest.raises(InvalidInput, match="non-empty mapping"):
        _audit(
            Statewise({"n": 0.0, "s": 3.0}),
            answers=answers,
            states={"none": {}, "some": {"s": "s"}},
        )


def test_states_naming_an_answer_that_is_not_open_are_refused():
    with pytest.raises(InvalidInput, match="not open"):
        _audit(states={**_states_of(OCCUPANCY), "ghost": {"g": "g"}})


def test_two_different_states_under_one_label_cannot_be_named_apart():
    answers = AnswerSet((Answer("hold", "h"), Answer("act", "a")))
    with pytest.raises(InvalidInput, match="share the label"):
        _audit(
            Statewise({"h": 0.0, "a": 6.0}),
            answers=answers,
            states={"hold": {"same": "h"}, "act": {"same": "a"}},
        )


def test_a_label_carrying_a_memory_address_is_refused():
    """It changes between runs, so the audit would not be reproducible."""
    answers = AnswerSet((Answer("hold", "h"), Answer("act", "a")))
    with pytest.raises(InvalidInput, match="memory address"):
        _audit(
            Statewise({"h": 0.0, "a": 6.0}),
            answers=answers,
            states={"hold": {"<x object at 0x7f00>": "h"}, "act": {"a": "a"}},
        )


def test_a_floor_outside_the_unit_interval_is_refused():
    with pytest.raises(InvalidInput):
        _audit(floor=1.5)
    with pytest.raises(InvalidInput):
        _audit(floor=math.nan)


def test_draws_and_max_pairs_are_validated():
    with pytest.raises(InvalidInput, match="draws must be at least 2"):
        _audit(draws=1)
    with pytest.raises(InvalidInput, match="max_pairs must be at least 1"):
        _audit(max_pairs=0)
    with pytest.raises(InvalidInput, match="draws must be an int"):
        _audit(draws=2.5)
    with pytest.raises(InvalidInput, match="max_pairs must be an int"):
        _audit(max_pairs=2.5)
    with pytest.raises(InvalidInput, match="delta must be"):
        _audit(delta=0.0)


def test_a_declared_structural_zero_is_reported_as_structural():
    """`indistinguishable` must key on `.state`, not on `.name`.

    Names differ between the two levels by design: an answer is named for the
    action it demands, a probed state for itself. A hook keyed on names is
    therefore silently inert at one level or the other, which downgrades a
    theorem to "a policy call at the floor you passed".
    """

    class Declaring(Statewise):
        def indistinguishable(self, left, right, conditions):
            def states(answer):
                return set(answer.state) if isinstance(answer.state, frozenset) else {answer.state}

            return states(left) | states(right) == {"empty", "occupied_still"}

    audit = _audit(Declaring(MOTION), force_samples=True)
    assert audit.worst.structural is True
    assert audit.worst.value == 0.0


def test_several_collapses_are_reported_worst_first():
    """The sort and `worst` only mean something against a field of candidates."""
    answers = AnswerSet.from_actions(
        ["far", "near", "same", "other_near", "other_same"],
        lambda s: "A" if s in ("far", "near", "same") else "B",
    )
    means = {"far": 9.0, "near": 0.30, "same": 0.0, "other_near": 0.35, "other_same": 0.0}
    audit = _audit(Statewise(means), answers=answers)

    assert audit.pairs_scored == 6
    values = [c.value for c in audit.collapses]
    assert values == sorted(values)
    assert audit.worst is audit.collapses[0]
    assert audit.worst.value == min(values)
    assert {audit.worst.left, audit.worst.right} == {"same", "other_same"}
    assert all(c.left_answer != c.right_answer for c in audit.collapses)


def test_a_collapse_between_the_later_answers_of_a_larger_set_is_found():
    """Three answers means three answer pairs, and the defect can sit in any of them."""
    answers = AnswerSet.from_actions(
        ["a", "b1", "b2", "c"],
        lambda s: {"a": "hold", "b1": "warn", "b2": "warn", "c": "act"}[s],
    )
    # The collapse is between 'warn' and 'act', never involving the first answer.
    audit = _audit(Statewise({"a": 12.0, "b1": 6.0, "b2": 0.0, "c": 0.0}), answers=answers)
    assert audit.sound is False
    assert {audit.worst.left_answer, audit.worst.right_answer} == {"warn", "act"}
    assert {audit.worst.left, audit.worst.right} == {"b2", "c"}


def test_over_singletons_the_audit_reproduces_sigma_exactly():
    """If the two levels disagree when there is no grouping, they are not comparable."""
    means = {"absent": 0.0, "faint": 0.35, "loud": 4.0}
    answers = AnswerSet(tuple(Answer(n, n) for n in means))
    audit = _audit(Statewise(means), answers=answers, floor=1.0)  # floor 1.0: score every pair
    direct = sigma(answers, Statewise(means), C, draws=20_000, rng=0)

    assert audit.pairs_scored == 3
    assert min(c.value for c in audit.collapses) == pytest.approx(direct.value, abs=1e-12)
    assert audit.worst.value == pytest.approx(direct.value, abs=1e-12)


def test_laws_that_are_close_but_distinct_are_not_called_structural():
    """Where the estimator decides the outcome, `structural` must stay a theorem."""
    audit = _audit(
        Statewise({"empty": 0.0, "occupied_still": 0.05, "occupied_moving": 4.0}),
        floor=0.5,
    )
    near = next(c for c in audit.collapses if {c.left, c.right} == {"empty", "occupied_still"})
    assert near.structural is False
    assert near.value > 0.0


# ---------------------------------------------------------------------------
# null_calibration prices the SAMPLER unless it is given two sets of conditions.
# Reported as a drift budget, the sampler value makes `bracket` tighten around
# whatever number the model produced as compute grows.
# ---------------------------------------------------------------------------


class _Drifting:
    def law(self, answer, conditions):
        return GaussianLaw(float(conditions.get("temp", 0)) * 0.1, 1.0)


C_COLD = Conditions({"temp": 20})
C_WARM = Conditions({"temp": 45})
PAIR = AnswerSet((Answer("x", "x"), Answer("y", "y")))


def test_a_same_conditions_null_prices_the_sampler_and_shrinks_with_draws():
    values = [
        null_calibration(_Drifting(), PAIR, C_COLD, draws=n, rng=0).value
        for n in (500, 8000, 128_000)
    ]
    assert values[0] > values[1] > values[2]
    assert values[-1] < 0.002
    assert null_calibration(_Drifting(), PAIR, C_COLD, draws=500, rng=0).prices == "sampler"


def test_a_two_conditions_null_prices_drift_and_does_not_shrink():
    """If it vanished with draws it would be measuring the sampler again."""
    values = [
        null_calibration(_Drifting(), PAIR, C_COLD, against=C_WARM, draws=n, rng=0).value
        for n in (500, 8000, 128_000)
    ]
    assert min(values) > 0.4
    calibration = null_calibration(_Drifting(), PAIR, C_COLD, against=C_WARM, draws=500, rng=0)
    assert calibration.prices == "drift" and calibration.prices_drift


def test_drift_against_the_same_conditions_is_refused():
    with pytest.raises(InvalidInput, match="no drift between them"):
        null_calibration(_Drifting(), PAIR, C_COLD, against=Conditions({"temp": 20}), draws=100)


def test_drift_against_something_that_is_not_conditions_is_refused():
    with pytest.raises(InvalidInput, match="must be a Conditions"):
        null_calibration(_Drifting(), PAIR, C_COLD, against={"temp": 45}, draws=100)


def test_a_high_sigma_makes_a_mass_bearing_collapse_impossible_without_scoring():
    """H2(mix_A, mix_B) <= 1 - sqrt(a*b)(1 - H2(P,Q)), so sigma > 1 - delta is sound.

    A delta-mass collapsed pair forces the answer-level sigma below 1 - delta, so
    above that line the audit is provably sound and need score nothing.
    """
    audit = _audit(Statewise({"empty": 0.0, "occupied_still": 20.0, "occupied_moving": 40.0}),
                   delta=0.4)
    assert audit.answer_sigma > 1.0 - 0.4
    assert audit.sound is True and audit.pairs_scored == 0


def test_a_state_carrying_less_than_delta_of_its_answer_is_not_a_collapse():
    """The refinement regress: on a discretised continuum every boundary pair
    collapses at any resolution, and only a mass floor stops that."""
    states = {str(i): float(i) for i in range(20)}
    answers = AnswerSet.from_actions(
        list(states.values()), lambda v: "act" if v >= 10 else "hold"
    )
    means = {v: (0.0 if v < 10 else 0.0) if v in (9.0, 10.0) else float(v) for v in states.values()}
    model = Statewise(means)
    explicit = {
        "hold": {f"{v:.0f}": v for v in states.values() if v < 10},
        "act": {f"{v:.0f}": v for v in states.values() if v >= 10},
    }
    # Each state carries 1/10 of its answer, so a delta above that rules the
    # boundary pair out even though it is genuinely indistinguishable.
    loose = _audit(model, answers=answers, states=explicit, delta=0.05, max_pairs=200)
    strict = _audit(model, answers=answers, states=explicit, delta=0.5, max_pairs=200)
    assert any({c.left, c.right} == {"9", "10"} for c in loose.collapses)
    assert strict.sound is True


def test_the_audit_carries_the_delta_it_was_run_at():
    """sound and laundered are relative to BOTH the floor and the mass floor."""
    assert _audit(delta=0.25).delta == 0.25
    assert "delta 0.25" in _audit(delta=0.25).reason()


# ---------------------------------------------------------------------------
# The survivors of a mutation pass: lines whose behaviour nothing pinned.
# ---------------------------------------------------------------------------


def test_a_sampled_state_pair_matches_the_same_pair_scored_by_sigma():
    """The comparability the whole check rests on, on the path where seeds matter.

    The closed-form version of this test cannot see a seeding defect, and there
    was one: sigma seeds a pair on the ANSWER names while the audit probed states
    under their labels, so the two levels scored one identical pair from
    different streams and disagreed on 40 of 40 seeds.
    """
    answers = AnswerSet((Answer("alpha", "s0"), Answer("beta", "s1")))
    model = Statewise({"s0": 0.0, "s1": 0.30})
    states = {"alpha": {"s0": "s0"}, "beta": {"s1": "s1"}}
    for seed in (0, 1, 7, 23):
        scored = sigma(answers, model, C, draws=2000, rng=seed, force_samples=True)
        audit = _audit(
            model, answers=answers, states=states, floor=0.4, delta=0.5,
            draws=2000, rng=seed, force_samples=True, result=scored,
        )
        assert audit.worst.value == pytest.approx(scored.value, abs=1e-12)


def test_changing_the_seed_changes_a_sampled_state_value():
    """Otherwise the audit is not actually using the seed it reports."""
    a = _audit(force_samples=True, rng=0, draws=2000, floor=0.9)
    b = _audit(force_samples=True, rng=9, draws=2000, floor=0.9)
    assert a.worst.value != b.worst.value


def test_changing_draws_changes_a_sampled_state_value():
    a = _audit(force_samples=True, rng=0, draws=500, floor=0.9)
    b = _audit(force_samples=True, rng=0, draws=8000, floor=0.9)
    assert a.worst.value != b.worst.value


def test_each_side_of_a_collapse_belongs_to_the_answer_it_is_reported_under():
    """A set-valued assertion cannot see left and right swapped."""
    audit = _audit()
    states = _states_of(OCCUPANCY)
    for collapse in audit.collapses:
        assert collapse.left in states[collapse.left_answer]
        assert collapse.right in states[collapse.right_answer]
        assert collapse.left_answer != collapse.right_answer


def test_the_collapse_line_names_the_pair_and_says_whether_it_was_admitted():
    audit = _audit()
    line = audit.reason()
    assert "'empty'" in line and "'occupied_still'" in line
    assert "'no-alarm'" in line and "'alarm'" in line
    assert ("admitted anyway" in line) is audit.laundered
    assert "structurally indistinguishable" in line


def test_a_collapse_that_is_not_admitted_is_not_laundered():
    """laundered is a conjunction, and dropping the sigma half survives most suites."""
    audit = _audit(floor=0.9)
    assert audit.sound is False
    assert audit.answer_sigma < 0.9
    assert audit.laundered is False


def test_the_floor_is_a_strict_boundary():
    """A pair scoring exactly the floor is not below it."""
    loose = _audit(Statewise({"empty": 0.0, "occupied_still": 0.4, "occupied_moving": 4.0}),
                   floor=0.5)
    exact_value = loose.worst.value
    tight = _audit(Statewise({"empty": 0.0, "occupied_still": 0.4, "occupied_moving": 4.0}),
                   floor=exact_value)
    assert all(c.value != exact_value for c in tight.collapses)


def test_the_audits_own_draws_guard_is_reachable_when_sigma_is_not_called():
    """With result= supplied, sigma never runs, so its guard cannot stand in."""
    scored = sigma(OCCUPANCY, Statewise(MOTION), C, draws=20_000, rng=0)
    with pytest.raises(InvalidInput, match="draws must be at least 2"):
        _audit(draws=1, result=scored)


def test_the_model_is_probed_with_the_STATE_not_with_its_label():
    """Correctness for non-string states rests entirely on that one argument."""
    answers = AnswerSet((Answer("hold", 1), Answer("act", 2)))

    class Numeric:
        def law(self, answer, conditions):
            if not isinstance(answer.state, int):
                raise TypeError(f"expected the state object, got {answer.state!r}")
            return GaussianLaw(0.0 if answer.state == 1 else 6.0, 1.0)

    audit = _audit(Numeric(), answers=answers, states={"hold": {"1": 1}, "act": {"2": 2}},
                   delta=1.0, floor=0.05)
    assert audit.sound is True


def test_the_sort_breaks_ties_so_equal_collapses_do_not_shuffle():
    # Every state alike, so all six cross-answer pairs tie at exactly zero and
    # only the (left, right) tiebreak decides the order.
    means = {"a1": 0.0, "a2": 0.0, "a3": 0.0, "b1": 0.0, "b2": 0.0, "H": 0.0, "A": 0.0}
    answers = AnswerSet((Answer("hold", "H"), Answer("act", "A")))
    states = {"hold": {"a1": "a1", "a2": "a2", "a3": "a3"}, "act": {"b1": "b1", "b2": "b2"}}
    audit = _audit(Statewise(means), answers=answers, states=states, delta=0.3, floor=0.05)
    assert len(audit.collapses) == 6
    keys = [(c.value, c.left, c.right) for c in audit.collapses]
    assert keys == sorted(keys)


def test_a_state_appearing_in_two_answers_is_refused():
    """Answers partition the states, so a shared one collapses against itself."""
    answers = AnswerSet((Answer("hold", "H"), Answer("act", "A")))
    with pytest.raises(InvalidInput, match="appears in both"):
        _audit(
            Statewise({"shared": 0.0, "x": 3.0, "y": 6.0}),
            answers=answers,
            states={"hold": {"shared": "shared", "x": "x"},
                    "act": {"shared": "shared", "y": "y"}},
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 42.0, -3.0])
def test_a_supplied_sigma_outside_the_unit_interval_is_refused(bad):
    scored = sigma(OCCUPANCY, Statewise(MOTION), C, draws=20_000, rng=0)
    broken = SigmaResult(
        value=bad,
        binding_pair=scored.binding_pair,
        per_pair=scored.per_pair,
        answers=scored.answers,
        exact=scored.exact,
    )
    with pytest.raises(InvalidInput):
        _audit(result=broken)


def test_a_silent_interval_is_structural_on_both_paths():
    """force_samples changes how a number is obtained, never what it means."""

    class Silent:
        class Nothing:
            def sample(self, n, rng):
                return np.empty(0)

        def law(self, answer, conditions):
            return Silent.Nothing()

    for force in (False, True):
        audit = _audit(Silent(), force_samples=force, delta=0.5)
        assert audit.worst.value == 0.0
        assert audit.worst.structural is True, f"force_samples={force}"


# ---------------------------------------------------------------------------
# The mass floor and the mixture bound. These pin the contract change itself,
# which a mutation pass showed could be deleted entirely without a test noticing.
# ---------------------------------------------------------------------------


class _Far:
    """Two answers, each holding a heavy state and a light one, far apart."""

    LOCI = {"a1": 0.0, "a2": 40.0, "b1": 0.5, "b2": -40.0}
    GROUPS = {"A": ["a1", "a2"], "B": ["b1", "b2"]}

    def law(self, answer, conditions):
        state = answer.state
        if state in self.LOCI:
            return GaussianLaw(self.LOCI[state], 1.0)
        locs = [self.LOCI[m] for m in self.GROUPS[state]]
        return _Mixture(locs, 1.0, [1.0] * len(locs))


FAR_ANSWERS = AnswerSet((Answer("A", "A"), Answer("B", "B")))
FAR_STATES = {"A": {"a1": "a1", "a2": "a2"}, "B": {"b1": "b1", "b2": "b2"}}


def _far(**kwargs):
    from standing.checks import quotient_audit

    options = {"floor": 0.30, "delta": 0.5, "states": FAR_STATES,
               "draws": 200_000, "rng": 0}
    options.update(kwargs)
    return quotient_audit(FAR_ANSWERS, _Far(), C, **options)


def test_the_shortcut_may_not_skip_a_pair_that_could_still_collapse():
    """The bound is sigma <= 1 - delta*(1 - v), not sigma <= 1 - delta.

    The second form is the v = 0 case and is strictly too low for any positive
    floor. Measured, a pair scoring 0.0308 with mass 0.5 on both sides was
    skipped at floor 0.30 because sigma came to 0.5148, just over 1 - delta, and
    reason() then asserted every pair separated while having scored none.
    """
    audit = _far()
    assert audit.answer_sigma > 1.0 - 0.5           # over the OLD cutoff
    assert audit.answer_sigma < 1.0 - 0.5 * (1.0 - 0.30)  # under the correct one
    assert audit.pairs_scored == 4
    assert audit.sound is False
    assert {audit.worst.left, audit.worst.right} == {"a1", "b1"}


def test_a_stricter_mass_floor_may_only_remove_pairs_that_fail_it():
    """`sound` was non-monotone in delta, which is the shape of the same defect."""
    found = {d: len(_far(delta=d).collapses) for d in (0.45, 0.50, 0.55)}
    assert found[0.45] == 1 and found[0.50] == 1
    assert found[0.55] == 0          # the pair carries 0.5, so 0.55 excludes it
    assert found[0.45] >= found[0.50] >= found[0.55]


def test_a_pair_light_on_one_side_only_is_not_a_collapse():
    """The mass test is a conjunction over BOTH sides, and deleting it is silent."""
    states = {"A": {"a1": "a1", "a2": "a2"}, "B": {"b1": "b1", "b2": "b2"}}
    weights = {"A": {"a1": 1.0, "a2": 1.0}, "B": {"b1": 0.01, "b2": 0.99}}
    light = _far(delta=0.10, weights=weights, states=states)
    assert all({c.left, c.right} != {"a1", "b1"} for c in light.collapses)

    heavy = _far(delta=0.005, weights=weights, states=states)
    assert any({c.left, c.right} == {"a1", "b1"} for c in heavy.collapses)


def test_the_reported_mass_is_the_smaller_normalised_share():
    weights = {"A": {"a1": 3.0, "a2": 1.0}, "B": {"b1": 1.0, "b2": 1.0}}
    audit = _far(delta=0.05, weights=weights)
    collapse = next(c for c in audit.collapses if {c.left, c.right} == {"a1", "b1"})
    assert collapse.mass == pytest.approx(min(3.0 / 4.0, 1.0 / 2.0))


def test_weights_are_normalised_within_each_answer_and_not_by_count():
    """Un-normalised weights change WHICH pairs clear the floor, so this is not cosmetic.

    The reported mass has to come from the LEFT side here. With the minimum on
    the right, dividing by the member count instead of by the weight total gives
    the same answer and the mistake is invisible: weights of 30 and 10 over two
    members read 0.75 correctly and 15.0 wrongly, and min(0.75, 0.5) equals
    min(15.0, 0.5). Weights of 1 and 3 make the left side the minimum, where
    0.25 and 0.5 come apart.
    """
    # BOTH sides must be non-uniform. For uniform weights, dividing by the total
    # and by the member count coincide, so whichever side is uniform is
    # unaffected and `min` picks it either way. That is why an earlier version of
    # this test, with B uniform, still passed against the by-count mistake.
    weights = {"A": {"a1": 1.0, "a2": 3.0}, "B": {"b1": 1.0, "b2": 3.0}}
    audit = _far(delta=0.05, weights=weights)
    collapse = next(c for c in audit.collapses if {c.left, c.right} == {"a1", "b1"})
    assert collapse.mass == pytest.approx(0.25)   # 1/4 by total, 1/2 by count


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_an_unusable_within_class_weight_is_refused(bad):
    with pytest.raises(InvalidInput, match="finite and non-negative"):
        _far(weights={"A": {"a1": bad, "a2": 1.0}, "B": {"b1": 1.0, "b2": 1.0}})


def test_an_answer_whose_weights_all_vanish_is_refused_not_divided_by_zero():
    with pytest.raises(InvalidInput, match="sum to zero"):
        _far(weights={"A": {"a1": 0.0, "a2": 0.0}, "B": {"b1": 1.0, "b2": 1.0}})


def test_a_single_zero_weight_is_accepted_and_carries_no_mass():
    """A state that never occurs is not an error; it simply cannot make a claim wrong."""
    audit = _far(delta=0.001, weights={"A": {"a1": 0.0, "a2": 1.0}, "B": {"b1": 1.0, "b2": 1.0}})
    assert all({c.left, c.right} != {"a1", "b1"} for c in audit.collapses)


def test_a_delta_outside_the_unit_interval_is_refused():
    for bad in (float("nan"), 2.0, -0.5):
        with pytest.raises(InvalidInput):
            _audit(delta=bad)


def test_the_audit_refuses_arguments_of_the_wrong_shape():
    from standing.checks import quotient_audit

    with pytest.raises(InvalidInput, match="must be an AnswerSet"):
        quotient_audit(None, Statewise(MOTION), C, floor=0.05, delta=0.5,
                       states=_states_of(OCCUPANCY))
    with pytest.raises(InvalidInput, match="states must map"):
        _audit(states=["no-alarm", "alarm"])
    with pytest.raises(InvalidInput, match="empty string"):
        _audit(states={"no-alarm": {"": "empty"},
                       "alarm": {"s": "occupied_still", "m": "occupied_moving"}})


def test_a_supplied_sigma_from_a_different_sample_size_is_refused():
    scored = sigma(OCCUPANCY, Statewise(MOTION), C, draws=500, rng=0)
    with pytest.raises(InvalidInput, match="draws"):
        _audit(result=scored, draws=20_000)


def test_a_supplied_sigma_naming_the_same_answers_in_another_order_is_accepted():
    """The comparison is about the question, not about tuple order."""
    reordered = AnswerSet(tuple(reversed(OCCUPANCY.answers)))
    scored = sigma(reordered, Statewise(MOTION), C, draws=20_000, rng=0)
    assert _audit(result=scored).answer_sigma == pytest.approx(scored.value)


def test_two_equal_but_distinct_state_objects_in_two_answers_are_refused():
    """A state shared by two answers collapses against itself, identity or not."""
    answers = AnswerSet((Answer("hold", "H"), Answer("act", "A")))
    with pytest.raises(InvalidInput, match="appears in both"):
        _audit(
            Statewise({(1, 2): 0.0, "x": 3.0, "y": 6.0}),
            answers=answers,
            states={"hold": {"shared": (1, 2), "x": "x"},
                    "act": {"shared2": tuple([1, 2]), "y": "y"}},
        )


def test_a_resource_failure_in_the_model_is_not_relabelled_as_a_model_defect():
    class OutOfMemory(Statewise):
        def law(self, answer, conditions):
            if not isinstance(answer.state, frozenset) and answer.state == "empty":
                raise MemoryError("out of memory")
            return super().law(answer, conditions)

    with pytest.raises(MemoryError):
        _audit(OutOfMemory(MOTION))


def test_a_structural_pair_is_reported_even_at_floor_zero():
    """value < 0.0 can never fire, so only the structural arm reports it."""
    audit = _audit(floor=0.0, delta=0.5)
    assert audit.worst is not None and audit.worst.structural is True


def test_a_sampled_zero_is_not_structural():
    """structural is `exact AND zero`; dropping the exact half is invisible otherwise."""

    class Point:
        def law(self, answer, conditions):
            return GaussianLaw(0.0, 0.0)

    audit = _audit(Point(), force_samples=True, delta=0.5, floor=0.05)
    assert audit.worst.value == 0.0
    assert audit.worst.structural is False


def test_laundered_fires_when_sigma_sits_exactly_on_the_floor():
    """`>=` against `>`, which no test reached because sigma never landed there."""
    scored = sigma(OCCUPANCY, Statewise(MOTION), C, draws=20_000, rng=0)
    audit = _audit(result=scored, floor=scored.value, delta=0.5)
    assert audit.answer_sigma == scored.value
    assert audit.collapses and audit.laundered is True


def test_the_internal_sigma_honours_force_samples():
    """Visible only when the grouped answers have a closed form to bypass."""
    singles = AnswerSet(tuple(Answer(n, n) for n in ("empty", "occupied_moving")))
    states = _states_of(singles)
    exact = _audit(answers=singles, states=states, delta=1.0, floor=0.9,
                   draws=4000, rng=0, force_samples=False)
    sampled = _audit(answers=singles, states=states, delta=1.0, floor=0.9,
                     draws=4000, rng=0, force_samples=True)
    assert exact.answer_sigma != sampled.answer_sigma
    assert sampled.answer_sigma == pytest.approx(
        sigma(singles, Statewise(MOTION), C, draws=4000, rng=0, force_samples=True).value
    )


def test_the_tie_order_survives_states_built_from_a_set(tmp_path):
    """A caller writing {s: s for s in group} enters in hash order.

    The contract change moved this hazard rather than removing it: label order is
    the caller's now, and only the (value, left, right) tiebreak absorbs a set's
    insertion order. Run in separate processes, because hash randomisation is
    fixed for the life of one.
    """
    import subprocess
    import sys

    probe = tmp_path / "probe.py"
    probe.write_text(
        "from standing import Answer, AnswerSet, Conditions, quotient_audit\n"
        "from standing.model import GaussianLaw\n"
        "C = Conditions({'c': 1})\n"
        "class Flat:\n"
        "    def law(self, a, c): return GaussianLaw(0.0, 1.0)\n"
        "hold = {'a1', 'a2', 'a3'}\n"
        "act = {'b1', 'b2'}\n"
        "answers = AnswerSet((Answer('hold', 'H'), Answer('act', 'A')))\n"
        "states = {'hold': {s: s for s in hold}, 'act': {s: s for s in act}}\n"
        "a = quotient_audit(answers, Flat(), C, floor=0.05, delta=0.3,\n"
        "                   states=states, draws=2000, rng=0)\n"
        "print('|'.join(f'{c.left}~{c.right}' for c in a.collapses))\n"
    )
    env_base = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"}
    seen = set()
    for seed in ("0", "1", "2", "12345", "99999"):
        out = subprocess.run(
            [sys.executable, str(probe)],
            capture_output=True, text=True, cwd=".",
            env={**env_base, "PYTHONHASHSEED": seed},
        )
        assert out.returncode == 0, out.stderr
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"collapse order varied across processes: {seen}"


def test_a_state_label_may_not_collide_with_another_answers_name():
    """Probing singleton states under the answer name merges two namespaces.

    A group holding a state labelled 'Q', alongside a second answer NAMED 'Q',
    gave two DIFFERENT pairs the same seed material. That produces no wrong
    number, but it destroys the invariant the probe naming exists for: the name
    a state is scored under must identify the pair. It also correlates the Monte
    Carlo errors of the colliding pairs and makes the seed useless as a
    provenance key.
    """
    answers = AnswerSet((Answer("P", "p"), Answer("Q", "q"), Answer("R", "r")))
    states = {"P": {"Q": "Q", "u": "u"}, "Q": {"v": "v"}, "R": {"w": "w"}}
    # The model knows every state INCLUDING the group ones, so removing the guard
    # produces a clean "did not raise" rather than an unrelated KeyError. A test
    # that fails for the wrong reason still kills its mutant and tells you
    # nothing about why.
    model = Statewise({"Q": 0.0, "u": 3.0, "v": 0.1, "w": 6.0,
                       "p": 0.0, "q": 0.1, "r": 6.0})
    with pytest.raises(InvalidInput, match="share one random stream"):
        _audit(model, answers=answers, states=states, delta=0.4, floor=0.05)


def test_a_label_matching_its_own_answers_name_is_fine():
    """The collision is with ANOTHER answer's name, not with its own."""
    answers = AnswerSet((Answer("empty", "e"), Answer("busy", "b")))
    states = {"empty": {"empty": "e"}, "busy": {"busy": "b"}}
    audit = _audit(
        Statewise({"e": 0.0, "b": 6.0}), answers=answers, states=states,
        delta=1.0, floor=0.05,
    )
    assert audit.sound is True


def test_a_structural_collapse_is_not_skipped_when_sigma_sits_on_the_bound():
    """The shortcut must be STRICT, and at floor 0 that is load-bearing.

    A non-structural collapse has v < floor strictly, so equality would be safe.
    But a structural pair is reported at any floor, and for those the binding
    cutoff is the v = 0 case where equality is attainable: two answers each
    holding a collapsed state of mass delta and a far-away state give sigma
    exactly 1 - delta. With `>=` the audit skipped a permanent T1 collapse and
    reported that every pair separated.
    """
    from standing.sigma import SigmaResult

    loci = {"sA": 0.0, "sB": 0.0, "farA": -1000.0, "farB": 1000.0}
    groups = {"A": ["sA", "farA"], "B": ["sB", "farB"]}

    class Pinned:
        def law(self, answer, conditions):
            state = answer.state
            if state in loci:
                return GaussianLaw(loci[state], 1.0)
            members = [loci[m] for m in groups[state]]
            return _Mixture(members, 1.0, [1.0] * len(members))

    answers = AnswerSet((Answer("A", "A"), Answer("B", "B")))
    states = {"A": {"sA": "sA", "farA": "farA"}, "B": {"sB": "sB", "farB": "farB"}}
    on_the_bound = SigmaResult(
        value=0.5, binding_pair=("A", "B"), per_pair={("A", "B"): 0.5},
        answers=("A", "B"), exact=True,
    )
    audit = _audit(Pinned(), answers=answers, states=states, floor=0.0, delta=0.5,
                   draws=2000, result=on_the_bound)
    assert audit.answer_sigma == 1.0 - 0.5          # exactly on the cutoff
    assert audit.pairs_scored == 4
    assert audit.worst is not None and audit.worst.structural is True
    assert {audit.worst.left, audit.worst.right} == {"sA", "sB"}


def test_the_collapse_line_names_the_minimum_pair_and_its_numbers():
    """Substring checks cannot see the wrong collapse chosen, or sides swapped."""
    answers = AnswerSet.from_actions(
        ["h1", "h2", "a1", "a2"], lambda s: "hold" if s.startswith("h") else "act"
    )
    means = {"h1": 0.0, "h2": 0.30, "a1": 0.02, "a2": 0.60, "hold": 0.0, "act": 0.0}
    audit = _audit(Statewise(means), answers=answers, states=_states_of(answers),
                   delta=0.4, floor=0.9)
    import re

    worst = audit.worst
    line = audit.reason()

    assert len(audit.collapses) > 1, "several collapses, so the minimum is a choice"
    assert f"{worst.left!r} (in {worst.left_answer!r})" in line
    assert f"{worst.right!r} (in {worst.right_answer!r})" in line
    # Anchored, because a widened format is a SUPERSTRING: "delta 0.40" sits
    # happily inside "delta 0.4000", so a substring test cannot see the change.
    assert re.search(rf"floor {audit.floor:.4f}(?!\d)", line)
    assert re.search(rf"delta {audit.delta:.2f}(?!\d)", line)
    assert re.search(rf"{worst.value:.4f}(?!\d)", line)

    # Pull the pair back out of the sentence rather than asking whether names
    # appear anywhere in it: a state is the left side of several collapses, so a
    # substring test cannot tell which pairing was reported.
    named = re.search(
        r"but '([^']+)' \(in '([^']+)'\) and '([^']+)' \(in '([^']+)'\)", line
    )
    assert named is not None, line
    assert named.groups() == (
        worst.left, worst.left_answer, worst.right, worst.right_answer
    )


def test_two_draws_is_accepted_at_both_levels():
    """sigma() allows it, so the audit must, or one value is legal at one level only.

    That two-levels-disagree asymmetry is the shape of the seeding defect and of
    the silent-interval bug, and it is worth one line to keep the guard from
    drifting.
    """
    assert _audit(draws=2).pairs_scored == 2


def test_supplying_result_does_not_move_the_audits_own_values():
    """`result=` is documented as sparing a recomputation, nothing more.

    When `rng` is a Generator, the result-is-None branch calls sigma(), which
    consumes one draw before the audit takes its base seed. Deriving the seed
    after that call made a documented pure optimisation change every state-pair
    value: measured 0.013081 against 0.013609 for the same pair. The seed is
    derived before the branch now, and this is the only test that says so.
    """
    def values(supply):
        prior = (
            sigma(OCCUPANCY, Statewise(MOTION), C, draws=2000,
                  rng=np.random.default_rng(0), force_samples=True)
            if supply else None
        )
        audit = _audit(
            floor=0.9, delta=0.5, draws=2000,
            rng=np.random.default_rng(0), force_samples=True, result=prior,
        )
        return [c.value for c in audit.collapses]

    assert values(False) == values(True)
