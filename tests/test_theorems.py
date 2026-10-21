"""T1 to T11 as executable propositions, over the answer set, sigma, certificates
and the drift bracket.

Each test names the theorem it is. A failure here is a claim in the paper being
false, not a regression in an implementation detail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.certify import certify, certify_pair, split
from standing.divergence import hellinger2_gaussian
from standing.drift import bracket
from standing.errors import AnswerSetError, CertificateError, InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.sigma import sigma


class Shifted:
    """A forward model: each answer shifts the statistic, conditions blur it."""

    def __init__(self, offsets: dict[str, float]):
        self.offsets = offsets

    def law(self, answer: Answer, conditions: Conditions) -> GaussianLaw:
        spread = float(conditions.get("spread", 1.0))
        return GaussianLaw(self.offsets[answer.name], spread)


def make(offsets: dict[str, float]) -> tuple[AnswerSet, Shifted]:
    answers = AnswerSet(tuple(Answer(name, (name,)) for name in offsets))
    return answers, Shifted(offsets)


C = Conditions({"spread": 1.0})


# --- T1: sigma is zero exactly when the laws coincide ----------------------


def test_t1_identical_laws_give_zero_and_nothing_else_does():
    answers, model = make({"a": 0.0, "b": 0.0})
    assert sigma(answers, model, C).value == 0.0
    answers, model = make({"a": 0.0, "b": 0.35})
    assert sigma(answers, model, C).value > 0.0


def test_t1_a_structural_zero_survives_any_amount_of_data():
    answers, model = make({"a": 0.0, "b": 0.0})
    for draws in (100, 10_000):
        assert sigma(answers, model, C, draws=draws, rng=0).value == 0.0


# --- T2: antitone in the answer set, which is what Recruit exploits --------


def test_t2_narrowing_never_lowers_sigma():
    rng = np.random.default_rng(4)
    for _ in range(300):
        offsets = {name: float(rng.normal(0, 2)) for name in ("a", "b", "c", "d")}
        if len(set(offsets.values())) < 4:
            continue
        answers, model = make(offsets)
        before = sigma(answers, model, C).value
        for dropped in answers.names:
            after = sigma(answers.narrowed(ruled_out=dropped), model, C).value
            assert after >= before - 1e-12


def test_t2_the_data_did_not_change_when_recruitment_raised_sigma():
    answers, model = make({"absent": 0.0, "present": 2.0, "confounded": 0.02})
    before = sigma(answers, model, C)
    after = sigma(answers.narrowed(ruled_out="confounded"), model, C)
    assert before.binding_pair == ("absent", "confounded")
    assert after.value > before.value
    assert after.provenance["conditions"] == before.provenance["conditions"]


# --- T4: sigma exists for an interval in which nothing was recorded --------


def test_t4_sigma_takes_no_readings_at_all():
    import inspect

    parameters = set(inspect.signature(sigma).parameters)
    assert not parameters & {"readings", "data", "observations", "values"}
    answers, model = make({"a": 0.0, "b": 1.0})
    assert sigma(answers, model, Conditions({"views_reporting": 0})).value > 0.0


def test_t4_conditions_refuse_to_hold_a_reading():
    with pytest.raises(InvalidInput, match="looks like a reading"):
        Conditions({"csi_reading": 3.2})


# --- T5: invariance under reparametrisation --------------------------------


def test_t5_sigma_is_unchanged_by_a_change_of_units():
    answers, model = make({"a": 0.0, "b": 1.5})
    base = sigma(answers, model, C).value

    class Rescaled(Shifted):
        def law(self, answer, conditions):
            inner = super().law(answer, conditions)
            return GaussianLaw(inner.mean * 100.0 - 7.0, inner.sd * 100.0)

    assert sigma(answers, Rescaled(model.offsets), C).value == pytest.approx(base)


# --- T6: fusion cannot create standing -------------------------------------


def test_t6_a_deterministic_statistic_never_exceeds_the_original():
    """A threshold is a kernel, so its separation bounds sigma from below."""
    rng = np.random.default_rng(17)
    for _ in range(400):
        delta = float(rng.uniform(0.0, 4.0))
        truth = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
        left = rng.normal(0.0, 1.0, 4000)
        right = rng.normal(delta, 1.0, 4000)
        ls, lz = split(left, rng)
        rs, rz = split(right, rng)
        certificate = certify_pair("a", ls, lz, "b", rs, rz, source="readings")
        assert certificate.value <= truth + 0.05


# --- T7 family lives in test_numerics; here it meets the answer set --------


def test_the_minimum_over_pairs_is_what_sigma_reports():
    answers, model = make({"a": 0.0, "b": 3.0, "c": 3.02})
    result = sigma(answers, model, C)
    assert result.binding_pair == ("b", "c")
    assert result.value == min(result.per_pair.values())


# --- T9: answers are regions, and a set of two is the minimum question -----


def test_t9_the_quotient_construction_groups_states_by_action():
    states = [round(x, 3) for x in np.linspace(0.0, 1.0, 101)]
    answers = AnswerSet.from_actions(states, lambda s: "act" if s >= 0.6 else "wait")
    assert set(answers.names) == {"act", "wait"}
    assert len(answers) == 2
    acting = next(a for a in answers.answers if a.name == "act")
    assert len(acting.state) == 41  # a region, not a point


def test_a_one_answer_set_refuses_rather_than_returning_zero():
    with pytest.raises(AnswerSetError, match="at least two"):
        AnswerSet((Answer("only", ("only",)),))


def test_a_synonym_is_refused_at_construction():
    with pytest.raises(AnswerSetError, match="same state"):
        AnswerSet((Answer("empty", ("x",)), Answer("vacant", ("x",))))


# --- T10: certificates, and the hypothesis that is not optional ------------


def test_t10_certificate_is_a_lower_bound_on_the_true_value():
    rng = np.random.default_rng(23)
    gaps = []
    for _ in range(200):
        delta = float(rng.uniform(0.5, 3.0))
        truth = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
        ls, lz = split(rng.normal(0.0, 1.0, 6000), rng)
        rs, rz = split(rng.normal(delta, 1.0, 6000), rng)
        value = certify_pair("a", ls, lz, "b", rs, rz, source="readings").value
        assert value <= truth + 0.05
        gaps.append(value / truth)
    assert 0.4 < float(np.mean(gaps)) < 1.0  # valid always, tight never


def test_t10_selecting_and_scoring_on_the_same_sample_is_refused():
    """Both shapes of the mistake: the same array, and two copies of it."""
    rng = np.random.default_rng(1)
    left = rng.normal(0.0, 1.0, 500)
    right = rng.normal(0.0, 1.0, 500)
    with pytest.raises(CertificateError, match="overlap in memory"):
        certify_pair("a", left, left, "b", right, right, source="readings")
    with pytest.raises(CertificateError, match="fit, not a certificate"):
        certify_pair("a", left, left.copy(), "b", right, right.copy(), source="readings")


def test_t10_scoring_on_a_slice_of_the_selection_sample_is_refused():
    """The subtle version: the scoring half is part of what chose the threshold."""
    rng = np.random.default_rng(2)
    full = rng.normal(0.0, 1.0, 1000)
    other = rng.normal(3.0, 1.0, 1000)
    with pytest.raises(CertificateError, match="overlap in memory"):
        certify_pair("a", full, full[:500], "b", other, other[:500], source="readings")


def test_t10_the_null_does_not_certify_a_separation_between_an_answer_and_itself():
    """The 0.9896 case: same answer both sides, scored honestly, must be small."""
    rng = np.random.default_rng(2026)
    values = []
    for _ in range(40):
        one = rng.normal(0.0, 1.0, 4000)
        other = rng.normal(0.0, 1.0, 4000)
        ls, lz = split(one, rng)
        rs, rz = split(other, rng)
        values.append(certify_pair("empty", ls, lz, "empty_again", rs, rz, source="readings").value)
    assert float(np.mean(values)) < 0.02
    assert max(values) < 0.10


def test_certify_over_a_set_returns_the_minimum():
    rng = np.random.default_rng(8)
    selection, scoring = {}, {}
    for name, mean in (("a", 0.0), ("b", 4.0), ("c", 4.05)):
        selection[name], scoring[name] = split(rng.normal(mean, 1.0, 4000), rng)
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in ("a", "b", "c")))
    floor, certificates = certify(answers, selection, scoring, source="readings")
    assert floor == min(c.value for c in certificates.values())
    assert certificates[("b", "c")].value == floor


def test_a_certificate_applies_to_fresh_data():
    rng = np.random.default_rng(3)
    ls, lz = split(rng.normal(0.0, 1.0, 2000), rng)
    rs, rz = split(rng.normal(3.0, 1.0, 2000), rng)
    certificate = certify_pair("a", ls, lz, "b", rs, rz, source="readings")
    fresh = rng.normal(3.0, 1.0, 1000)
    assert float(np.mean(certificate.decide(fresh))) > 0.8


# --- T11: the drift bracket -------------------------------------------------


def test_t11_bracket_contains_the_truth_under_simulated_drift():
    """Same answer at two epochs, plus a real difference: the bound must hold."""
    rng = np.random.default_rng(77)
    for _ in range(300):
        drift = float(rng.uniform(0.0, 1.0))
        separation = float(rng.uniform(0.0, 3.0))
        truth = hellinger2_gaussian(0.0, 1.0, separation, 1.0)
        null = hellinger2_gaussian(0.0, 1.0, drift, 1.0)
        measured = hellinger2_gaussian(0.0, 1.0, separation + drift, 1.0)
        band = bracket(measured, null)
        assert band.low <= truth + 1e-9 <= band.high + 1e-9


def test_t11_reproduces_the_deployment_number():
    band = bracket(0.9019, 0.1710)
    assert band.low == pytest.approx(0.2875, abs=0.001)
    assert band.clears(0.25)
    assert band.ratio() == pytest.approx(5.27, abs=0.01)


def test_t11_a_zero_null_leaves_the_measurement_alone():
    band = bracket(0.64, 0.0)
    assert band.low == pytest.approx(0.64)
    assert band.high == pytest.approx(0.64)


def test_t11_a_null_as_large_as_the_measurement_certifies_nothing():
    band = bracket(0.5, 0.5)
    assert band.low == 0.0
    assert not band.clears(0.01)


def test_t11_bracket_rejects_impossible_input():
    for bad in (-0.1, 1.4, float("nan")):
        with pytest.raises(InvalidInput):
            bracket(bad, 0.1)
        with pytest.raises(InvalidInput):
            bracket(0.5, bad)


def test_t11_high_never_exceeds_one():
    assert bracket(0.9, 0.9).high == 1.0
    assert 0.0 <= bracket(0.9, 0.9).low <= 1.0


# --- reproducibility --------------------------------------------------------


def test_sampled_sigma_is_reproducible_from_its_seed():
    answers, model = make({"a": 0.0, "b": 1.0})
    first = sigma(answers, model, C, rng=42, force_samples=True)
    second = sigma(answers, model, C, rng=42, force_samples=True)
    assert first.value == second.value
    assert first.noise == pytest.approx(1.0 / math.sqrt(2000))


def test_sampled_sigma_tracks_the_exact_value():
    answers, model = make({"a": 0.0, "b": 1.2})
    exact = sigma(answers, model, C)
    sampled = sigma(answers, model, C, draws=8000, rng=5, force_samples=True)
    assert exact.exact and not sampled.exact
    assert sampled.value == pytest.approx(exact.value, abs=0.03)


# --- structural zeros must be declared, not estimated ----------------------


class WithStructuralZero(Shifted):
    """A model that knows a pair cannot be separated under these conditions."""

    def indistinguishable(self, left, right, conditions) -> bool:
        return {left.name, right.name} == {"absent", "confounded"}


def test_a_declared_structural_zero_is_exactly_zero_not_nearly_zero():
    """The CARRY/ABSTAIN boundary sits at exact zero, so an estimate is wrong.

    Found by differential test: the deployment short-circuits a pair its
    reporting aspects cannot separate and returns 0.0000, where an estimator
    returns about 0.005 and turns ABSTAIN into CARRY.
    """
    offsets = {"absent": 0.0, "present": 3.0, "confounded": 0.4}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    estimated = sigma(answers, Shifted(offsets), C).value
    declared = sigma(answers, WithStructuralZero(offsets), C)
    assert estimated > 0.0
    assert declared.value == 0.0
    assert declared.binding_pair == ("absent", "confounded")


def test_the_declaration_only_touches_the_pairs_it_names():
    offsets = {"absent": 0.0, "present": 3.0, "confounded": 0.4}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    result = sigma(answers, WithStructuralZero(offsets), C)
    assert result.per_pair[("absent", "present")] > 0.0
    assert result.per_pair[("present", "confounded")] > 0.0


# --- regressions from the bug hunt, one test per defect found --------------


def test_a_silent_interval_reads_the_same_on_both_paths():
    """force_samples changes how a number is obtained, never what it is."""

    class Silent:
        def law(self, answer, conditions):
            class NoStatistic:
                def sample(self, size, rng):
                    return np.empty(0)

            return NoStatistic()

    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    assert sigma(answers, Silent(), C).value == 0.0
    assert sigma(answers, Silent(), C, force_samples=True).value == 0.0


def test_one_empty_side_is_a_model_defect_and_raises_on_both_paths():
    class HalfSilent:
        def law(self, answer, conditions):
            empty = answer.name == "a"

            class L:
                def sample(self, size, rng):
                    return np.empty(0) if empty else rng.normal(0, 1, size)

            return L()

    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    for force in (False, True):
        with pytest.raises(InvalidInput, match="no draws"):
            sigma(answers, HalfSilent(), C, force_samples=force)


def test_an_unhashable_answer_state_is_refused_in_this_vocabulary():
    with pytest.raises(AnswerSetError, match="unhashable state"):
        AnswerSet((Answer("a", {"x": 1}), Answer("b", {"x": 2})))


def test_conditions_refuse_a_value_with_no_stable_representation():
    """Its repr embeds a memory address, so two identical runs disagree."""

    class Opaque:
        pass

    with pytest.raises(InvalidInput, match="no stable representation"):
        Conditions({"model": Opaque()})
    assert Conditions({"windows": 96, "views": ("a", "b"), "filled": None}).key()


def test_condition_keys_are_reproducible_for_equal_conditions():
    first = Conditions({"windows": 96, "views": ("speech", "network")})
    second = Conditions({"views": ("speech", "network"), "windows": 96})
    assert first.key() == second.key()


def test_a_pair_value_does_not_depend_on_how_many_pairs_ran_before_it():
    """Sharing one generator made the recorded value unreproducible in isolation."""
    from standing.sigma import sigma_of_pair

    answers = AnswerSet(tuple(Answer(n, (n,)) for n in ("a", "b", "c")))
    model = Shifted({"a": 0.0, "b": 1.0, "c": 2.0})
    result = sigma(answers, model, C, draws=3000, rng=0, force_samples=True)
    for left, right in answers.pairs():
        alone = sigma_of_pair(
            left, right, model, C, draws=3000, rng=0, force_samples=True
        )
        assert result.per_pair[(left.name, right.name)] == pytest.approx(alone, abs=1e-12)


def test_a_result_cannot_be_edited_after_the_fact():
    answers, model = make({"a": 0.0, "b": 1.0})
    result = sigma(answers, model, C)
    with pytest.raises(TypeError):
        result.per_pair[("a", "b")] = 99.0
    with pytest.raises(TypeError):
        result.provenance["divergence"] = "something else"


def test_a_certificate_refuses_to_judge_unusable_data():
    rng = np.random.default_rng(3)
    ls, lz = split(rng.normal(0.0, 1.0, 2000), rng)
    rs, rz = split(rng.normal(3.0, 1.0, 2000), rng)
    certificate = certify_pair("a", ls, lz, "b", rs, rz, source="readings")
    assert certificate.decide([1.0, 5.0]).tolist() == [False, True]
    with pytest.raises(InvalidInput, match="non-finite"):
        certificate.decide([1.0, float("nan"), 5.0])


def test_recruitment_refuses_a_recompute_that_is_not_a_scoring():
    from standing.errors import StandingError
    from standing.rules import Anchor, decide

    answers, model = make({"a": 0.0, "b": 1.0, "c": 1.01})
    with pytest.raises(StandingError, match="must return a SigmaResult"):
        decide(
            sigma(answers, model, C),
            floor=0.99,
            answers=answers,
            anchors=[
                Anchor(
                    "x",
                    "c",
                    standing=sigma(*make({"p": 0.0, "q": 4.0}), C),
                    floor=0.25,
                )
            ],
            recompute=lambda narrowed: 0.5,
        )


# ---------------------------------------------------------------------------
# The per-pair scoring path itself: argument order, seed material, and the one
# public entry point that was a third copy of it.
# ---------------------------------------------------------------------------

_PC = Conditions({"c": 1})


class _Asymmetric:
    """`indistinguishable` that answers differently depending on argument order.

    A symmetric hook cannot detect the arguments being swapped, which is how a
    swap survived a mutation sweep.
    """

    def law(self, answer, conditions):
        return GaussianLaw(0.0 if answer.name == "x" else 2.5, 1.0)

    def indistinguishable(self, left, right, conditions):
        return (left.name, right.name) == ("x", "y")


_XY = AnswerSet((Answer("x", "x"), Answer("y", "y")))


def test_the_declared_hook_receives_the_pair_in_the_order_sigma_iterates():
    """Swapping the arguments silently loses a declared structural zero."""
    assert sigma(_XY, _Asymmetric(), _PC, draws=4000, rng=0).value == 0.0


def test_sigma_of_pair_consults_the_declared_hook_like_sigma_does():
    """It was a third copy of the scoring logic and had drifted.

    A pair the model DECLARES unseparable read 0.675 from here while sigma() and
    the quotient audit both reported an exact zero, against a docstring
    promising it reproduced the pair table.
    """
    from standing.sigma import sigma_of_pair

    model = _Asymmetric()
    left, right = _XY.answers
    assert sigma_of_pair(left, right, model, _PC, draws=4000, rng=0) == 0.0
    assert sigma(_XY, model, _PC, draws=4000, rng=0).value == 0.0


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_sigma_of_pair_reproduces_the_entry_in_the_pair_table(seed):
    class Plain:
        def law(self, answer, conditions):
            return GaussianLaw(0.0 if answer.name == "x" else 0.4, 1.0)

    from standing.sigma import sigma_of_pair

    left, right = _XY.answers
    table = sigma(_XY, Plain(), _PC, draws=4000, rng=seed, force_samples=True)
    direct = sigma_of_pair(left, right, Plain(), _PC, draws=4000, rng=seed,
                           force_samples=True)
    if seed is None:
        assert 0.0 <= direct <= 1.0        # unseeded, so only the range is pinned
    else:
        assert direct == pytest.approx(table.per_pair[("x", "y")], abs=1e-12)


def test_the_two_laws_are_not_interchangeable_in_the_estimator_call():
    """Swapping law_a and law_b is invisible unless the two laws differ in shape."""

    class Skewed:
        def law(self, answer, conditions):
            return GaussianLaw(0.0, 1.0) if answer.name == "x" else GaussianLaw(0.0, 4.0)

    value = sigma(_XY, Skewed(), _PC, draws=40_000, rng=0, force_samples=True).value
    exact = hellinger2_gaussian(0.0, 1.0, 0.0, 4.0)
    assert value == pytest.approx(exact, abs=0.02)


def test_the_per_pair_seed_material_cannot_be_reassociated():
    """f"{seed}|{left}|{right}" made ("a|b","c") and ("a","b|c") share a stream.

    Labels are supplied by the caller now, so a separator that can occur inside
    one is a collision waiting to happen.
    """
    from standing.sigma import _pair_rng

    left = _pair_rng(0, "a|b", "c").normal(0.0, 1.0, 8)
    right = _pair_rng(0, "a", "b|c").normal(0.0, 1.0, 8)
    assert not np.allclose(left, right)


def test_the_per_pair_generator_depends_on_the_whole_seed_material():
    from standing.sigma import _pair_rng

    base = _pair_rng(0, "a", "b").normal(0.0, 1.0, 8)
    assert not np.allclose(base, _pair_rng(1, "a", "b").normal(0.0, 1.0, 8))
    assert not np.allclose(base, _pair_rng(0, "b", "a").normal(0.0, 1.0, 8))


def test_a_pair_is_scored_on_the_stream_named_by_its_own_order():
    """Swapping the pair names in the seed derivation changes nothing observable.

    Every comparison in this suite runs BOTH sides through _pair_value, so a swap
    moves both together and cancels. The only way to pin it is to rebuild the
    expected value from _pair_rng directly. This also pins the digest width,
    since a narrower digest gives a different stream for the same names.
    """
    from standing.divergence import hellinger2_samples
    from standing.sigma import _pair_rng

    class Skewed:
        def law(self, answer, conditions):
            return GaussianLaw(0.0, 1.0) if answer.name == "x" else GaussianLaw(0.9, 2.0)

    model = Skewed()
    left, right = _XY.answers
    generator = _pair_rng(0, "x", "y")
    expected = hellinger2_samples(
        model.law(left, _PC).sample(4000, generator),
        model.law(right, _PC).sample(4000, generator),
    )
    table = sigma(_XY, model, _PC, draws=4000, rng=0, force_samples=True)
    assert table.per_pair[("x", "y")] == pytest.approx(expected, abs=1e-12)

    reversed_stream = _pair_rng(0, "y", "x")
    other = hellinger2_samples(
        model.law(left, _PC).sample(4000, reversed_stream),
        model.law(right, _PC).sample(4000, reversed_stream),
    )
    assert other != pytest.approx(expected, abs=1e-12), "the two orders must differ"


def test_a_recorded_sigma_still_reproduces_exactly():
    """One golden value, pinning the whole seeded pipeline against silent drift.

    Everything else in this suite compares two computations that move together,
    so a change anywhere in the seeding chain, the digest width for instance,
    shifts every recorded number without failing anything. That matters here
    more than in most libraries: a record claims to be re-derivable, and a
    deployment holds numbers computed months earlier.

    This test is SUPPOSED to fail when the estimator or the seeding changes. It
    is not a correctness assertion, it is a decision point: if the change was
    deliberate, re-record the constant in the same commit and say so in the
    message, and every stored record from before that commit is now on a
    different scale.
    """

    class Fixed:
        def law(self, answer, conditions):
            return GaussianLaw({"x": 0.0, "y": 0.9}[answer.name], 1.0)

    result = sigma(_XY, Fixed(), _PC, draws=4000, rng=0, force_samples=True)
    assert result.value == pytest.approx(0.11091201804491613, abs=1e-15)
