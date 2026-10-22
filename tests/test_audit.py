"""The audit: every check runs when it can, says what it needed when it cannot."""

from __future__ import annotations

import json

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.audit import FAILED, NOT_RUN, PASSED, Audit, Check, Probe, audit, coverage_report, grade_of
from standing.certify import certify, split
from standing.errors import InvalidInput, StandingError
from standing.model import Conditions, GaussianLaw
from standing.record import Record, verify
from standing.rules import Rule, decide
from standing.sigma import sigma


class Shifted:
    """Laws that move with the answer, and optionally with a condition."""

    def __init__(self, offsets, drift_key=None):
        self.offsets = offsets
        self.drift_key = drift_key

    def law(self, answer, conditions):
        bias = float(conditions.get(self.drift_key, 0.0)) if self.drift_key else 0.0
        return GaussianLaw(self.offsets[answer.name] + bias, 1.0)


C = Conditions({"spread": 1.0, "bias": 0.0})


def build(offsets, drift_key=None):
    return AnswerSet(tuple(Answer(n, (n,)) for n in offsets)), Shifted(offsets, drift_key)


def certificates_for(answers, model, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    selection, scoring = {}, {}
    for answer in answers.answers:
        selection[answer.name], scoring[answer.name] = split(model.law(answer, C).sample(n, rng), rng)
    return certify(answers, selection, scoring, source="readings")[1]


def test_with_no_inputs_every_check_says_what_it_needed():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    report = audit(ruling)
    assert all(c.status == NOT_RUN for c in report)
    assert all(c.reason.startswith("not run: needs") for c in report)
    assert report.grade == "admissible, unverified"
    assert report.ran == () and not report.contested


def test_a_structural_refusal_needs_no_audit_to_be_trusted():
    answers, model = build({"a": 0.0, "b": 0.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    report = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 5.0}))
    assert ruling.rule is Rule.ABSTAIN
    assert report.grade == "refused, structural"
    assert report["reference"].status == NOT_RUN and "refusal needs no reference" in report["reference"].reason


def test_the_quotient_check_passes_when_the_policy_agrees_and_fails_when_it_splits():
    states = ("empty", "still", "moving")
    alarm = lambda s: "alarm" if s == "moving" else "quiet"  # noqa: E731
    answers = AnswerSet.from_actions(states, alarm)
    model = Shifted({"quiet": 0.0, "alarm": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert audit(ruling, answers=answers, policy=alarm, states=states)["quotient"].status == PASSED

    finer = lambda s: {"empty": "quiet", "still": "watch", "moving": "alarm"}[s]  # noqa: E731
    report = audit(ruling, answers=answers, policy=finer, states=states)
    assert report["quotient"].status == FAILED
    assert report["quotient"].values["splits"] >= 1
    assert report.contested and report.grade == "admissible, unverified, contested"


def test_the_reference_check_fails_when_the_conditions_alone_explain_the_separation():
    answers, model = build({"a": 0.0, "b": 1.6}, drift_key="bias")
    ruling = decide(sigma(answers, model, C), floor=0.25)
    steady = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 0.0}))
    assert steady["reference"].status == PASSED
    # the same-answer null is the estimator's floor, small and positive, never zero
    assert 0.0 < steady["reference"].values["null"] < 0.05

    moved = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 5.0}))
    assert moved["reference"].status == FAILED
    assert moved["reference"].values["null"] >= moved["reference"].values["sigma"]
    assert "conditions alone explain" in moved["reference"].reason


def test_the_certificate_check_reads_the_ruling_or_the_certificates():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    admitted = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    report = audit(admitted)
    assert report["certificate"].status == PASSED
    assert report.grade == "admitted (1 of 11 checks run, all passed)"

    admissible = decide(sigma(answers, model, C), floor=0.25)
    assert audit(admissible, certificates=certs)["certificate"].status == PASSED

    answers, model = build({"a": 0.0, "b": 1.6})
    weak = decide(sigma(answers, model, C), floor=0.25, certificates=certificates_for(answers, model))
    report = audit(weak)
    assert weak.rule is Rule.ADMISSIBLE
    assert report["certificate"].status == FAILED
    assert report.grade == "admissible, unverified, contested"


def test_a_certificate_that_skips_a_pair_fails_the_check_rather_than_raising():
    answers, model = build({"a": 0.0, "b": 3.0, "c": 6.0})
    certs = certificates_for(answers, model)
    partial = {pair: cert for pair, cert in certs.items() if pair == ("a", "b")}
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, certificates=partial)["certificate"]
    assert check.status == FAILED and "skips a pair" in check.reason


def test_the_seed_check_passes_on_a_closed_form_and_reports_its_spread():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, answers=answers, model=model, conditions=C)["seed"]
    assert check.status == PASSED
    assert check.values["spread"] == pytest.approx(0.0, abs=1e-12)


def test_the_full_audit_on_an_admitted_claim_counts_what_ran():
    states = ("empty", "still", "moving")
    alarm = lambda s: "alarm" if s == "moving" else "quiet"  # noqa: E731
    answers = AnswerSet.from_actions(states, alarm)
    model = Shifted({"quiet": 0.0, "alarm": 3.0}, drift_key="bias")
    certs = certificates_for(answers, model)
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    rng = np.random.default_rng(7)
    lit = Probe("switch the light", "turn the room light on and read the alarm statistic",
                predicted=model.law(answers.answers[1], C), observed=rng.normal(3.0, 1.0, 400))
    report = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 0.2}),
                   policy=alarm, states=states, probes=[lit])
    assert [c.status for c in report] == [PASSED, NOT_RUN, NOT_RUN, PASSED, PASSED, NOT_RUN, NOT_RUN, NOT_RUN, NOT_RUN, PASSED, PASSED]
    assert report.grade == "admitted (5 of 11 checks run, all passed)"


def test_an_audit_round_trips_through_its_dict_and_the_record():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25, certificates=certs)
    report = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 0.0}))
    again = Audit.from_dict(json.loads(json.dumps(report.as_dict())))
    assert again == report

    record = Record.build(claim="c", interval="i", ruling=ruling, result=result, certificates=certs, checks=report)
    restored = Record.from_json(record.to_json())
    assert restored == record
    assert verify(restored) == []
    assert Audit.from_dict(restored.checks).grade == report.grade


def test_the_reference_check_does_not_run_against_the_interval_s_own_conditions():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, answers=answers, model=model, conditions=C, against=C)["reference"]
    assert check.status == NOT_RUN and "interval's own" in check.reason


def test_a_record_may_carry_no_audit_and_says_so():
    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    record = Record.build(claim="c", interval="i", ruling=decide(result, floor=0.25), result=result)
    assert record.checks is None
    assert verify(record) == []


def test_a_check_refuses_an_unknown_status_or_step():
    with pytest.raises(InvalidInput, match="status"):
        Check(name="x", step=1, status="maybe", reason="")
    with pytest.raises(InvalidInput, match="step"):
        Check(name="x", step=9, status=PASSED, reason="")


def test_audit_takes_a_ruling_and_nothing_else():
    with pytest.raises(StandingError, match="takes a Ruling"):
        audit(0.9)


@pytest.mark.parametrize(
    "rule, statuses, expected",
    [
        (Rule.ABSTAIN, [FAILED], "refused, structural"),
        (Rule.ADMISSIBLE, [NOT_RUN, NOT_RUN], "admissible, unverified"),
        (Rule.ADMISSIBLE, [PASSED, FAILED], "admissible, unverified, contested"),
        (Rule.ADMITTED, [PASSED, NOT_RUN, PASSED], "admitted (2 of 3 checks run, all passed)"),
        (Rule.ADMITTED, [PASSED, FAILED, NOT_RUN], "admitted (2 of 3 checks run, 1 failed), contested"),
        (Rule.RECRUIT, [PASSED], "admitted (1 of 1 checks run, all passed)"),
        (Rule.CARRY, [NOT_RUN], "carried, below the floor"),
    ],
)
def test_the_grade_reads_the_rule_and_the_checks(rule, statuses, expected):
    checks = [Check(name=f"c{i}", step=1, status=s, reason="") for i, s in enumerate(statuses)]
    assert grade_of(rule, checks) == expected



# --- the probe --------------------------------------------------------------


def test_a_probe_that_comes_out_as_predicted_passes_and_one_that_does_not_fails():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    rng = np.random.default_rng(11)
    law_b = model.law(answers.answers[1], C)
    as_predicted = Probe("move", "put the scatterer where b says", predicted=law_b, observed=rng.normal(3.0, 1.0, 500))
    check = audit(ruling, probes=[as_predicted])["probe"]
    assert check.status == PASSED
    assert check.values["probes"][0]["h2_observed"] < check.values["probes"][0]["h2_floor"] + 0.05

    off = Probe("move", "the same action", predicted=law_b, observed=rng.normal(0.0, 1.0, 500))
    check = audit(ruling, probes=[as_predicted, off])["probe"]
    assert check.status == FAILED and "1 of 2" in check.reason
    assert [row["passed"] for row in check.values["probes"]] == [True, False]


def test_a_model_without_a_probe_is_recorded_as_an_assumption():
    answers, model = build({"a": 0.0, "b": 3.0})
    check = audit(decide(sigma(answers, model, C), floor=0.25))["probe"]
    assert check.status == NOT_RUN and "Probe" in check.reason


def test_a_probe_refuses_a_law_without_sample_and_a_sample_too_small_to_read():
    answers, model = build({"a": 0.0, "b": 3.0})
    law_b = model.law(answers.answers[1], C)
    with pytest.raises(InvalidInput, match="sample"):
        Probe("x", "y", predicted=3.0, observed=np.zeros(20))
    with pytest.raises(InvalidInput, match="at least 8"):
        Probe("x", "y", predicted=law_b, observed=[1.0, 2.0])
    with pytest.raises(InvalidInput, match="finite"):
        Probe("x", "y", predicted=law_b, observed=[float("nan")] * 20)


def test_the_probe_tolerance_is_the_caller_s_and_is_checked():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    probe = Probe("p", "a", predicted=model.law(answers.answers[1], C), observed=np.random.default_rng(1).normal(3.0, 1.0, 100))
    with pytest.raises(InvalidInput, match="tolerance"):
        audit(ruling, probes=[probe], probe_tolerance=1.5)



# --- the vocabulary, the scope, the question -------------------------------


class Vocab(Shifted):
    """A model that owns its states."""

    def __init__(self, offsets, states):
        super().__init__(offsets)
        self._states = tuple(states)

    def states(self):
        return self._states


def test_the_finer_vocabulary_revokes_an_admission_the_coarse_one_manufactured():
    """Coarse: present = moving person only. Fine: present also holds a still person."""
    coarse = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    coarse_model = Shifted({"empty": 0.0, "present": 3.0})
    ruling = decide(sigma(coarse, coarse_model, C), floor=0.25)
    assert ruling.rule is Rule.ADMISSIBLE

    class Fine:
        def law(self, answer, conditions):
            # the law is per STATE: a still person reads like an empty room
            return GaussianLaw({"empty": 0.0, "still": 0.05, "moving": 3.0}[answer.name], 1.0)

    fine = AnswerSet((Answer("empty", ("empty",)), Answer("still", ("still",)), Answer("moving", ("moving",))))
    check = audit(ruling, fine=(fine, Fine()), conditions=C)["vocabulary"]
    assert check.status == FAILED and "revokes the admission" in check.reason
    assert check.values["sigma_fine"] < 0.25 <= check.values["sigma_coarse"]


def test_a_finer_model_that_raises_sigma_is_not_a_refinement():
    coarse = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    ruling = decide(sigma(coarse, Shifted({"a": 0.0, "b": 1.0}), C), floor=0.05)
    fine = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",)), Answer("c", ("c",))))
    check = audit(ruling, fine=(fine, Shifted({"a": 0.0, "b": 4.0, "c": 8.0})), conditions=C)["vocabulary"]
    assert check.status == FAILED and "not a refinement" in check.reason


def test_a_refinement_that_keeps_the_side_of_the_floor_passes():
    coarse = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    ruling = decide(sigma(coarse, Shifted({"a": 0.0, "b": 3.0}), C), floor=0.25)
    fine = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",)), Answer("c", ("c",))))
    check = audit(ruling, fine=(fine, Shifted({"a": 0.0, "b": 3.0, "c": 6.0})), conditions=C)["vocabulary"]
    assert check.status == PASSED and check.values["sigma_fine"] <= check.values["sigma_coarse"] + 1e-9


def test_the_model_s_own_vocabulary_wins_over_a_typed_list():
    answers = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("still", "moving"))))
    model = Vocab({"empty": 0.0, "present": 3.0}, ("empty", "still", "moving"))
    ruling = decide(sigma(answers, model, C), floor=0.25)
    misses = [{"state": "still", "sigma": 0.9}, {"state": "moving", "sigma": 0.1}]
    check = audit(ruling, model=model, misses=misses)["scope"]
    assert check.status == PASSED and check.values["misses_above_floor"] == 1
    with pytest.raises(InvalidInput, match="not the caller's to edit"):
        audit(ruling, model=model, states=("empty", "moving"), misses=misses)


def test_a_miss_on_a_state_the_model_never_described_fails_the_scope():
    answers = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    model = Vocab({"empty": 0.0, "present": 3.0}, ("empty", "moving"))
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, model=model, misses=[{"state": "behind the sofa"}])["scope"]
    assert check.status == FAILED and "never described" in check.reason
    assert check.values["unknown_states"] == ["behind the sofa"]


def test_misses_above_the_floor_are_reported_by_the_scope_and_not_judged_by_it():
    """A view with standing still misses at the rate its separation allows: step 5's question."""
    answers = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    model = Vocab({"empty": 0.0, "present": 3.0}, ("empty", "moving"))
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, model=model, misses=[{"state": "moving", "sigma": 0.8}, {"state": "moving", "sigma": 0.6}])["scope"]
    assert check.status == PASSED
    assert check.values["misses_above_floor"] == 2 and check.values["share_above_floor"] == 1.0
    assert "2 of 2" in check.reason


def test_a_miss_without_a_state_is_refused_not_counted_as_unknown():
    answers = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    model = Vocab({"empty": 0.0, "present": 3.0}, ("empty", "moving"))
    ruling = decide(sigma(answers, model, C), floor=0.25)
    with pytest.raises(InvalidInput, match="carries no state"):
        audit(ruling, model=model, misses=[{"sigma": 0.8}])
    with pytest.raises(InvalidInput, match="carries no state"):
        audit(ruling, model=model, misses=["moving"])


def test_a_coarser_model_offered_as_finer_is_refused_by_the_ladder():
    fine_three = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",)), Answer("c", ("c",))))
    ruling = decide(sigma(fine_three, Shifted({"a": 0.0, "b": 3.0, "c": 6.0}), C), floor=0.25)
    two = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    check = audit(ruling, fine=(two, Shifted({"a": 0.0, "b": 3.0})), conditions=C)["vocabulary"]
    assert check.status == FAILED and "coarser" in check.reason


def test_checks_on_a_bracket_ruling_compare_against_the_scored_sigma_when_given():
    """A bracket ruling carries the bracket's low end as its sigma, not the scored value."""
    from standing.drift import bracket

    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)                     # 0.675
    band = bracket(result.value, 0.60)                    # low end 0.075, well under the value
    ruling = decide(band, floor=0.05, answers=answers)
    fine = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",)), Answer("c", ("c",))))
    finer = Shifted({"a": 0.0, "b": 3.0, "c": 6.0})       # a true refinement, sigma stays 0.675
    without = audit(ruling, fine=(fine, finer), conditions=C)["vocabulary"]
    assert without.status == FAILED and "not a refinement" in without.reason   # the trap
    with_result = audit(ruling, fine=(fine, finer), conditions=C, result=result)["vocabulary"]
    assert with_result.status == PASSED
    assert with_result.values["sigma_coarse"] == pytest.approx(result.value)


def test_a_sampled_perturbation_of_a_structural_zero_is_not_a_flip():
    """The sampled estimator is biased upward at zero; that bias is not a flipped verdict."""
    answers, blind = build({"a": 0.0, "b": 0.0})
    ruling = decide(sigma(answers, blind, C), floor=0.25)

    class SampledBlind:
        class _Law:
            def sample(self, size, rng):
                return rng.normal(0.0, 1.0, size)

        def law(self, answer, conditions):
            return self._Law()

    check = audit(ruling, answers=answers, conditions=C, perturbations=[SampledBlind()], draws=4000)["stability"]
    assert check.status == PASSED, check.reason
    assert 0.0 < check.values["perturbed"][0] < 0.02


def test_a_probe_is_not_compared_or_hashed_by_its_readings():
    answers, model = build({"a": 0.0, "b": 3.0})
    law = model.law(answers.answers[1], C)
    one = Probe("p", "a", predicted=law, observed=np.zeros(20))
    two = Probe("p", "a", predicted=law, observed=np.zeros(20))
    assert (one == two) is False and (one == one) is True
    assert isinstance(hash(one), int)


def test_the_record_refuses_a_quotient_audit_in_the_checks_slot_and_a_malformed_block():
    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25)
    with pytest.raises(InvalidInput, match="A QuotientAudit goes in"):
        Record.build(claim="c", interval="i", ruling=ruling, result=result, checks=object())
    with pytest.raises(InvalidInput, match="carries 'grade'"):
        Record.build(claim="c", interval="i", ruling=ruling, result=result, checks={"rule": "admissible", "checks": []})


def test_the_verifier_reads_an_audit_that_belongs_to_another_verdict():
    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25)
    report = audit(ruling)
    record = Record.build(claim="c", interval="i", ruling=ruling, result=result, checks=report)
    assert verify(record) == []
    payload = json.loads(record.to_json())
    payload["checks"]["rule"] = "admitted"
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("audits a different verdict" in p for p in problems)
    payload["checks"]["rule"] = "admissible"
    payload["checks"]["checks"][0]["status"] = "maybe"
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("not a status" in p for p in problems)
    payload["checks"] = "yes"
    with pytest.raises(InvalidInput, match="object or null"):
        Record.from_json(json.dumps(payload))


def test_scope_needs_a_vocabulary_and_says_so():
    answers, model = build({"a": 0.0, "b": 3.0})
    check = audit(decide(sigma(answers, model, C), floor=0.25), misses=[{"state": "a"}])["scope"]
    assert check.status == NOT_RUN and "vocabulary" in check.reason
    check = audit(decide(sigma(answers, model, C), floor=0.25), misses=[{"state": "a"}], states=("a", "b"))["scope"]
    assert check.status == PASSED


def test_a_question_no_view_can_answer_is_malformed():
    answers, blind = build({"a": 0.0, "b": 0.0})
    ruling = decide(sigma(answers, blind, C), floor=0.25)
    also_blind = Shifted({"a": 1.0, "b": 1.0})
    check = audit(ruling, answers=answers, views=[blind, also_blind], conditions=C)["question"]
    assert check.status == FAILED and "malformed" not in check.reason and "no law anywhere" in check.reason
    seeing = Shifted({"a": 0.0, "b": 3.0})
    check = audit(ruling, answers=answers, views=[blind, seeing], conditions=C)["question"]
    assert check.status == PASSED and "1 of 2" in check.reason
    check = audit(ruling, answers=answers, views=[blind], conditions=C)["question"]
    assert check.status == NOT_RUN and "at least two views" in check.reason



# --- stability, the statistic, coverage --------------------------------------


def test_a_structural_zero_survives_every_perturbation_and_a_fitted_zero_does_not():
    answers, blind = build({"a": 0.0, "b": 0.0})
    ruling = decide(sigma(answers, blind, C), floor=0.25)
    still_blind = [Shifted({"a": 1.0, "b": 1.0}), Shifted({"a": -2.0, "b": -2.0})]
    assert audit(ruling, answers=answers, conditions=C, perturbations=still_blind)["stability"].status == PASSED
    not_blind = [Shifted({"a": 0.0, "b": 0.5})]
    check = audit(ruling, answers=answers, conditions=C, perturbations=not_blind)["stability"]
    assert check.status == FAILED and "not structural" in check.reason and check.values["zero_flips"] == 1


def test_a_magnitude_that_crosses_the_floor_under_a_plausible_law_is_reported():
    answers, model = build({"a": 0.0, "b": 1.7})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert ruling.sigma >= 0.25
    check = audit(ruling, answers=answers, conditions=C, perturbations=[Shifted({"a": 0.0, "b": 1.2})])["stability"]
    assert check.status == FAILED and "across the floor" in check.reason
    check = audit(ruling, answers=answers, conditions=C, perturbations=[Shifted({"a": 0.0, "b": 2.5})])["stability"]
    assert check.status == PASSED


def test_the_statistic_check_compares_the_two_names_and_says_it_is_the_caller_s_account():
    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C, statistic="window_mean")
    ruling = decide(result, floor=0.25)
    ok = audit(ruling, result=result, decision_statistic="window_mean")["statistic"]
    assert ok.status == PASSED and "caller's own account" in ok.reason
    bad = audit(ruling, result=result, decision_statistic="window_max")["statistic"]
    assert bad.status == FAILED and "bounds nothing" in bad.reason
    assert audit(ruling, result=sigma(answers, model, C), decision_statistic="x")["statistic"].status == NOT_RUN


def test_coverage_counts_every_disposition_over_a_run():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    strong = sigma(answers, model, C)
    weak = sigma(*build({"a": 0.0, "b": 0.4})[:2], C)
    none = sigma(*build({"a": 0.0, "b": 0.0})[:2], C)
    rulings = [
        decide(strong, floor=0.25, certificates=certs),
        decide(strong, floor=0.25),
        decide(weak, floor=0.25),
        decide(none, floor=0.25),
        decide(none, floor=0.25),
    ]
    cov = coverage_report(rulings)
    assert (cov.total, cov.admitted, cov.admissible, cov.withheld, cov.refused) == (5, 1, 1, 1, 2)
    assert cov.share_admitted == pytest.approx(0.2) and cov.share_refused == pytest.approx(0.4)
    assert cov.share_uncertified == pytest.approx(0.2)
    with pytest.raises(InvalidInput):
        coverage_report([])
    with pytest.raises(StandingError):
        coverage_report([0.9])


# --- the boundaries a mutation pass found unpinned ---------------------------


def test_a_null_equal_to_sigma_means_the_conditions_explain_all_of_it(monkeypatch):
    """The reference gate is >=: equality is the whole separation explained, and failed."""
    import importlib

    A = importlib.import_module("standing.audit")  # the package re-exports the function under the same name

    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25)

    class Flat:
        value = float(result.value)
        answer = "a"

        def budget_for(self, measured):
            from standing.drift import bracket
            return bracket(measured, self.value)

    monkeypatch.setattr(A, "null_calibration", lambda *a, **k: Flat())
    check = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 1.0}))["reference"]
    assert check.status == FAILED and "conditions alone explain" in check.reason


def test_a_coarse_sigma_exactly_at_the_floor_is_still_an_admission_the_ladder_can_revoke():
    coarse = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    coarse_model = Shifted({"empty": 0.0, "present": 3.0})
    result = sigma(coarse, coarse_model, C)
    ruling = decide(result, floor=result.value)           # floor exactly at sigma
    assert ruling.rule is Rule.ADMISSIBLE
    fine = AnswerSet((Answer("empty", ("empty",)), Answer("still", ("still",)), Answer("moving", ("moving",))))
    check = audit(ruling, fine=(fine, Shifted({"empty": 0.0, "still": 0.05, "moving": 3.0})), conditions=C)["vocabulary"]
    assert check.status == FAILED and "revokes" in check.reason


def test_the_seed_check_fails_when_the_seed_alone_crosses_the_floor():
    class Sampled:
        class _Law:
            def __init__(self, mean):
                self.mean = mean

            def sample(self, size, rng):
                return rng.normal(self.mean, 1.0, size)

        def law(self, answer, conditions):
            return self._Law({"a": 0.0, "b": 0.9}[answer.name])

    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    model = Sampled()
    seeds = (0, 1, 2, 3, 4, 5, 6, 7)
    values = [sigma(answers, model, C, draws=40, rng=seed).value for seed in seeds]
    floor = (min(values) + max(values)) / 2.0
    ruling = decide(sigma(answers, model, C, draws=40, rng=0), floor=floor)
    check = audit(ruling, answers=answers, model=model, conditions=C, draws=40, seeds=seeds)["seed"]
    assert check.status == FAILED and "seed alone moves" in check.reason


def test_a_certificate_exactly_at_the_floor_passes_the_certificate_check():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    bound = min(c.value for c in certs.values())
    ruling = decide(sigma(answers, model, C), floor=bound, certificates=certs)
    assert ruling.rule is Rule.ADMITTED
    assert audit(ruling)["certificate"].status == PASSED


def test_the_certified_bound_is_the_worst_pair_not_the_best():
    answers, model = build({"a": 0.0, "b": 1.6, "c": 8.0})
    certs = certificates_for(answers, model)
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert max(c.value for c in certs.values()) > 0.25
    assert ruling.rule is Rule.ADMISSIBLE
    assert ruling.certificate == pytest.approx(min(c.value for c in certs.values()))


def test_a_record_whose_certificate_sits_exactly_at_the_floor_verifies_as_admitted():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    result = sigma(answers, model, C)
    bound = min(c.value for c in certs.values())
    ruling = decide(result, floor=bound, certificates=certs)
    record = Record.build(claim="c", interval="i", ruling=ruling, result=result, certificates=certs)
    assert record.rule == "admitted" and verify(record) == []


def test_a_record_that_lost_a_pair_s_certificate_no_longer_verifies_as_admitted():
    answers, model = build({"a": 0.0, "b": 3.0, "c": 6.0})
    certs = certificates_for(answers, model)
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25, certificates=certs)
    record = Record.build(claim="c", interval="i", ruling=ruling, result=result, certificates=certs)
    assert verify(record) == []
    payload = json.loads(record.to_json())
    payload["certificates"] = payload["certificates"][:-1]
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("implies 'admissible'" in p for p in problems)


def test_a_hand_built_admitted_ruling_cannot_enter_a_record_without_its_certificates():
    from standing.rules import Ruling

    ruling = Ruling(rule=Rule.ADMITTED, admitted=True, sigma=0.9, floor=0.25, answers=("a", "b"),
                    binding_pair=("a", "b"), certificate=0.5)
    with pytest.raises(InvalidInput, match="no certificates were given"):
        Record.build(claim="c", interval="i", ruling=ruling)


# --- what the systematic sweep left unconstrained ---------------------------


def test_checks_audits_probes_and_coverage_are_immutable():
    import dataclasses

    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    report = audit(ruling)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.grade = "admitted"
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.checks[0].status = PASSED
    probe = Probe("p", "a", predicted=model.law(answers.answers[1], C), observed=np.zeros(20))
    with pytest.raises(dataclasses.FrozenInstanceError):
        probe.name = "q"
    cov = coverage_report([ruling])
    with pytest.raises(dataclasses.FrozenInstanceError):
        cov.total = 9


def test_the_audit_s_passed_and_not_run_views_partition_the_checks():
    answers, model = build({"a": 0.0, "b": 3.0})
    certs = certificates_for(answers, model)
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    report = audit(ruling, answers=answers, model=model, conditions=C)
    assert {c.name for c in report.passed} == {"seed", "certificate"}
    assert {c.name for c in report.not_run} == {c.name for c in report} - {"seed", "certificate"}
    assert report.failed == ()
    assert len(report.passed) + len(report.not_run) + len(report.failed) == len(report.checks)


def test_a_probe_takes_exactly_eight_readings_and_refuses_one_nan_among_them():
    answers, model = build({"a": 0.0, "b": 3.0})
    law = model.law(answers.answers[1], C)
    assert Probe("p", "a", predicted=law, observed=np.zeros(8)).observed.size == 8
    with pytest.raises(InvalidInput, match="finite"):
        Probe("p", "a", predicted=law, observed=[0.0] * 19 + [float("nan")])


def test_the_probe_tolerance_accepts_zero_and_refuses_one():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    probe = Probe("p", "a", predicted=model.law(answers.answers[1], C), observed=np.random.default_rng(1).normal(3.0, 1.0, 100))
    assert audit(ruling, probes=[probe], probe_tolerance=0.0)["probe"].status in (PASSED, FAILED)
    with pytest.raises(InvalidInput, match="tolerance"):
        audit(ruling, probes=[probe], probe_tolerance=1.0)


def test_a_finer_model_with_the_same_number_of_answers_is_not_called_coarser():
    coarse = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    ruling = decide(sigma(coarse, Shifted({"a": 0.0, "b": 3.0}), C), floor=0.25)
    same = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    check = audit(ruling, fine=(same, Shifted({"a": 0.0, "b": 3.0})), conditions=C)["vocabulary"]
    assert check.status == PASSED and "coarser" not in check.reason


def test_a_miss_with_sigma_exactly_at_the_floor_counts_as_above_it():
    answers = AnswerSet((Answer("empty", ("empty",)), Answer("present", ("moving",))))
    model = Vocab({"empty": 0.0, "present": 3.0}, ("empty", "moving"))
    ruling = decide(sigma(answers, model, C), floor=0.25)
    check = audit(ruling, model=model, misses=[{"state": "moving", "sigma": 0.25}])["scope"]
    assert check.values["misses_above_floor"] == 1


def test_a_check_with_only_some_of_its_inputs_does_not_run():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    states = ("a", "b")
    report = audit(ruling, policy=lambda s_: s_, states=states)           # no answers
    assert report["quotient"].status == NOT_RUN
    report = audit(ruling, answers=answers, model=model)                  # no conditions
    assert report["reference"].status == NOT_RUN and report["seed"].status == NOT_RUN
    report = audit(ruling, answers=answers, model=model, conditions=C)    # no against
    assert report["reference"].status == NOT_RUN and report["seed"].status == PASSED
    report = audit(ruling, answers=answers, conditions=C, perturbations=None)
    assert report["stability"].status == NOT_RUN
    # each single missing input on its own, for the guards with several terms
    report = audit(ruling, answers=answers, policy=lambda s_: s_)                      # no states, no model
    assert report["quotient"].status == NOT_RUN
    against = Conditions({"bias": 1.0})
    assert audit(ruling, answers=answers, conditions=C, against=against)["reference"].status == NOT_RUN     # no model
    assert audit(ruling, model=model, conditions=C, against=against)["reference"].status == NOT_RUN         # no answers
    assert audit(ruling, model=model, answers=answers, against=against)["reference"].status == NOT_RUN      # no conditions
    assert audit(ruling, model=model, conditions=C)["seed"].status == NOT_RUN                              # no answers


def test_the_reference_bracket_is_built_from_sigma_and_the_null():
    answers, model = build({"a": 0.0, "b": 1.6}, drift_key="bias")
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25)
    check = audit(ruling, answers=answers, model=model, conditions=C, against=Conditions({"bias": 0.0}), result=result)["reference"]
    from standing.drift import bracket

    assert check.status == PASSED
    band = bracket(check.values["sigma"], check.values["null"])       # T11, on the Hellinger DISTANCE
    assert check.values["bracket_low"] == pytest.approx(band.low, abs=1e-12)
    assert check.values["bracket_high"] == pytest.approx(band.high, abs=1e-12)
    assert 0.0 < band.low < check.values["sigma"]


def test_a_perturbation_that_drops_sigma_from_exactly_the_floor_is_a_crossing():
    answers, model = build({"a": 0.0, "b": 1.7})
    result = sigma(answers, model, C)
    ruling = decide(result, floor=result.value)                            # sigma == floor
    check = audit(ruling, answers=answers, conditions=C, perturbations=[Shifted({"a": 0.0, "b": 1.6})])["stability"]
    assert check.status == FAILED and "across the floor" in check.reason


def test_coverage_counts_a_recruitment_as_admitted_only_when_certified():
    from standing.rules import Anchor

    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    lender_answers, lender = build({"seen": 0.0, "unseen": 4.0})
    anchor = Anchor("clock", "confounded", standing=sigma(lender_answers, lender, C), floor=0.25)
    recompute = lambda narrowed: sigma(narrowed, model, C)
    refused = decide(sigma(*build({"a": 0.0, "b": 0.0})[:2], C), floor=0.25)

    # Recruited on the model alone: the rule is RECRUIT, the entitlement is
    # that of an admissible claim, and coverage counts it as such.
    bare = decide(sigma(answers, model, C), floor=0.25, answers=answers, anchors=[anchor], recompute=recompute)
    assert bare.rule is Rule.RECRUIT and not bare.admitted
    cov = coverage_report([bare, refused])
    assert cov.recruited == 0 and cov.admissible == 1 and cov.share_admitted == 0.0

    # With a certificate on readings clearing the floor on the pair still open,
    # the recruitment is admitted and counted.
    rng = np.random.default_rng(0)
    selection, scoring = {}, {}
    for a in answers.answers:
        selection[a.name], scoring[a.name] = split(model.law(a, C).sample(2000, rng), rng)
    _, certs = certify(answers, selection, scoring, source="readings")
    certified = decide(sigma(answers, model, C), floor=0.25, answers=answers, anchors=[anchor], recompute=recompute, certificates=certs)
    assert certified.rule is Rule.RECRUIT and certified.admitted
    cov = coverage_report([certified, refused])
    assert cov.recruited == 1 and cov.share_admitted == pytest.approx(0.5)


def test_a_perturbation_that_lands_exactly_on_the_floor_is_not_a_crossing():
    answers, model = build({"a": 0.0, "b": 2.2})
    at_floor = sigma(*build({"a": 0.0, "b": 1.7})[:2], C).value
    ruling = decide(sigma(answers, model, C), floor=at_floor)                 # sigma above the floor
    check = audit(ruling, answers=answers, conditions=C, perturbations=[Shifted({"a": 0.0, "b": 1.7})])["stability"]
    assert check.status == PASSED
    assert check.values["perturbed"][0] == pytest.approx(at_floor)
