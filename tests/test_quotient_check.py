"""The action quotient checked against the policy it serves, and recorded.

This covers the one condition the library could not check at all. It still
cannot check whether a policy is the right thing to do about the world, and
nothing here pretends to: every test below asks whether the answer set and the
policy agree with each other, which is the half that is decidable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet, QuotientOrigin
from standing.checks import QuotientCheck, quotient_check
from standing.cli import main
from standing.errors import AnswerSetError, InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.record import Record, verify
from standing.rules import Anchor, decide
from standing.sigma import sigma

STATES = ("empty", "still", "moving")


def alarm(state):
    """The policy the motion sensor actually feeds: three states, two actions."""
    return "ring" if state in ("still", "moving") else "quiet"


def three_ways(state):
    """A policy that treats a still occupant as its own case."""
    return {"empty": "quiet", "still": "check", "moving": "ring"}[state]


class Shifted:
    def __init__(self, offsets):
        self.offsets = offsets

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], 1.0)


class _MixtureLaw:
    """An equal-weight Gaussian mixture over an answer's member states."""

    def __init__(self, means, sd):
        self.means, self.sd = np.array(sorted(means), dtype=float), float(sd)

    def sample(self, size, rng):
        picked = self.means[rng.integers(0, len(self.means), size)]
        return picked + rng.normal(0.0, self.sd, size)


class Mixture:
    """A grouped answer's law really IS the mixture of its members' laws.

    That is what `from_actions` promises, and it is the property the merge case
    turns on: a mixture can sit closer to a rival than any of its components.
    The rival here is the wide law at the origin; every other answer is unit
    width around its member means.
    """

    def law(self, answer, conditions):
        members = (answer.state if isinstance(answer.state, frozenset)
                   else frozenset({answer.state}))
        means = sorted(float(m) for m in members)
        return _MixtureLaw(means, 2.0 if means == [0.0] else 1.0)


# --- the origin an answer set carries ---------------------------------------


def test_a_derived_set_says_so_and_names_the_map_it_came_from():
    answers = AnswerSet.from_actions(STATES, alarm)
    assert answers.origin.kind == "derived"
    assert answers.origin.states == 3
    assert answers.origin.action

    # The fingerprint is of the MAP, so it is stable across calls and differs
    # when the grouping differs. That is what makes it usable by a third party
    # holding two records and no code.
    again = AnswerSet.from_actions(STATES, alarm)
    assert again.origin.action == answers.origin.action
    assert AnswerSet.from_actions(STATES, three_ways).origin.action != answers.origin.action


def test_the_fingerprint_does_not_depend_on_the_order_the_states_arrived_in():
    # Two callers with the same map and a different iteration order must get the
    # same digest, or the claim that a shared digest means a shared grouping is
    # false and a reader comparing two records is misled.
    forward = AnswerSet.from_actions(("empty", "still", "moving"), alarm)
    backward = AnswerSet.from_actions(("moving", "still", "empty"), alarm)
    assert forward.origin.action == backward.origin.action


def test_the_fingerprint_separates_the_state_from_the_action():
    # Without a separator between the two fields these two maps hash the same
    # bytes: repr(1) + repr(23) and repr(12) + repr(3) are both "123". Different
    # groupings would then share a digest, which is the one thing it must not do.
    left = AnswerSet.from_actions((1, 4), lambda s: 23 if s == 1 else 56)
    right = AnswerSet.from_actions((12, 45), lambda s: 3 if s == 12 else 6)
    assert left.origin.action != right.origin.action


def test_a_set_written_by_hand_is_recorded_as_declared():
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in ("absent", "present")))
    assert answers.origin.kind == "declared"
    assert answers.origin.states is None
    assert answers.origin.action is None


def test_origin_does_not_change_what_the_question_is():
    # Two sets holding the same answers are the same question however they were
    # built. Making equality depend on provenance would let a caller settle a
    # disagreement by rebuilding the set.
    pair = tuple(Answer(n, (n,)) for n in ("absent", "present"))
    assert AnswerSet(pair) == AnswerSet(pair, origin=QuotientOrigin(kind="derived"))
    assert hash(AnswerSet(pair)) == hash(AnswerSet(pair, origin=QuotientOrigin("derived")))


def test_narrowing_keeps_the_map_the_set_came_from():
    answers = AnswerSet.from_actions(("empty", "still", "moving"), three_ways)
    narrowed = answers.narrowed(ruled_out="quiet")
    assert narrowed.origin == answers.origin


def test_an_origin_that_is_not_an_origin_is_refused():
    pair = tuple(Answer(n, (n,)) for n in ("absent", "present"))
    with pytest.raises(AnswerSetError, match="QuotientOrigin"):
        AnswerSet(pair, origin="derived")


# --- the check itself --------------------------------------------------------


def test_a_set_built_from_the_policy_agrees_with_it():
    answers = AnswerSet.from_actions(STATES, alarm)
    report = quotient_check(answers, alarm, STATES)
    assert report.agrees
    assert report.states_checked == 3
    assert report.splits == ()
    assert "matches the policy" in report.reason()


def test_the_motion_sensor_case_is_caught():
    # The canonical C1 failure: a still occupant and a moving one are one answer
    # `occupied`, while the deployment does different things in the two. sigma
    # over this set is arithmetic on a coarser question than the one asked.
    answers = AnswerSet.from_actions(STATES, alarm)
    report = quotient_check(answers, three_ways, STATES)

    assert not report.agrees
    assert len(report.splits) == 1
    split = report.splits[0]
    assert split.answer == "ring"
    assert {split.left, split.right} == {"still", "moving"}
    assert {split.left_action, split.right_action} == {"check", "ring"}
    assert "coarser than the deployment" in report.reason()


def test_one_witness_per_pair_of_actions_not_per_pair_of_states():
    # An answer holding a thousand states of two kinds has one disagreement to
    # report. Reporting half a million pairs would bury it.
    states = [f"s{i}" for i in range(1000)] + ["z"]
    answers = AnswerSet.from_actions(states, lambda s: "other" if s == "z" else "bulk")

    def parity(state):
        if state == "z":
            return "leave"
        return "even" if int(state[1:]) % 2 == 0 else "odd"

    report = quotient_check(answers, parity, states)
    assert len(report.splits) == 1
    assert report.splits[0].answer == "bulk"
    assert report.states_checked == 1001


def test_a_set_finer_than_the_policy_is_a_failure_too():
    # This was written the other way round, on the argument that refining only
    # adds pairs to a minimum so sigma can only fall. That argument counts the
    # pairs and forgets the surviving pairs change: a coarse answer's law is the
    # MIXTURE of the fine laws inside it. See the test below for the numbers.
    answers = AnswerSet.from_actions(STATES, three_ways)
    report = quotient_check(answers, alarm, STATES)
    assert not report.agrees
    assert report.splits == ()
    assert len(report.merges) == 1
    assert report.merges[0].action == "ring"
    assert report.merges[0].answers == ("check", "ring")
    assert "finer than the deployment" in report.reason()


def test_a_set_finer_than_the_policy_can_manufacture_standing():
    """Why a merge is not the conservative direction, in numbers.

    Two states the policy treats alike, at N(-2,1) and N(+2,1), against a rival
    at N(0,2). The question the consumer actually asks compares the MIXTURE with
    the rival. The finer declared set never forms that mixture, so its minimum
    is taken over component pairs, every one of which is further from the rival
    than the mixture is. At a floor of 0.25 the real question carries and the
    declared one is admitted.

    This is the convex-hull failure wearing another costume: a situation can be
    separable from every rival individually and still be a blend of them.
    """
    coarse = AnswerSet((Answer("same", frozenset({-2.0, 2.0})), Answer("rival", 0.0)))
    fine = AnswerSet((
        Answer("low", frozenset({-2.0})), Answer("high", frozenset({2.0})),
        Answer("rival", 0.0),
    ))
    conditions = Conditions({"windows": 96})
    asked = sigma(coarse, Mixture(), conditions, rng=0, draws=200_000).value
    declared = sigma(fine, Mixture(), conditions, rng=0, draws=200_000).value

    assert asked < 0.25 <= declared, (asked, declared)
    assert declared > 3.0 * asked

    # And the check that should have refused it does.
    report = quotient_check(fine, lambda s: "act" if s in (-2.0, 2.0) else "wait",
                            (-2.0, 2.0, 0.0))
    assert not report.agrees and len(report.merges) == 1


def test_the_message_names_the_first_merge_in_the_sorted_order():
    # Every other test here has exactly one merge, so nothing could see which
    # one reason() names. A mutation sweep found the line unpinned: reporting
    # merges[-1] instead of merges[0] survived the whole suite.
    states = ("a1", "a2", "b1", "b2")
    answers = AnswerSet.from_actions(states, lambda s: s)
    report = quotient_check(answers, lambda s: "zeta" if s[0] == "a" else "alpha", states)

    assert not report.agrees
    assert [m.action for m in report.merges] == ["alpha", "zeta"]
    assert report.merges[0].answers == ("b1", "b2")
    assert "alpha" in report.reason() and "zeta" not in report.reason()


def test_a_state_in_no_answer_is_reported():
    answers = AnswerSet.from_actions(("empty", "still"), alarm)
    report = quotient_check(answers, alarm, ("empty", "still", "moving"))
    assert not report.agrees
    assert report.unplaced == ("moving",)
    assert "in no answer" in report.reason()


def test_a_state_in_two_answers_is_reported_rather_than_resolved():
    # A frozenset is legal both as a region and as one opaque state. Where the
    # two readings disagree the check says so instead of preferring one.
    region = frozenset({"still", "moving"})
    answers = AnswerSet((Answer("ring", region), Answer("quiet", frozenset({region}))))
    report = quotient_check(answers, lambda s: "x", (region, "still"))
    assert not report.agrees
    assert any("quiet" in entry and "ring" in entry for entry in report.ambiguous)


def test_the_check_is_only_as_wide_as_the_states_it_was_given():
    # Agreement on a sample is agreement on a sample. The width is reported
    # beside the verdict so that a reader cannot mistake one for the other.
    answers = AnswerSet.from_actions(STATES, alarm)
    report = quotient_check(answers, three_ways, ("empty", "still"))
    assert report.agrees
    assert report.states_checked == 2


# --- what it refuses ---------------------------------------------------------


def test_a_policy_that_is_not_callable_is_refused():
    answers = AnswerSet.from_actions(STATES, alarm)
    # Matched on the whole phrase deliberately. A dict is not callable either
    # way, so calling it inside the loop raises TypeError and the error text
    # still contains the word "callable": a looser match passed with the guard
    # deleted, which is a test that cannot fail.
    with pytest.raises(InvalidInput, match="policy must be callable"):
        quotient_check(answers, {"empty": "quiet"}, STATES)


def test_a_check_with_no_pair_in_it_is_refused():
    answers = AnswerSet.from_actions(STATES, alarm)
    with pytest.raises(InvalidInput, match="at least two states"):
        quotient_check(answers, alarm, ("empty",))


def test_a_policy_that_fails_on_a_state_names_the_state():
    answers = AnswerSet.from_actions(STATES, alarm)
    with pytest.raises(InvalidInput, match="'unknown'"):
        quotient_check(answers, three_ways, ("empty", "unknown"))


def test_an_unhashable_action_is_refused():
    answers = AnswerSet.from_actions(STATES, alarm)
    with pytest.raises(InvalidInput, match="unhashable"):
        quotient_check(answers, lambda s: [s], STATES)


# --- what the record carries -------------------------------------------------


def build(check=None, answer_set=None, policy=alarm):
    answers = answer_set or AnswerSet.from_actions(STATES, policy)
    model = Shifted({"quiet": 0.0, "ring": 3.0, "check": 1.5})
    result = sigma(answers, model, Conditions({"windows": 96}))
    ruling = decide(result, floor=0.25)
    return Record.build(
        claim="occupied", interval="2026-08-30", ruling=ruling, result=result,
        answer_set=answers, check=check,
    )


def test_a_record_with_no_partition_block_says_nothing_was_declared(capsys, tmp_path):
    answers = AnswerSet.from_actions(STATES, alarm)
    model = Shifted({"quiet": 0.0, "ring": 3.0})
    result = sigma(answers, model, Conditions({"windows": 96}))
    record = Record.build(
        claim="occupied", interval="2026-08-30",
        ruling=decide(result, floor=0.25), result=result,
    )
    assert record.answer_map is None
    assert verify(record) == []

    path = tmp_path / "r.json"
    path.write_text(record.to_json())
    main(["verify", str(path)])
    assert "partition not declared" in capsys.readouterr().out


def test_the_record_carries_the_origin_and_round_trips():
    record = build(check=quotient_check(AnswerSet.from_actions(STATES, alarm), alarm, STATES))
    assert record.answer_map["kind"] == "derived"
    assert record.answer_map["checked"]["agrees"] is True
    assert verify(record) == []
    assert Record.from_json(record.to_json()) == record


def test_derived_and_checked_are_not_the_same_record():
    # The whole point of the block: "built from a policy" and "verified against
    # that policy" must not look alike to a consumer.
    assert build().answer_map["checked"] is None
    assert build(check=quotient_check(
        AnswerSet.from_actions(STATES, alarm), alarm, STATES)) .answer_map["checked"] is not None


def test_admitting_over_a_partition_the_policy_contradicts_is_caught():
    record = build(check=quotient_check(AnswerSet.from_actions(STATES, alarm), three_ways, STATES))
    problems = verify(record)
    assert any("different question from the one the deployment asks" in p for p in problems)


def test_a_block_claiming_agreement_over_its_own_splits_is_caught():
    # `agrees` re-derives from the disagreements listed beside it, so a tampered
    # record is caught with no policy, no states and no model in the room.
    record = build(check=quotient_check(AnswerSet.from_actions(STATES, alarm), three_ways, STATES))
    payload = json.loads(record.to_json())
    payload["answer_map"]["checked"]["agrees"] = True
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("contradicts itself" in p for p in problems)


def test_a_check_reporting_fewer_than_two_states_is_caught():
    record = build(check=quotient_check(AnswerSet.from_actions(STATES, alarm), alarm, STATES))
    payload = json.loads(record.to_json())
    payload["answer_map"]["checked"]["states_checked"] = 1
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("cannot disagree" in p for p in problems)


def test_an_unknown_kind_is_caught():
    record = build()
    payload = json.loads(record.to_json())
    payload["answer_map"]["kind"] = "vouched"
    assert any("not one of" in p for p in verify(Record.from_json(json.dumps(payload))))


def test_the_provenance_of_a_different_partition_is_refused():
    # Provenance for the wrong partition is worse than none: a third party would
    # act on a fingerprint that describes a question this claim never faced.
    answers = AnswerSet.from_actions(STATES, alarm)
    model = Shifted({"quiet": 0.0, "ring": 3.0})
    result = sigma(answers, model, Conditions({"windows": 96}))
    other = AnswerSet.from_actions(("a", "b"), lambda s: f"act-{s}")
    with pytest.raises(InvalidInput, match="actually scored on"):
        Record.build(
            claim="occupied", interval="2026-08-30",
            ruling=decide(result, floor=0.25), result=result, answer_set=other,
        )


def test_a_recruitment_may_carry_the_set_it_was_narrowed_from():
    # decide() records the NARROWED names, so the set the caller is holding is
    # wider than the ruling. That is the one place a superset is legitimate, and
    # the guard has to let it through or every recruited claim loses its
    # provenance at exactly the point a reader most wants it.
    answers = AnswerSet.from_actions(STATES, three_ways)
    model = Shifted({"quiet": 0.0, "check": 1.5, "ring": 3.0})
    conditions = Conditions({"windows": 96})
    before = sigma(answers, model, conditions)
    lender = sigma(
        AnswerSet((Answer("seen", ("seen",)), Answer("unseen", ("unseen",)))),
        Shifted({"seen": 0.0, "unseen": 4.0}), conditions,
    )
    ruling = decide(
        before, floor=0.25, answers=answers,
        anchors=[Anchor("clock", "check", standing=lender, floor=0.25)],
        recompute=lambda narrowed: sigma(narrowed, model, conditions),
    )
    assert ruling.rule.value == "recruit"
    assert set(ruling.answers) < set(answers.names)

    record = Record.build(
        claim="occupied", interval="2026-08-30", ruling=ruling, result=before,
        answer_set=answers, check=quotient_check(answers, three_ways, STATES),
    )
    assert record.answer_map["kind"] == "derived"
    assert verify(record) == []


def test_misuse_of_the_new_arguments_is_a_sentence_not_a_stack_trace():
    answers = AnswerSet.from_actions(STATES, alarm)
    model = Shifted({"quiet": 0.0, "ring": 3.0})
    result = sigma(answers, model, Conditions({"windows": 96}))
    ruling = decide(result, floor=0.25)
    common = dict(claim="occupied", interval="2026-08-30", ruling=ruling, result=result)
    with pytest.raises(InvalidInput, match="must be an AnswerSet"):
        Record.build(**common, answer_set=("quiet", "ring"))
    with pytest.raises(InvalidInput, match="must be a QuotientCheck"):
        Record.build(**common, check={"agrees": True})


def test_the_cli_prints_the_disagreement(tmp_path, capsys):
    record = build(check=quotient_check(AnswerSet.from_actions(STATES, alarm), three_ways, STATES))
    path = tmp_path / "r.json"
    path.write_text(record.to_json())
    assert main(["verify", str(path)]) == 0
    out = capsys.readouterr().out
    assert "DISAGREES" in out
    assert "'still'" in out and "'moving'" in out


def test_a_check_without_the_set_it_ran_on_reads_as_the_weaker_record():
    answers = AnswerSet.from_actions(STATES, alarm)
    model = Shifted({"quiet": 0.0, "ring": 3.0})
    result = sigma(answers, model, Conditions({"windows": 96}))
    record = Record.build(
        claim="occupied", interval="2026-08-30",
        ruling=decide(result, floor=0.25), result=result,
        check=quotient_check(answers, alarm, STATES),
    )
    assert record.answer_map["kind"] == "unknown"
    assert verify(record) == []
