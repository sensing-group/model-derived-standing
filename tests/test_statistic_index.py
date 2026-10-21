"""Which statistic a sigma is over, and the one ordering a stranger can check.

Recording the value and not the index let two objects travel under one symbol:
a ceiling over the joint reading of the live sensors, and the operational value
over the scalar the pipeline thresholds. For one claim those can be 0.9933 and
0.2212, which is the difference between "this hardware cannot do it" and "your
fusion is throwing it away", and a consumer could not tell them apart.

The interesting property is that their ORDER is a theorem. The pooled statistic
is a function of the joint one, so T6 forbids the operational value exceeding
its own ceiling, and a record claiming otherwise is caught with no model and no
data.
"""

from __future__ import annotations

import json

import pytest

from standing.answers import Answer, AnswerSet
from standing.cli import main
from standing.errors import InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.record import Record, Statistic, verify
from standing.rules import decide
from standing.sigma import sigma

C = Conditions({"windows": 96})


class Shifted:
    def __init__(self, offsets):
        self.offsets = offsets

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], 1.0)


OFFSETS = {"absent": 0.0, "present": 3.0}
ANSWERS = AnswerSet(tuple(Answer(n, (n,)) for n in OFFSETS))


def build(statistic=None) -> Record:
    result = sigma(ANSWERS, Shifted(OFFSETS), C)
    return Record.build(
        claim="present", interval="2026-08-30",
        ruling=decide(result, floor=0.25), result=result, statistic=statistic,
    )


# --- the index itself --------------------------------------------------------


def test_a_record_that_does_not_say_reads_as_not_declared():
    record = build()
    assert record.statistic is None
    assert verify(record) == []


def test_the_kind_and_the_name_travel():
    record = build(Statistic(kind="operational", name="window_mean"))
    assert record.statistic["kind"] == "operational"
    assert record.statistic["name"] == "window_mean"
    assert record.statistic["ceiling"] is None and record.statistic["gap"] is None
    assert verify(record) == []
    assert Record.from_json(record.to_json()) == record


def test_the_gap_is_derived_and_not_passed():
    # Every other number in a record is recomputed rather than trusted, and the
    # gap is no exception: the caller supplies the ceiling and nothing else.
    record = build(Statistic(kind="operational", name="pooled", ceiling=0.9933))
    value = record.sigma["value"]
    assert record.statistic["gap"] == pytest.approx(0.9933 - value)
    assert verify(record) == []


def test_a_guessed_kind_is_refused():
    with pytest.raises(InvalidInput, match="must be one of"):
        Statistic(kind="joint")


def test_unknown_is_allowed_because_it_is_honest():
    record = build(Statistic(kind="unknown"))
    assert record.statistic["kind"] == "unknown"
    assert verify(record) == []


def test_a_ceiling_record_may_not_carry_its_own_ceiling():
    with pytest.raises(InvalidInput, match="already the ceiling"):
        Statistic(kind="ceiling", ceiling=0.5)


def test_a_ceiling_outside_the_unit_interval_is_refused():
    for bad in (1.5, -0.1, float("nan")):
        with pytest.raises(InvalidInput):
            Statistic(kind="operational", ceiling=bad)


def test_something_that_is_not_a_statistic_is_refused():
    with pytest.raises(InvalidInput, match="must be a Statistic"):
        build(statistic={"kind": "operational"})


# --- the ordering, which is the part worth having ----------------------------


def test_an_operational_value_above_its_own_ceiling_contradicts_T6():
    """The one check here that is a theorem rather than a convention.

    R is a function of J, so processing cannot create standing. A record whose
    operational value exceeds its ceiling is not merely odd, it contradicts the
    data processing inequality, and that is visible to a reader holding no model
    and no data.
    """
    record = build(Statistic(kind="operational", name="pooled", ceiling=0.9))
    payload = json.loads(record.to_json())
    payload["statistic"]["ceiling"] = 0.10          # below the sigma
    payload["statistic"]["gap"] = 0.10 - payload["sigma"]["value"]
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("T6 forbids this" in p for p in problems)


def test_the_ordering_holds_at_equality():
    # A pipeline that throws nothing away sits exactly on its ceiling, and that
    # is legal: T6 is an inequality, not a strict one.
    result = sigma(ANSWERS, Shifted(OFFSETS), C)
    record = Record.build(
        claim="c", interval="i", ruling=decide(result, floor=0.25), result=result,
        statistic=Statistic(kind="operational", ceiling=result.value),
    )
    assert record.statistic["gap"] == pytest.approx(0.0)
    assert verify(record) == []


def test_a_tampered_gap_is_caught():
    record = build(Statistic(kind="operational", ceiling=0.9933))
    payload = json.loads(record.to_json())
    payload["statistic"]["gap"] = 0.01
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("ceiling minus" in p for p in problems)


def test_a_gap_with_no_ceiling_is_caught():
    record = build(Statistic(kind="operational"))
    payload = json.loads(record.to_json())
    payload["statistic"]["gap"] = 0.4
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("no ceiling to derive it from" in p for p in problems)


def test_a_ceiling_beside_a_ceiling_record_is_caught_on_read():
    # The constructor refuses it, so a record can only arrive this way by being
    # written by someone else. The verifier is the reader's defence.
    record = build(Statistic(kind="operational", ceiling=0.9933))
    payload = json.loads(record.to_json())
    payload["statistic"]["kind"] = "ceiling"
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("belongs beside an operational one" in p for p in problems)


def test_an_unknown_kind_on_read_is_caught():
    record = build(Statistic(kind="operational"))
    payload = json.loads(record.to_json())
    payload["statistic"]["kind"] = "vouched"
    assert any("not one of" in p for p in verify(Record.from_json(json.dumps(payload))))


def test_a_malformed_statistic_block_is_reported_not_raised():
    record = build(Statistic(kind="operational", ceiling=0.9933))
    for mangle in (
        lambda s: s.update(ceiling="high"),
        lambda s: s.update(gap="wide"),
    ):
        payload = json.loads(record.to_json())
        mangle(payload["statistic"])
        assert verify(Record.from_json(json.dumps(payload))), "must report, not raise"


# --- what a reader sees ------------------------------------------------------


def test_the_cli_says_when_nobody_declared_which_statistic(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(build().to_json())
    main(["verify", str(path)])
    assert "which statistic this is: not declared" in capsys.readouterr().out


def test_the_cli_prints_the_ceiling_and_the_gap(tmp_path, capsys):
    path = tmp_path / "r.json"
    path.write_text(build(Statistic(kind="operational", name="pooled",
                                    ceiling=0.9933)).to_json())
    assert main(["verify", str(path)]) == 0
    out = capsys.readouterr().out
    assert "operational" in out and "'pooled'" in out
    assert "ceiling 0.9933" in out and "gap" in out


def _recruit_ruling():
    """A recruitment, where the answer set narrows and sigma rises with it."""
    from standing.rules import Anchor

    offsets = {"absent": 0.0, "present": 3.0, "confounded": 0.05}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    before = sigma(answers, Shifted(offsets), C)
    lender = sigma(
        AnswerSet((Answer("seen", ("s",)), Answer("unseen", ("u",)))),
        Shifted({"seen": 0.0, "unseen": 4.0}), C,
    )
    ruling = decide(
        before, floor=0.25, answers=answers,
        anchors=[Anchor("clock", "confounded", standing=lender, floor=0.25)],
        recompute=lambda narrowed: sigma(narrowed, Shifted(offsets), C),
    )
    assert ruling.rule.value == "recruit"
    return before, ruling


def test_a_ceiling_on_a_recruit_is_refused_because_the_answer_set_moved():
    """The defect this guard exists for, measured.

    On a recruit the record's value is over the FULL answer set and the claim
    rests on `after_recruitment` over the narrowed one: 0.0003 against 0.6753 in
    this fixture. T6 orders a ceiling against a value only when both are over
    the same set, and T2 says narrowing RAISES sigma, so a full-set ceiling can
    be legitimately exceeded after recruitment. Before this guard a ceiling of
    0.30 verified clean while the claim rested on 0.6753 above it.
    """
    before, ruling = _recruit_ruling()
    assert ruling.sigma < 0.01 < 0.6 < ruling.sigma_after

    with pytest.raises(InvalidInput, match="cannot be recorded on a RECRUIT"):
        Record.build(claim="c", interval="i", ruling=ruling, result=before,
                     statistic=Statistic(kind="operational", ceiling=0.30))


def test_a_recruit_may_still_say_which_statistic_it_is():
    # The kind is unambiguous under narrowing; only the ceiling is not.
    before, ruling = _recruit_ruling()
    record = Record.build(claim="c", interval="i", ruling=ruling, result=before,
                          statistic=Statistic(kind="operational", name="pooled"))
    assert record.statistic["ceiling"] is None
    assert verify(record) == []


def test_a_ceiling_on_a_recruit_written_elsewhere_is_caught_on_read():
    before, ruling = _recruit_ruling()
    record = Record.build(claim="c", interval="i", ruling=ruling, result=before,
                          statistic=Statistic(kind="operational"))
    payload = json.loads(record.to_json())
    payload["statistic"]["ceiling"] = 0.30
    payload["statistic"]["gap"] = 0.30 - payload["sigma"]["value"]
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("recorded on a recruit" in p for p in problems)
