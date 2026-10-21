"""The record must replay into its own claim, and say so when it cannot."""

from __future__ import annotations

import json

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.certify import certify, split
from standing.cli import main
from standing.drift import bracket
from standing.errors import InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.record import SCHEMA, Record, verify
from standing.rules import Rule, decide
from standing.sigma import sigma


class Shifted:
    def __init__(self, offsets):
        self.offsets = offsets

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], 1.0)


C = Conditions({"windows": 96, "weight": 0.25})


def build_record() -> Record:
    offsets = {"absent": 0.0, "present": 3.0}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    model = Shifted(offsets)
    result = sigma(answers, model, C)

    rng = np.random.default_rng(3)
    selection, scoring = {}, {}
    for name, mean in offsets.items():
        selection[name], scoring[name] = split(rng.normal(mean, 1.0, 4000), rng)
    _, certificates = certify(answers, selection, scoring, source="readings")
    # The honest path: the ruling is made WITH the certificates, so the record
    # says admitted and carries the rule a stranger can re-apply.
    ruling = decide(result, floor=0.25, certificates=certificates)

    return Record.build(
        claim="episode present",
        interval="2026-08-24",
        ruling=ruling,
        result=result,
        certificates=certificates,
        conditions=C.fields,
        provenance={"library": "standing 0.0.1"},
    )


def test_a_record_round_trips_through_json():
    record = build_record()
    restored = Record.from_json(record.to_json())
    assert restored == record
    assert json.loads(record.to_json())["schema"] == "standing/record/6"


def test_a_good_record_verifies_with_no_model_and_no_data():
    assert verify(build_record()) == []


def test_verification_catches_a_tampered_certificate():
    record = build_record()
    payload = json.loads(record.to_json())
    payload["certificates"][0]["value"] = 0.99
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("its own rates give" in p for p in problems)


def test_verification_catches_a_tampered_bracket():
    band = bracket(0.9019, 0.1710)
    record = build_record()
    payload = json.loads(record.to_json())
    payload["bracket"] = {
        "low": 0.60,  # wrong
        "high": band.high,
        "measured": band.measured,
        "null": band.null,
    }
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("bracket low" in p for p in problems)


def test_verification_catches_a_disposition_that_does_not_follow():
    record = build_record()
    payload = json.loads(record.to_json())
    payload["sigma"]["value"] = 0.01
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("implies 'carry'" in p for p in problems)


def test_abstain_is_reserved_for_exact_zero():
    record = build_record()
    payload = json.loads(record.to_json())
    payload["rule"] = "abstain"
    payload["sigma"]["value"] = 0.004
    problems = verify(Record.from_json(json.dumps(payload)))
    assert any("reserved for sigma exactly zero" in p for p in problems)


def test_an_unknown_schema_is_refused():
    with pytest.raises(InvalidInput, match="unknown record schema"):
        Record.from_json(json.dumps({"schema": "someone-elses/1"}))


def test_a_record_without_its_answer_set_is_flagged():
    record = build_record()
    payload = json.loads(record.to_json())
    payload["answers"] = ["only"]
    assert any("at least two" in p for p in verify(Record.from_json(json.dumps(payload))))


def test_the_cli_verifies_a_record_and_reports_success(tmp_path, capsys):
    path = tmp_path / "record.json"
    path.write_text(build_record().to_json())
    assert main(["verify", str(path)]) == 0
    assert "re-derives from the record itself" in capsys.readouterr().out


def test_the_cli_fails_strictly_on_a_tampered_record(tmp_path, capsys):
    payload = json.loads(build_record().to_json())
    payload["certificates"][0]["value"] = 0.99
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    assert main(["verify", str(path), "--strict"]) == 1
    assert "discrepancy" in capsys.readouterr().err


def test_the_cli_rejects_a_file_that_is_not_a_record(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text("not json at all")
    assert main(["verify", str(path)]) == 2


def test_the_cli_reports_deployment_level_rates(tmp_path, capsys):
    import json as _json

    for index, rule in enumerate(("admitted", "carry", "abstain", "admitted")):
        payload = _json.loads(build_record().to_json())
        payload["rule"] = rule
        if rule == "carry":
            payload["sigma"]["value"] = 0.1
        if rule == "abstain":
            payload["sigma"]["value"] = 0.0
            payload["certificates"] = []
        (tmp_path / f"r{index}.json").write_text(_json.dumps(payload))

    paths = [str(p) for p in sorted(tmp_path.glob("*.json"))]
    assert main(["report", *paths]) == 0
    out = capsys.readouterr().out
    assert "records                     4" in out
    assert "entitled                     50.0%" in out
    assert "admissible, uncertified       0.0%" in out
    assert "refusals needing no floor    25.0%" in out


def test_the_report_needs_at_least_one_readable_record(tmp_path):
    (tmp_path / "junk.json").write_text("nonsense")
    assert main(["report", str(tmp_path / "junk.json")]) == 2


def test_a_malformed_record_is_reported_not_raised():
    """The verifier is handed records it did not write. It must never crash."""
    import json as _json

    for mangle in (
        lambda p: p.update(sigma={}),
        lambda p: p.update(sigma={"value": "not a number"}),
        lambda p: p.update(floor="high"),
        lambda p: p.update(bracket={"low": 0.1}),
        lambda p: p.update(bracket={"measured": 2.0, "null": 0.1, "low": 0, "high": 1}),
        lambda p: p.update(certificates=[{"pair": ["a", "b"], "value": 0.5}]),
    ):
        payload = _json.loads(build_record().to_json())
        mangle(payload)
        problems = verify(Record.from_json(_json.dumps(payload)))
        assert problems, "a malformed record must produce a discrepancy"


def test_a_bracket_ruling_produces_a_record_that_verifies():
    """The drift path is the one every moving deployment uses; it must verify."""
    from standing.drift import bracket as make_bracket

    offsets = {"absent": 0.0, "present": 3.0}
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    band = make_bracket(0.9019, 0.1710)
    ruling = decide(band, floor=0.25, answers=answers)
    record = Record.build(claim="c", interval="i", ruling=ruling, band=band)
    assert record.answers == answers.names
    assert verify(record) == []


def test_a_record_carrying_a_bracket_must_have_been_ruled_on_that_bracket():
    """Otherwise the record fails its own verifier the moment they straddle the floor.

    The worked example did exactly this: sigma 0.3262 admitted, bracket low
    0.2334, floor 0.2500, and verify() reported the record as inconsistent with
    itself. Catch it where the record is built, not where it is read.
    """
    from standing.drift import bracket
    from standing.rules import Rule, Ruling

    band = bracket(0.3262, 0.0078)
    ruled_on_sigma = Ruling(
        rule=Rule.ADMISSIBLE,
        admitted=False,
        sigma=0.3262,
        floor=0.25,
        answers=("a", "b"),
        binding_pair=("a", "b"),
    )
    with pytest.raises(InvalidInput, match="but its ruling was made on"):
        Record.build(claim="c", interval="i", ruling=ruled_on_sigma, band=band)

    # Ruled on the bracket, at a floor its low end actually clears.
    ruled_on_band = Ruling(
        rule=Rule.ADMISSIBLE,
        admitted=False,
        sigma=band.low,
        floor=0.20,
        answers=("a", "b"),
        binding_pair=("a", "b"),
    )
    record = Record.build(claim="c", interval="i", ruling=ruled_on_band, band=band)
    assert verify(record) == []


def test_the_verifier_reports_a_malformed_bracket_instead_of_raising():
    """verify() is handed records it did not write, including broken ones."""
    payload = {
        "schema": SCHEMA,
        "claim": "x",
        "interval": "i",
        "answers": ["a", "b"],
        "floor": 0.25,
        "rule": "admissible",
        "sigma": {"value": 0.9},
        "bracket": {"low": "x", "high": 1.0, "measured": 0.9, "null": 0.1},
        "certificates": [],
        "conditions": [],
        "provenance": {},
    }
    problems = verify(Record.from_json(json.dumps(payload)))
    assert problems and any("bracket" in p for p in problems)


BROKEN_RECORDS = {
    "empty_sigma": {"sigma": {}},
    "unusable_floor": {"floor": "not-a-number"},
    "unusable_bracket": {"bracket": {"low": "x", "high": 1.0, "measured": 0.9, "null": 0.1}},
    "certificate_not_an_object": {"certificates": ["not-an-object"]},
    "certificate_missing_everything": {"certificates": [{"pair": "ab"}]},
}


@pytest.mark.parametrize("name", sorted(BROKEN_RECORDS))
def test_the_cli_reports_a_malformed_record_instead_of_dying_on_it(tmp_path, name):
    """The CLI is the third-party audit entry point, so it gets hostile input.

    `verify` was hardened for exactly this and the PRINTING was not, so a record
    with an empty sigma block took the whole command down with a KeyError while
    the discrepancy list that would have explained it had already been computed.
    """
    payload = {
        "schema": SCHEMA,
        "claim": "x",
        "interval": "i",
        "answers": ["a", "b"],
        "floor": 0.25,
        "rule": "admissible",
        "sigma": {"value": 0.9},
        "bracket": None,
        "certificates": [],
        "conditions": [],
        "provenance": {},
        **BROKEN_RECORDS[name],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    assert main(["verify", str(path), "--strict"]) == 1


class _Mixture:
    def __init__(self, means):
        self.means = means

    def sample(self, n, rng):
        pick = rng.integers(0, len(self.means), n)
        return np.asarray(self.means, dtype=float)[pick] + rng.normal(0.0, 1.0, n)


class _MotionOnly:
    """A still occupant and an empty room look identical to this view."""

    LOCI = {"empty": 0.0, "occupied_still": 0.0, "occupied_moving": 4.0}

    def law(self, answer, conditions):
        members = sorted(answer.state) if isinstance(answer.state, frozenset) else [answer.state]
        if len(members) == 1:
            return GaussianLaw(self.LOCI[members[0]], 1.0)
        return _Mixture([self.LOCI[m] for m in members])


def _audited_claim():
    """A claim admitted over a structural collapse, with the audit attached.

    Built locally rather than imported from another test module: a cross-test
    import makes `tests` look like an undeclared dependency to the packaging
    check, which is correct, because following the README's install command
    would give an environment where these files cannot be collected.
    """
    from standing.checks import quotient_audit

    conditions = Conditions({"spread": 1.0})
    answers = AnswerSet.from_actions(
        ["empty", "occupied_still", "occupied_moving"],
        lambda s: "no-alarm" if s == "empty" else "alarm",
    )
    states = {a.name: {str(m): m for m in sorted(a.state)} for a in answers.answers}
    model = _MotionOnly()
    result = sigma(answers, model, conditions, draws=20_000, rng=0)
    audit = quotient_audit(
        answers, model, conditions, floor=0.05, delta=0.5, states=states,
        result=result, draws=20_000, rng=0,
    )
    return result, audit, decide(result, floor=0.05)


def test_a_claim_admitted_over_a_collapse_no_longer_verifies_clean():
    """The defect the schema change exists for.

    A record carrying one standing per claim could say ADMIT while the audit
    said the negative verdicts were unentitled, and verify() returned nothing.
    """
    result, audit, ruling = _audited_claim()
    assert ruling.rule is Rule.ADMISSIBLE and audit.collapses
    record = Record.build(claim="occupancy", interval="k", ruling=ruling,
                          result=result, audit=audit)
    problems = verify(record)
    assert any("collapse" in p for p in problems)


def test_not_audited_is_recorded_as_not_audited():
    """Absence must read as absence, not as a clean bill of health."""
    result, _, ruling = _audited_claim()
    record = Record.build(claim="occupancy", interval="k", ruling=ruling, result=result)
    assert record.quotient is None
    assert verify(record) == []


def test_an_audit_against_another_floor_is_not_about_this_claim():
    result, audit, ruling = _audited_claim()
    record = Record.build(claim="occupancy", interval="k", ruling=ruling,
                          result=result, audit=audit)
    payload = json.loads(record.to_json())
    payload["quotient"]["floor"] = 0.9
    assert any("different question" in p
               for p in verify(Record.from_json(json.dumps(payload))))


def test_a_recruitment_must_record_its_chain():
    """A derived claim and a grounded one used to look identical."""
    from standing.rules import Ruling

    orphan = Ruling(rule=Rule.RECRUIT, admitted=True, sigma=0.01, sigma_after=0.9,
                    floor=0.25, answers=("a", "b"), binding_pair=("a", "b"))
    assert any("no chain is recorded" in p
               for p in verify(Record.build(claim="c", interval="i", ruling=orphan)))


def test_a_recruitment_cannot_claim_rank_zero():
    from standing.rules import Ruling

    shallow = Ruling(rule=Rule.RECRUIT, admitted=True, sigma=0.01, sigma_after=0.9,
                     floor=0.25, answers=("a", "b"), binding_pair=("a", "b"),
                     rank=0, recruited="anchor")
    assert any("at least one deeper" in p
               for p in verify(Record.build(claim="c", interval="i", ruling=shallow)))


@pytest.mark.parametrize("broken", [
    {"floor": "x", "collapses": []},
    {"collapses": "not-a-list"},
    "not-an-object",
    {"floor": 0.05, "collapses": ["not-an-object"]},
])
def test_the_verifier_reports_a_malformed_quotient_instead_of_raising(broken):
    result, audit, ruling = _audited_claim()
    record = Record.build(claim="c", interval="k", ruling=ruling, result=result, audit=audit)
    payload = json.loads(record.to_json())
    payload["quotient"] = broken
    assert verify(Record.from_json(json.dumps(payload)))    # reports, does not raise


@pytest.mark.parametrize("bad", ["not-an-audit", 42, {"answer_sigma": 0.5}])
def test_something_that_is_not_an_audit_is_refused(bad):
    _, _, ruling = _audited_claim()
    with pytest.raises(InvalidInput, match="must be a QuotientAudit"):
        Record.build(claim="c", interval="i", ruling=ruling, audit=bad)


@pytest.mark.parametrize("bad", ["x", 42, [1, 2]])
def test_a_span_that_is_not_a_mapping_is_refused(bad):
    _, _, ruling = _audited_claim()
    with pytest.raises(InvalidInput, match="span must be a mapping"):
        Record.build(claim="c", interval="i", ruling=ruling, span=bad)


def test_a_v1_record_is_refused():
    """The new blocks changed what a record means, so the version moved."""
    _, _, ruling = _audited_claim()
    payload = json.loads(Record.build(claim="c", interval="i", ruling=ruling).to_json())
    payload["schema"] = "standing/record/1"
    with pytest.raises(InvalidInput, match="unknown record schema"):
        Record.from_json(json.dumps(payload))


# --- the verifier re-derives every disposition, at its boundary ---------------


def _two(offsets):
    answers = AnswerSet(tuple(Answer(n, (n,)) for n in offsets))
    return answers, Shifted(offsets)


def test_carry_and_abstain_records_verify_clean():
    for offsets in ({"a": 0.0, "b": 0.4}, {"a": 0.0, "b": 0.0}):
        answers, model = _two(offsets)
        result = sigma(answers, model, C)
        ruling = decide(result, floor=0.25)
        record = Record.build(claim="c", interval="i", ruling=ruling, result=result)
        assert record.rule in ("carry", "abstain") and verify(record) == []


def test_a_record_with_sigma_exactly_at_the_floor_verifies_as_admissible():
    answers, model = _two({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    ruling = decide(result, floor=result.value)
    record = Record.build(claim="c", interval="i", ruling=ruling, result=result)
    assert record.rule == "admissible" and verify(record) == []


def test_an_admissible_record_with_certificates_below_the_floor_verifies_clean():
    """The certified bound is the WORST pair; the record must agree with the ruling."""
    offsets = {"a": 0.0, "b": 1.6, "c": 8.0}
    answers, model = _two(offsets)
    rng = np.random.default_rng(5)
    selection, scoring = {}, {}
    for name, mean in offsets.items():
        selection[name], scoring[name] = split(rng.normal(mean, 1.0, 3000), rng)
    _, certificates = certify(answers, selection, scoring, source="readings")
    result = sigma(answers, model, C)
    ruling = decide(result, floor=0.25, certificates=certificates)
    assert ruling.rule.value == "admissible" and max(c.value for c in certificates.values()) > 0.25
    record = Record.build(claim="c", interval="i", ruling=ruling, result=result, certificates=certificates)
    assert verify(record) == []


def test_a_recruit_record_is_re_derived_from_its_narrowed_value():
    from standing.rules import Anchor

    answers, model = _two({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    lender_answers, lender = _two({"seen": 0.0, "unseen": 4.0})
    result = sigma(answers, model, C)
    narrowed_value = sigma(answers.narrowed(ruled_out="confounded"), model, C).value
    for floor in (0.25, narrowed_value):                      # a floor below, and exactly at
        ruling = decide(
            result, floor=floor, answers=answers,
            anchors=[Anchor("clock", "confounded", standing=sigma(lender_answers, lender, C), floor=0.25)],
            recompute=lambda n: sigma(n, model, C),
        )
        assert ruling.rule.value == "recruit"
        record = Record.build(claim="c", interval="i", ruling=ruling, result=result)
        assert verify(record) == []
        payload = json.loads(record.to_json())
        payload["sigma"]["after_recruitment"] = floor - 0.01
        problems = verify(Record.from_json(json.dumps(payload)))
        assert any("implies 'carry'" in p for p in problems)
