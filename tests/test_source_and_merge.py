"""Two guards from adversarial round 2 (19 September 2026).

The first: the only certificate the CML run ever issued was scored on samples
drawn from the forward model, and the deck called it held-out data. So a
certificate now carries its source and a model-sourced one cannot admit.

The second: T14 protects a zero against a change of LAW, and every model in the
9-of-42 comparison read one shared locus map. A zero under a map that merges
two states the world keeps apart is authored, not structural, and nothing in
the audit looked in that direction. `check_merge` does.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from standing.answers import Answer, AnswerSet
from standing.audit import FAILED, NOT_RUN, PASSED, audit
from standing.certify import SOURCES, Certificate, certify, certify_pair, split
from standing.errors import CertificateError, InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.record import Record, verify
from standing.rules import Rule, decide, unmeasured_pairs
from standing.sigma import sigma

C = Conditions({"spread": 1.0})


class Placed:
    """A law that depends on the answer only through a locus map."""

    def __init__(self, loci):
        self.loci = loci

    def law(self, answer, conditions):
        return GaussianLaw(self.loci[answer.name], 1.0)


def build(loci):
    return AnswerSet(tuple(Answer(n, (n,)) for n in loci)), Placed(loci)


def samples_for(answers, model, source, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    selection, scoring = {}, {}
    for answer in answers.answers:
        selection[answer.name], scoring[answer.name] = split(model.law(answer, C).sample(n, rng), rng)
    return certify(answers, selection, scoring, source=source)


# ---------------------------------------------------------------------------
# the source of a certificate
# ---------------------------------------------------------------------------
def test_a_certificate_must_say_what_it_was_scored_on():
    rng = np.random.default_rng(0)
    ls, lz = split(rng.normal(0.0, 1.0, 400), rng)
    rs, rz = split(rng.normal(3.0, 1.0, 400), rng)
    with pytest.raises(TypeError):
        certify_pair("a", ls, lz, "b", rs, rz)  # type: ignore[call-arg]
    with pytest.raises(InvalidInput):
        certify_pair("a", ls, lz, "b", rs, rz, source="held-out")
    with pytest.raises(InvalidInput):
        Certificate(("a", "b"), 1.0, 0.5, 0.1, 0.9, 400, 400, source="simulated")
    assert SOURCES == ("readings", "model")


def test_a_model_sourced_certificate_leaves_the_claim_admissible():
    answers, model = build({"a": 0.0, "b": 3.0})
    bound, certs = samples_for(answers, model, "model")
    assert bound >= 0.25, "the samples separate; the point is that it does not matter"
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.rule is Rule.ADMISSIBLE
    assert ruling.admitted is False
    assert ruling.certificate == pytest.approx(bound)
    assert "model's own samples" in ruling.reason
    assert unmeasured_pairs(certs, answers.names) == [("a", "b")]


def test_a_readings_certificate_still_admits():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "readings")
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.rule is Rule.ADMITTED
    assert unmeasured_pairs(certs, answers.names) == []


def test_one_simulated_pair_is_enough_to_block_admission():
    answers, model = build({"a": 0.0, "b": 3.0, "c": 6.0})
    _, measured = samples_for(answers, model, "readings")
    _, simulated = samples_for(answers, model, "model")
    mixed = dict(measured)
    mixed[("b", "c")] = simulated[("b", "c")]
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=mixed)
    assert ruling.rule is Rule.ADMISSIBLE
    assert "1 of 3 open pair(s)" in ruling.reason


def test_the_certificate_check_fails_a_simulated_certificate_by_name():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "model")
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    with_mapping = audit(ruling, certificates=certs)["certificate"]
    assert with_mapping.status == FAILED
    assert "forward model" in with_mapping.reason
    # The same case is visible from the ruling alone: a certificate that
    # clears the floor on a ruling that did not admit.
    from_ruling = audit(ruling)["certificate"]
    assert from_ruling.status == FAILED
    assert "model's own samples" in from_ruling.reason


def test_an_admissible_ruling_offered_readings_later_is_not_misread():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "readings")
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert ruling.certificate is None
    assert audit(ruling, certificates=certs)["certificate"].status == PASSED


def test_the_record_carries_the_source_and_verify_reads_it():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "readings")
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    record = Record.build(claim="c", interval="t", ruling=ruling, certificates=certs)
    payload = json.loads(record.to_json())
    assert payload["schema"] == "standing/record/6"
    assert all(entry["source"] == "readings" for entry in payload["certificates"])
    assert verify(record) == []

    # Forge the one field that matters and the verifier objects.
    payload["certificates"][0]["source"] = "model"
    forged = Record.from_json(json.dumps(payload))
    problems = verify(forged)
    assert any("model's own samples" in p for p in problems)

    del payload["certificates"][0]["source"]
    unstated = Record.from_json(json.dumps(payload))
    assert any("does not say" in p for p in verify(unstated))


# ---------------------------------------------------------------------------
# the merge check
# ---------------------------------------------------------------------------
def test_an_authored_merge_is_split_by_an_alternative_map():
    # One map puts "still" and "empty" at one locus: a zero. The Fresnel-shaped
    # alternative puts a body inside the footprint at its own locus.
    coarse_answers, coarse = build({"empty": 0.0, "still": 0.0})
    ruling = decide(sigma(coarse_answers, coarse, C), floor=0.25)
    assert ruling.rule is Rule.ABSTAIN
    fine_answers, fine = build({"empty": 0.0, "still": 2.0})
    report = audit(ruling, conditions=C, alternatives=[(fine_answers, fine)])
    check = report["merge"]
    assert check.status == FAILED
    assert check.values["split_by"] == 1
    assert "change of map" in check.reason
    assert report.grade == "refused, structural", "the grade string is the ruling's; the audit is where the objection lives"
    assert report.contested


def test_a_zero_that_holds_under_alternative_maps_passes():
    answers, model = build({"empty": 0.0, "still": 0.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    alt_answers, alt = build({"empty": 1.0, "still": 1.0})
    assert audit(ruling, conditions=C, alternatives=[(alt_answers, alt)])["merge"].status == PASSED


def test_the_merge_check_reads_refusals_only():
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    alt_answers, alt = build({"a": 0.0, "b": 0.0})
    check = audit(ruling, conditions=C, alternatives=[(alt_answers, alt)])["merge"]
    assert check.status == NOT_RUN
    assert check.reason.startswith("not run: needs")
    assert "admissible" in check.reason


def test_the_merge_check_names_what_it_needs():
    answers, model = build({"empty": 0.0, "still": 0.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert audit(ruling)["merge"].status == NOT_RUN
    assert audit(ruling, conditions=C, alternatives=[])["merge"].status == NOT_RUN


def test_stability_and_merge_answer_different_questions():
    # The same zero: perturbing the LAW under the shared map cannot split it
    # (stability passes), an alternative MAP can (merge fails). That is the
    # whole of what T14 does and does not say.
    answers, model = build({"empty": 0.0, "still": 0.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)

    class Wider(Placed):
        def law(self, answer, conditions):
            return GaussianLaw(self.loci[answer.name], 3.0)

    report = audit(
        ruling, answers=answers, conditions=C,
        perturbations=[Wider({"empty": 0.0, "still": 0.0})],
        alternatives=[build({"empty": 0.0, "still": 2.0})],
    )
    assert report["stability"].status == PASSED
    assert report["merge"].status == FAILED


# ---------------------------------------------------------------------------
# the review of the first cut: scope, validation, and the verifier
# ---------------------------------------------------------------------------
def test_a_model_sourced_record_verifies_as_admissible():
    # decide() says admissible; verify() must re-derive the same word and not
    # "implies admitted" from a bound that cleared the floor on model samples.
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "model")
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.rule is Rule.ADMISSIBLE
    record = Record.build(claim="c", interval="t", ruling=ruling, certificates=certs)
    assert verify(record) == []


def test_a_stray_pair_outside_the_open_set_does_not_block_admission():
    answers, model = build({"a": 0.0, "b": 3.0, "c": 6.0})
    _, measured = samples_for(answers, model, "readings")
    wider_answers, wider = build({"a": 0.0, "b": 3.0, "c": 6.0, "d": 9.0})
    _, simulated = samples_for(wider_answers, wider, "model")
    certs = dict(measured)
    certs[("c", "d")] = simulated[("c", "d")]
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.rule is Rule.ADMITTED
    assert unmeasured_pairs(certs, answers.names) == []
    assert unmeasured_pairs(certs, ("c", "d")) == [("c", "d")]


def test_a_reversed_key_is_read_either_way():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "model")
    reversed_keys = {(r, l): c for (l, r), c in certs.items()}
    assert unmeasured_pairs(reversed_keys, answers.names) == [("a", "b")]


def test_a_number_in_place_of_a_certificate_fails_the_check_and_never_raises():
    answers, model = build({"a": 0.0, "b": 3.0})
    _, certs = samples_for(answers, model, "readings")
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.certificate is not None
    check = audit(ruling, certificates={("a", "b"): 0.9})["certificate"]
    assert check.status == FAILED
    assert "not a Certificate" in check.reason
    with pytest.raises(CertificateError):
        unmeasured_pairs({("a", "b"): 0.9}, answers.names)
