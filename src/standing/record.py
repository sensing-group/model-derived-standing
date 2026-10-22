"""The record that crosses the module boundary with the claim.

Standing is worth little if it stops at the function that computed it. The claim
travels with its entitlement, and the record is the form that entitlement takes:
plain JSON, no Python on the other side, so a consumer written in another
language can read it and a third party can check it.

What makes this more than logging is that the record is REDERIVABLE. Every number
in it is either a certificate's arithmetic, which can be recomputed from the rates
it carries, or a bracket, which can be recomputed from its measurement and its
null. `verify` does exactly that and needs neither the model nor the data. A
record that cannot be replayed into its own claim is not provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from standing.certify import Certificate
from standing.drift import Bracket, bracket
from standing.errors import InvalidInput
from standing.model import Conditions, canonical_repr
from standing.numerics import h2_bernoulli
from standing.sigma import SigmaResult

__all__ = ["Record", "Statistic", "verify", "SCHEMA"]

SCHEMA = "standing/record/6"
_TOLERANCE = 1e-9


KINDS = ("ceiling", "operational", "unknown")


@dataclass(frozen=True)
class Statistic:
    """WHICH statistic a sigma was computed over, which the value alone hides.

    Two objects travelled under one symbol until this existed. Sigma over the
    joint reading of the live sensors is a CEILING: what any pipeline on this
    hardware could achieve. Sigma over the scalar the pipeline actually
    thresholds is the OPERATIONAL value, and it is what a claim rests on. T6
    orders them, since the pooled statistic is a function of the joint one, so
    the operational value can never exceed the ceiling.

    Recording only the value left a consumer unable to tell 0.9933 from 0.2212
    for one claim, which is the difference between "this hardware cannot do it"
    and "your fusion is throwing it away". The gap is the second sentence, and
    it is actionable in a way neither number is alone.

    Attributes:
        kind: "ceiling", "operational", or "unknown" when a caller records a
            value without saying which. Unknown is honest; absent is honest
            too. Guessing is not, so nothing here infers it.
        name: the statistic's own name, as passed to `sigma`.
        ceiling: sigma over the joint reading, when the caller computed it.
            Only meaningful beside an operational value, so it is refused on a
            record that IS the ceiling.
    """

    kind: str
    name: str | None = None
    ceiling: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise InvalidInput(
                f"statistic kind must be one of {KINDS}, got {self.kind!r}. "
                f"Use 'unknown' rather than guessing: a wrong index is worse "
                f"than an absent one, because a consumer would act on it."
            )
        if self.ceiling is None:
            return
        if self.kind != "operational":
            raise InvalidInput(
                f"a ceiling belongs beside an OPERATIONAL value, and this record "
                f"is {self.kind!r}. A ceiling record is already the ceiling."
            )
        try:
            value = float(self.ceiling)
        except (TypeError, ValueError) as exc:
            raise InvalidInput(f"ceiling is not a number: {self.ceiling!r}") from exc
        if not (0.0 <= value <= 1.0) or value != value:
            raise InvalidInput(
                f"ceiling must lie in [0, 1], got {self.ceiling!r}"
            )


@dataclass(frozen=True)
class Record:
    """One claim, its standing, and everything needed to check both.

    Attributes:
        claim: what was asserted, in the consumer's own vocabulary.
        interval: the interval it covers.
        answers: the open answers it was scored against. A sigma without its
            answer set cannot be read, so this is not optional.
        floor: the admission threshold in force.
        rule: admissible, admitted, recruit, carry or abstain.
        checks: the audit, when one was run: every failure mode tested
            for this verdict with its outcome, and the ones that could not
            be tested and why. None records that nobody audited, which is
            not the same as auditing and finding nothing.
        sigma: the value and how it was obtained.
        bracket: present when conditions drifted and sigma was bracketed.
        certificates: explicit rules, one per pair, each checkable by a stranger.
        conditions: what was true about the observation. Never a reading.
    """

    claim: str
    interval: str
    answers: tuple[str, ...]
    floor: float
    rule: str
    sigma: Mapping[str, Any]
    bracket: Mapping[str, Any] | None = None
    admitted: bool | None = None
    """Whether the ruling admitted the claim. None in records written before
    recruitment was gated on certificates, where `recruit` always meant admitted."""
    certificates: Sequence[Mapping[str, Any]] = ()
    conditions: Sequence[Sequence[str]] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    quotient: Mapping[str, Any] | None = None
    chain: Mapping[str, Any] | None = None
    span: Mapping[str, Any] | None = None
    answer_map: Mapping[str, Any] | None = None
    statistic: Mapping[str, Any] | None = None
    checks: Mapping[str, Any] | None = None
    schema: str = SCHEMA

    # `quotient`, `chain`, `span` and `answer_map` default to None, and None
    # means NOT DECLARED rather than clean. That distinction is the same one the
    # four dispositions already make between carry and abstain, and it matters
    # more here than anywhere else: a claim admitted over a structural collapse
    # used to verify clean and look identical to one that had been audited and
    # passed. A consumer reading a record must be able to tell "checked and
    # fine" from "nobody checked".
    #
    # `statistic` is the newest. Sigma over the joint reading of the live
    # sensors is a CEILING and sigma over the scalar the pipeline thresholds is
    # the ENTITLEMENT, T6 orders them, and the value alone hides which one it
    # is. The gap between them is the standing the caller's own fusion
    # discarded, which is more actionable than either number by itself.
    #
    # `answer_map` covers the condition that carried the most
    # weight while leaving no trace at all: which partition the claim was scored
    # over, where that partition came from, and whether anyone ran the
    # consumer's own policy against it. Its `checked` sub-block is None on the
    # same convention, so "derived from a policy" and "derived and verified
    # against that policy" are not the same record.

    @classmethod
    def build(
        cls,
        *,
        claim: str,
        interval: str,
        ruling,
        result: SigmaResult | None = None,
        band: Bracket | None = None,
        certificates: Mapping[tuple[str, str], Certificate] | None = None,
        conditions: Mapping[str, object] | None = None,
        provenance: Mapping[str, Any] | None = None,
        audit: Any = None,
        span: Mapping[str, Any] | None = None,
        answer_set: Any = None,
        check: Any = None,
        statistic: Any = None,
        checks: Any = None,
    ) -> "Record":
        """Assemble a record from what the rules actually decided.

        Args:
            audit: a QuotientAudit, when one was run. Omitting it records that
                nothing was audited, which is NOT the same as auditing and
                finding nothing, and `verify` reads the difference.
            span: the dependency span the verdict rests on, as a mapping, when
                it was measured. A rule with memory decides over a window and
                the interval string cannot express that.
            answer_set: the AnswerSet the claim was scored against, so the
                record can say whether its partition was derived from a policy
                or written by hand, and name the map it came from.
            check: a QuotientCheck, when the answer set was run against the
                consumer's policy. Omitting it records that nobody checked,
                which is not the same as checking and agreeing.
            statistic: a Statistic saying WHICH statistic this sigma is over,
                and the ceiling beside it when one was computed. Omitting it
                records that nobody said, which a consumer can act on; guessing
                would not be.

        The chain block is derived from the ruling rather than passed: a
        recruitment's rank and the anchor it used live on the Ruling and never
        reached the record, so a chained claim and a grounded one looked alike
        to any consumer.
        """
        # Misuse is refused with a library error, not a raw AttributeError from
        # deep inside an attribute access. `audit` and `span` are new, so these
        # are new ways to get a stack trace where the house style promises a
        # sentence.
        if audit is not None:
            missing = [
                name for name in
                ("answer_sigma", "floor", "delta", "pairs_scored", "refines", "collapses")
                if not hasattr(audit, name)
            ]
            if missing:
                raise InvalidInput(
                    f"audit must be a QuotientAudit, got "
                    f"{type(audit).__name__} without {missing}. Pass the object "
                    f"quotient_audit returned, or omit it and record that nothing "
                    f"was audited."
                )
        if span is not None and not isinstance(span, Mapping):
            raise InvalidInput(
                f"span must be a mapping describing the window a verdict rests "
                f"on, got {type(span).__name__}"
            )
        if answer_set is not None and not hasattr(answer_set, "origin"):
            raise InvalidInput(
                f"answer_set must be an AnswerSet, got {type(answer_set).__name__}. "
                f"Pass the set the claim was scored against, or omit it and record "
                f"that the partition's provenance is unknown."
            )
        if statistic is not None and not isinstance(statistic, Statistic):
            raise InvalidInput(
                f"statistic must be a Statistic, got {type(statistic).__name__}. "
                f"Build one so the kind is validated, or omit it and record that "
                f"nobody said which statistic this is."
            )
        if (statistic is not None and statistic.ceiling is not None
                and str(getattr(ruling.rule, "value", ruling.rule)) == "recruit"):
            raise InvalidInput(
                f"a ceiling cannot be recorded on a RECRUIT. The ceiling bounds "
                f"the value it sits beside, and T6 only orders the two when both "
                f"are over the SAME answer set. A recruitment narrows the set, so "
                f"the claim rests on {ruling.sigma_after:.4f} over the narrowed "
                f"answers while the recorded value is {float(ruling.sigma):.4f} "
                f"over the full ones, and one ceiling cannot bound both. Record "
                f"the kind without a ceiling, or record the narrowed claim "
                f"separately."
            )
        if check is not None and not hasattr(check, "as_dict"):
            raise InvalidInput(
                f"check must be a QuotientCheck, got {type(check).__name__}. Pass "
                f"the object quotient_check returned, or omit it and record that "
                f"nobody ran the policy against the answer set."
            )
        # The set the claim was scored against must be the set it was RULED on.
        # An answer_set from a different claim would put a fingerprint in the
        # record that a third party could not act on, which is worse than
        # putting none there: it would be provenance for the wrong partition.
        if answer_set is not None:
            declared = set(getattr(answer_set, "names", ()))
            stray = sorted(set(ruling.answers) - declared)
            if declared and stray:
                raise InvalidInput(
                    f"the ruling was made over {stray}, which the answer_set does "
                    f"not hold ({sorted(declared)}). A record may only carry the "
                    f"provenance of the partition it was actually scored on. A "
                    f"recruitment is fine here, since a narrowed set is a subset "
                    f"of the one it was narrowed from."
                )

        # A record that carries a bracket must have been RULED on that bracket.
        # `verify` re-derives the disposition from the bracket's low end, because
        # that is the value a claim may rest on under drifting conditions, so a
        # ruling made on the raw sigma while a bracket rides along produces a
        # record that fails its own verifier the moment the two straddle the
        # floor. That is exactly what the worked example did: sigma 0.3262
        # admitted, bracket low 0.2334, floor 0.2500, and verify reported the
        # record as inconsistent with itself. Catch it where it is created rather
        # than where it is read.
        rule_name = str(getattr(ruling.rule, "value", ruling.rule))
        if rule_name == "admitted" and not certificates:
            raise InvalidInput(
                "this ruling is ADMITTED but no certificates were given to the "
                "record. An admission rests on a rule scored on held-out "
                "readings, and a record that does not carry that rule cannot "
                "be checked by anyone, so it cannot say admitted."
            )
        if band is not None and abs(float(ruling.sigma) - float(band.low)) > _TOLERANCE:
            raise InvalidInput(
                f"this record carries a drift bracket with low end {band.low:.4f} "
                f"but its ruling was made on {ruling.sigma:.4f}. Under drift the "
                f"claim rests on the bracket, so pass the bracket to decide() "
                f"rather than the SigmaResult, or leave the bracket off the "
                f"record. As it stands the record contradicts itself and verify() "
                f"will say so."
            )

        sigma_block: dict[str, Any] = {"value": float(ruling.sigma)}
        if result is not None:
            sigma_block.update(
                exact=bool(result.exact),
                draws=result.draws,
                noise=result.noise,
                binding_pair=list(result.binding_pair),
                per_pair={"|".join(k): v for k, v in result.per_pair.items()},
            )
        if ruling.sigma_after is not None:
            sigma_block["after_recruitment"] = float(ruling.sigma_after)
        if isinstance(conditions, Conditions):
            conditions = conditions.fields
        elif conditions is not None and not isinstance(conditions, Mapping):
            raise InvalidInput(
                f"conditions must be a Conditions object or a mapping, got "
                f"{type(conditions).__name__}"
            )
        return cls(
            claim=claim,
            interval=interval,
            answers=tuple(ruling.answers),
            floor=float(ruling.floor),
            rule=str(getattr(ruling.rule, "value", ruling.rule)),
            admitted=bool(ruling.admitted),
            sigma=sigma_block,
            bracket=None if band is None else asdict(band),
            # pairs are stored as lists, not tuples: JSON has no tuple, and a
            # record that does not round-trip to an equal object cannot be
            # compared across a boundary, which is the whole point of it.
            certificates=[
                {**asdict(c), "pair": list(c.pair)} for c in (certificates or {}).values()
            ],
            conditions=[[str(k), canonical_repr(v)] for k, v in (conditions or {}).items()],
            provenance=dict(provenance or {}),
            quotient=None if audit is None else {
                "answer_sigma": float(audit.answer_sigma),
                "floor": float(audit.floor),
                "delta": float(audit.delta),
                "pairs_scored": int(audit.pairs_scored),
                "refines": bool(audit.refines),
                "collapses": [
                    {
                        "left": c.left, "left_answer": c.left_answer,
                        "right": c.right, "right_answer": c.right_answer,
                        "value": float(c.value), "structural": bool(c.structural),
                        "mass": float(c.mass),
                    }
                    for c in audit.collapses
                ],
            },
            chain=None if ruling.rank == 0 and ruling.recruited is None else {
                "rank": int(ruling.rank),
                "recruited": ruling.recruited,
                "ruled_out": ruling.ruled_out,
            },
            span=None if span is None else dict(span),
            answer_map=_answer_map(answer_set, check),
            statistic=_statistic_block(statistic, float(ruling.sigma)),
            checks=_checks_block(checks),
        )

    def to_json(self, **kwargs: Any) -> str:
        payload = asdict(self)
        payload["answers"] = list(self.answers)
        return json.dumps(payload, sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "Record":
        payload = json.loads(text)
        if payload.get("schema") != SCHEMA:
            raise InvalidInput(
                f"unknown record schema {payload.get('schema')!r}, expected {SCHEMA!r}"
            )
        answers = payload.get("answers", ())
        if isinstance(answers, str) or not isinstance(answers, (list, tuple)):
            # A string is iterable, so tuple("ab") would silently become two
            # answers named 'a' and 'b' and the record would look well formed.
            raise InvalidInput(
                f"answers must be a list of names, got {type(answers).__name__}"
            )
        if not all(isinstance(name, str) for name in answers):
            raise InvalidInput("every answer name must be a string")
        payload["answers"] = tuple(answers)
        for name, expected in (("sigma", Mapping), ("provenance", Mapping)):
            if name in payload and not isinstance(payload[name], dict):
                raise InvalidInput(f"{name} must be an object, got {type(payload[name]).__name__}")
        if payload.get("checks") is not None and not isinstance(payload["checks"], dict):
            raise InvalidInput(f"checks must be an object or null, got {type(payload['checks']).__name__}")
        if "certificates" in payload and not isinstance(payload["certificates"], list):
            raise InvalidInput(
                f"certificates must be a list, got "
                f"{type(payload['certificates']).__name__}"
            )
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise InvalidInput(f"record holds unknown fields: {sorted(unknown)}")
        return cls(**payload)


def _checks_block(checks: Any) -> dict[str, Any] | None:
    """The audit, as its dict, or None. Anything else is refused.

    A QuotientAudit also has an as_dict, and passing one here by mistake would
    have serialised it under the wrong key and looked like an audit that ran.
    """
    if checks is None:
        return None
    if isinstance(checks, Mapping):
        block = dict(checks)
    elif hasattr(checks, "grade") and hasattr(checks, "checks") and hasattr(checks, "as_dict"):
        block = checks.as_dict()
    else:
        raise InvalidInput(
            f"checks must be an Audit from standing.audit (or its dict), got "
            f"{type(checks).__name__}. A QuotientAudit goes in `audit=`."
        )
    for key in ("rule", "grade", "checks"):
        if key not in block:
            raise InvalidInput(f"checks: an audit block carries {key!r}, this one does not")
    return block


def _statistic_block(statistic: Any, value: float) -> dict[str, Any] | None:
    """The index block, with the gap derived rather than passed.

    `gap` is not an input. It is ceiling minus value, computed here and
    re-derived by `verify`, on the same principle as every other number in a
    record: a consumer recomputes it rather than trusting it.
    """
    if statistic is None:
        return None
    block: dict[str, Any] = {
        "kind": statistic.kind,
        "name": statistic.name,
        "ceiling": None if statistic.ceiling is None else float(statistic.ceiling),
    }
    block["gap"] = (
        None if statistic.ceiling is None else float(statistic.ceiling) - value
    )
    return block


def _answer_map(answer_set: Any, check: Any) -> dict[str, Any] | None:
    """The provenance block for the partition, or None when nothing is known.

    `kind` is "unknown" only when a check was supplied without the set it was
    run on. That combination is legal, because a caller may check a partition it
    did not build, but it is a weaker record and it should read as one.
    """
    if answer_set is None and check is None:
        return None
    origin = getattr(answer_set, "origin", None) if answer_set is not None else None
    block: dict[str, Any] = (
        dict(origin.as_dict()) if origin is not None
        else {"kind": "unknown", "states": None, "action": None}
    )
    block["checked"] = None if check is None else check.as_dict()
    return block


def verify(record: Record) -> list[str]:
    """Re-derive every number in a record, using neither the model nor the data.

    This is what a third party runs. It recomputes each certificate from the
    crossing rates it carries, recomputes the bracket from its measurement and
    its null, and checks that the disposition follows from the value and the
    floor.

    Returns:
        A list of discrepancies, empty when the record replays into itself.
    """
    problems: list[str] = []

    if len(record.answers) < 2:
        problems.append(
            f"answers: {len(record.answers)} listed; standing needs at least two, "
            f"and a set of one means nothing was asked"
        )

    if not isinstance(record.certificates, (list, tuple)):
        problems.append(
            f"certificates: expected a list, found {type(record.certificates).__name__}"
        )
    for entry in record.certificates if isinstance(record.certificates, (list, tuple)) else ():
        if not isinstance(entry, Mapping):
            problems.append(f"certificate: expected an object, found {type(entry).__name__}")
            continue
        pair = tuple(entry.get("pair", ("?", "?")))
        try:
            recomputed = h2_bernoulli(entry["rate_left"], entry["rate_right"])
        except (KeyError, TypeError, ValueError, InvalidInput) as exc:
            problems.append(f"certificate {pair}: rates unusable ({exc})")
            continue
        try:
            reported = float(entry["value"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"certificate {pair}: no usable value")
            continue
        if abs(recomputed - reported) > 1e-6:
            problems.append(
                f"certificate {pair}: reported {reported:.6f} but its own "
                f"rates give {recomputed:.6f}"
            )
        try:
            sizes_ok = int(entry.get("n_select", 0)) > 0 and int(entry.get("n_score", 0)) > 0
        except (TypeError, ValueError):
            sizes_ok = False
        if not sizes_ok:
            problems.append(
                f"certificate {pair}: missing selection or scoring size, so it "
                f"cannot be shown to be a certificate rather than a fit"
            )
        source = entry.get("source")
        if source not in ("readings", "model"):
            problems.append(
                f"certificate {pair}: source {source!r} does not say whether it "
                f"was scored on readings or on the model's own samples"
            )
        elif source == "model" and record.rule == "admitted":
            problems.append(
                f"certificate {pair}: scored on the model's own samples, and the "
                f"record says admitted. A simulated certificate inherits the "
                f"model's errors and cannot admit"
            )

    if record.bracket is not None:
        try:
            recomputed = bracket(record.bracket["measured"], record.bracket["null"])
        except (KeyError, TypeError, InvalidInput) as exc:
            problems.append(f"bracket: unusable ({exc})")
            recomputed = None
    if record.bracket is not None and recomputed is not None:
        for edge in ("low", "high"):
            if edge not in record.bracket:
                problems.append(f"bracket: missing {edge}")
                continue
            # Guarded for the same reason the sigma block is: a verifier is
            # handed records it did not write, and one whose bracket edge was
            # the string 'x' took the check down instead of reporting it.
            try:
                reported_edge = float(record.bracket[edge])
            except (TypeError, ValueError):
                problems.append(
                    f"bracket {edge}: {record.bracket[edge]!r} is not a number"
                )
                continue
            if abs(getattr(recomputed, edge) - reported_edge) > 1e-6:
                problems.append(
                    f"bracket {edge}: reported {reported_edge:.6f}, "
                    f"recomputed {getattr(recomputed, edge):.6f}"
                )

    # A verifier is the adversarial entry point: it is handed records it did not
    # write, including malformed ones. It must REPORT on them, not raise. An
    # earlier version indexed sigma["value"] directly and a record with an empty
    # sigma block crashed the check and the CLI with it.
    try:
        value = float(record.sigma["value"])
    except (KeyError, TypeError, ValueError):
        problems.append(
            "sigma: the record carries no usable value, so nothing about its "
            "disposition can be checked"
        )
        return problems
    try:
        floor = float(record.floor)
    except (TypeError, ValueError):
        problems.append("floor: not a number, so admission cannot be checked")
        return problems
    expected = _expected_rule(value, floor, record)
    if expected is not None and expected != record.rule:
        problems.append(
            f"rule: recorded {record.rule!r} but sigma {value:.4f} against floor "
            f"{floor:.4f} implies {expected!r}"
        )

    # The quotient block, when present, must be about THIS claim and must not
    # be contradicted by the disposition. When absent, say nothing: absence is
    # "not audited", which is a fact about the record rather than a defect in it.
    if record.quotient is not None:
        if not isinstance(record.quotient, Mapping):
            problems.append(
                f"quotient: expected an object, found {type(record.quotient).__name__}"
            )
        else:
            collapses = record.quotient.get("collapses")
            if not isinstance(collapses, (list, tuple)):
                problems.append("quotient: collapses must be a list")
                collapses = ()
            try:
                audited_floor = float(record.quotient["floor"])
            except (KeyError, TypeError, ValueError):
                problems.append("quotient: no usable floor, so it cannot be tied to this claim")
                audited_floor = None
            if audited_floor is not None and abs(audited_floor - floor) > _TOLERANCE:
                problems.append(
                    f"quotient: audited against floor {audited_floor:.4f} but the "
                    f"claim faced {floor:.4f}, so the audit is about a different "
                    f"question"
                )
            if collapses and record.rule in ("admissible", "admitted", "recruit"):
                worst = collapses[0] if isinstance(collapses[0], Mapping) else {}
                kind = "structurally " if worst.get("structural") else ""
                problems.append(
                    f"rule: {record.rule!r} over {len(collapses)} reported "
                    f"collapse(s); {worst.get('left')!r} and {worst.get('right')!r} "
                    f"are {kind}indistinguishable, so verdicts resting on that "
                    f"pair were not entitled"
                )

    # The partition block. Its `checked` half re-derives like everything else
    # here: `agrees` is not a claim a record gets to make on its own, it is a
    # function of the disagreements listed beside it, so a record asserting
    # agreement over its own reported splits is caught without the policy, the
    # states or the model being available to the reader.
    if record.answer_map is not None:
        if not isinstance(record.answer_map, Mapping):
            problems.append(
                f"answer_map: expected an object, found "
                f"{type(record.answer_map).__name__}"
            )
        else:
            kind = record.answer_map.get("kind")
            if kind not in ("derived", "declared", "unknown"):
                problems.append(
                    f"answer_map: kind {kind!r} is not one of 'derived', "
                    f"'declared' or 'unknown'"
                )
            checked = record.answer_map.get("checked")
            if checked is not None and not isinstance(checked, Mapping):
                problems.append(
                    f"answer_map: checked must be an object or absent, found "
                    f"{type(checked).__name__}"
                )
            elif isinstance(checked, Mapping):
                found = {
                    name: checked.get(name)
                    for name in ("splits", "merges", "unplaced", "ambiguous")
                }
                counts = {
                    name: len(value) if isinstance(value, (list, tuple)) else None
                    for name, value in found.items()
                }
                if any(count is None for count in counts.values()):
                    problems.append(
                        "answer_map: splits, merges, unplaced and ambiguous must "
                        "each be a list, so agreement cannot be re-derived"
                    )
                else:
                    clean = not any(counts.values())
                    if bool(checked.get("agrees")) != clean:
                        problems.append(
                            f"answer_map: reports agrees="
                            f"{bool(checked.get('agrees'))} but lists "
                            f"{counts['splits']} split(s), {counts['merges']} "
                            f"merge(s), {counts['unplaced']} unplaced and "
                            f"{counts['ambiguous']} ambiguous state(s), so it "
                            f"contradicts itself"
                        )
                    elif not clean and record.rule in ("admissible", "admitted", "recruit"):
                        # Both directions, deliberately. Coarser than the policy,
                        # sigma is scored on a pair the consumer separates;
                        # finer, it is a minimum over pairs nobody asked about
                        # while the consumer faces the mixture. Neither is the
                        # conservative side.
                        problems.append(
                            f"rule: {record.rule!r} on an answer set that is not "
                            f"the consumer's own partition "
                            f"({counts['splits']} split(s), {counts['merges']} "
                            f"merge(s), {counts['unplaced']} unplaced); sigma "
                            f"answers a different question from the one the "
                            f"deployment asks"
                        )
                try:
                    width = int(checked.get("states_checked"))
                except (TypeError, ValueError):
                    width = None
                if width is None or width < 2:
                    problems.append(
                        "answer_map: a check reports fewer than two states, but a "
                        "check with no pair in it cannot disagree"
                    )

    # The statistic index. Its ordering is a THEOREM a stranger can check: the
    # operational value is over a function of the joint reading, so T6 forbids
    # it exceeding the ceiling. A record claiming otherwise contradicts the data
    # processing inequality, and it can be caught with no model and no data.
    if record.statistic is not None:
        if not isinstance(record.statistic, Mapping):
            problems.append(
                f"statistic: expected an object, found {type(record.statistic).__name__}"
            )
        else:
            kind = record.statistic.get("kind")
            if kind not in KINDS:
                problems.append(
                    f"statistic: kind {kind!r} is not one of {KINDS}"
                )
            ceiling = record.statistic.get("ceiling")
            if ceiling is not None:
                try:
                    ceiling = float(ceiling)
                except (TypeError, ValueError):
                    problems.append(f"statistic: ceiling {ceiling!r} is not a number")
                    ceiling = None
            if ceiling is not None and record.rule == "recruit":
                problems.append(
                    "statistic: a ceiling is recorded on a recruit. T6 orders a "
                    "ceiling against a value only when both are over the same "
                    "answer set, and a recruitment narrows the set, so this "
                    "ceiling bounds neither the recorded value nor the narrowed "
                    "one a reader would act on"
                )
            if ceiling is not None and kind != "operational":
                problems.append(
                    f"statistic: a ceiling is recorded beside a {kind!r} value; a "
                    f"ceiling belongs beside an operational one"
                )
            if ceiling is not None:
                if ceiling + _TOLERANCE < value:
                    problems.append(
                        f"statistic: the operational value {value:.4f} exceeds its "
                        f"own ceiling {ceiling:.4f}. T6 forbids this: the pooled "
                        f"statistic is a function of the joint reading, so "
                        f"processing cannot create standing"
                    )
                reported = record.statistic.get("gap")
                try:
                    reported = float(reported)
                except (TypeError, ValueError):
                    problems.append("statistic: no usable gap beside a ceiling")
                    reported = None
                if reported is not None and abs((ceiling - value) - reported) > 1e-6:
                    problems.append(
                        f"statistic: gap reported {reported:.6f} but ceiling minus "
                        f"value gives {ceiling - value:.6f}"
                    )
            elif "gap" in record.statistic and record.statistic["gap"] is not None:
                problems.append(
                    "statistic: a gap is recorded with no ceiling to derive it from"
                )

    # A recruitment is a derived claim and must say so.
    if record.rule == "recruit" and record.chain is None:
        problems.append(
            "chain: the rule is 'recruit' but no chain is recorded, so a derived "
            "claim is indistinguishable from a grounded one"
        )
    if isinstance(record.chain, Mapping):
        try:
            rank = int(record.chain["rank"])
        except (KeyError, TypeError, ValueError):
            problems.append("chain: no usable rank")
            rank = None
        if rank is not None and record.rule == "recruit" and rank < 1:
            problems.append(
                f"chain: rule is 'recruit' but rank is {rank}; a recruitment is at "
                f"least one deeper than the anchor it used"
            )

    if isinstance(record.checks, Mapping):
        audited_rule = record.checks.get("rule")
        if audited_rule is not None and str(audited_rule) != record.rule:
            problems.append(
                f"checks: the audit was run on a {audited_rule!r} ruling but the record says "
                f"{record.rule!r}, so it audits a different verdict"
            )
        entries = record.checks.get("checks")
        if not isinstance(entries, (list, tuple)):
            problems.append("checks: the audit carries no list of checks")
        else:
            for entry in entries:
                status = entry.get("status") if isinstance(entry, Mapping) else None
                if status not in ("passed", "failed", "not_run"):
                    problems.append(f"checks: a check carries status {status!r}, which is not a status")
                    break
    if record.rule == "abstain" and value != 0.0:
        problems.append(
            f"rule: abstain is reserved for sigma exactly zero, which is a "
            f"theorem; this record abstains at {value:.6f}, which is a policy "
            f"and must be recorded as carry"
        )
    return problems


def _expected_rule(value: float, floor: float, record: Record) -> str | None:
    """What the disposition should have been, or None when it cannot be told.

    Every conversion here is guarded. This runs inside `verify`, which is handed
    records it did not write, and returning None on an unreadable field lets the
    caller report a discrepancy instead of dying on one: a bracket whose low end
    was the string 'x' took the whole verifier down, and the CLI with it, which
    is precisely the failure the module docstring says was already fixed once.
    """
    if not isinstance(record.sigma, Mapping):
        return None
    if record.rule == "recruit":
        after = record.sigma.get("after_recruitment")
        if after is None:
            return None
        try:
            return "recruit" if float(after) >= floor else "carry"
        except (TypeError, ValueError):
            return None
    if isinstance(record.bracket, Mapping) and "low" in record.bracket:
        try:
            value = float(record.bracket["low"])
        except (TypeError, ValueError):
            return None
    if value >= floor:
        bound = _certified_bound(record)
        # decide() admits only on certificates scored on readings. A covering
        # set with one model-sourced pair leaves the claim admissible, whatever
        # the bound, and the verifier has to re-derive the same answer.
        return "admitted" if bound is not None and bound >= floor and _all_measured(record) else "admissible"
    return "carry" if value > 0.0 else "abstain"


def _all_measured(record: Record) -> bool:
    """Whether every certificate covering the record's answers was scored on readings."""
    if not isinstance(record.certificates, (list, tuple)) or not record.answers:
        return False
    sources: dict[frozenset, object] = {}
    for entry in record.certificates:
        if not isinstance(entry, Mapping):
            continue
        try:
            sources[frozenset(str(x) for x in entry.get("pair"))] = entry.get("source")
        except TypeError:
            continue
    names = [str(n) for n in record.answers]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sources.get(frozenset((names[i], names[j]))) != "readings":
                return False
    return True


def _certified_bound(record: Record) -> float | None:
    """The minimum certificate value over every pair of the record's answers.

    None when any pair has no readable certificate. A record can only claim
    ADMITTED when the certificates it carries cover the set, and this is what
    `verify` re-derives the disposition from, so a record that says "admitted"
    over a missing pair is reported as inconsistent with itself.
    """
    if not isinstance(record.certificates, (list, tuple)) or not record.answers:
        return None
    values: dict[frozenset, float] = {}
    for entry in record.certificates:
        if not isinstance(entry, Mapping):
            continue
        pair = entry.get("pair")
        try:
            key = frozenset(str(x) for x in pair)
            values[key] = float(entry["value"])
        except (KeyError, TypeError, ValueError):
            continue
    bounds = []
    names = list(record.answers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = frozenset((str(names[i]), str(names[j])))
            if key not in values:
                return None
            bounds.append(values[key])
    return min(bounds) if bounds else None
