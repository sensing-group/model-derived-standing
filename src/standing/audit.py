"""The audit: every way a verdict can be wrong, as a check on the ruling.

A ruling is a number against a floor. An audit is the list of failure modes
that were tested for that ruling, with the outcome of each, and the list that
could not be tested and why. It is the provenance a consumer can act on: not
"how sigma was derived" in the abstract, but which of the ways sigma could be
wrong were excluded, by what, on what evidence.

The procedure a ruling comes from has five steps, and each step fails in its
own way (verification-results v7, "how can we know when each step is wrong"):

    1  the answers    do not match the actions, or leave a covered state out
    2  the conditions are wrong, or the regime changed under them
    3  the law        puts the answers in the wrong place, or omits states
    4  the comparison is over a statistic the decision does not use
    5  the verdict    is taken as licence, or refuses more than it need

Every check here guards one of those. Every check returns a `Check` with a
status in {passed, failed, not_run} and never raises for a missing input: a
check that cannot run says what it needed, because the ABSENCE of a check is
information a consumer must see, and it is not the same as a check that passed.

Two facts organise the grade at the bottom. Errors of omission in steps 1 and
3 can only inflate sigma, so a refusal at exactly zero survives every
correction of the model and needs no audit to be trusted. And every check is
the same check: what the description said the sensor could do, against
something it did do, a reference period, a probe, a held-out reading. What
none of them catches is a condition instrument that is wrong and consistent
with the reference. That is the one silent failure this module leaves, and it
is said here rather than left in a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from standing.answers import AnswerSet
from standing.certify import Certificate
from standing.checks import null_calibration, quotient_check, seed_stability
from standing.divergence import hellinger2_samples
from standing.errors import CertificateError, InvalidInput, StandingError
from standing.model import Conditions, ForwardModel, Law
from standing.rules import Rule, Ruling, certificate_bound, unmeasured_pairs
from standing.sigma import SigmaResult, sigma

__all__ = ["PASSED", "FAILED", "NOT_RUN", "Check", "Audit", "Probe", "Coverage", "audit", "coverage_report"]

PASSED = "passed"
FAILED = "failed"
NOT_RUN = "not_run"
_STATUSES = (PASSED, FAILED, NOT_RUN)


@dataclass(frozen=True)
class Check:
    """One failure mode, tested or not.

    Attributes:
        name: which check, stable across versions so records can be compared.
        step: the step of the procedure it guards, 1 to 5.
        status: passed, failed or not_run.
        reason: one sentence a reviewer can read. For not_run, what was needed.
        values: the numbers the status rests on, so a third party can argue.
    """

    name: str
    step: int
    status: str
    reason: str
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise InvalidInput(f"check {self.name!r}: status must be one of {_STATUSES}, got {self.status!r}")
        if self.step not in (1, 2, 3, 4, 5):
            raise InvalidInput(f"check {self.name!r}: step must be 1 to 5, got {self.step!r}")

    @property
    def ran(self) -> bool:
        return self.status != NOT_RUN

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step": self.step,
            "status": self.status,
            "reason": self.reason,
            "values": {k: _plain(v) for k, v in self.values.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Check":
        return cls(
            name=str(payload["name"]),
            step=int(payload["step"]),
            status=str(payload["status"]),
            reason=str(payload.get("reason", "")),
            values=dict(payload.get("values", {}) or {}),
        )


def _plain(value: Any) -> Any:
    """JSON-safe copies of the numbers a check reports."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Audit:
    """The checks run on one ruling, and the grade they add up to.

    Attributes:
        rule: the disposition audited, as its string.
        checks: every check the audit knows, run or not.
        grade: one line. `refused, structural` for a zero, which needs no
            audit to be trusted. `admissible, unverified` for a floor cleared
            on the model alone. `admitted (k of n checks run, all passed)`
            when the certificate is present. `contested` appended to either
            of the last two when a check that ran failed. `carried, below
            the floor` for CARRY.
    """

    rule: str
    checks: tuple[Check, ...]
    grade: str

    def __iter__(self):
        return iter(self.checks)

    def __getitem__(self, name: str) -> Check:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    @property
    def ran(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.ran)

    @property
    def passed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == PASSED)

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == FAILED)

    @property
    def not_run(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == NOT_RUN)

    @property
    def contested(self) -> bool:
        return bool(self.failed)

    def as_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "grade": self.grade, "checks": [c.as_dict() for c in self.checks]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Audit":
        checks = tuple(Check.from_dict(c) for c in payload.get("checks", ()))
        return cls(rule=str(payload.get("rule", "")), checks=checks, grade=str(payload.get("grade", "")))


def grade_of(rule: Rule | str, checks: Sequence[Check], admitted: bool | None = None) -> str:
    """The one-line reading of a ruling and its checks.

    `admitted` is the ruling's flag. A RECRUIT ruling is admitted only when a
    certificate cleared the floor on the narrowed set, so without the flag a
    recruit reads as admitted the way it did before that gate existed.
    """
    name = str(getattr(rule, "value", rule))
    if name == Rule.RECRUIT.value and admitted is False:
        return f"admissible after recruitment, unverified{", contested" if [c for c in checks if c.status == FAILED] else ""}"
    ran = [c for c in checks if c.ran]
    failed = [c for c in checks if c.status == FAILED]
    suffix = ", contested" if failed else ""
    if name == Rule.ABSTAIN.value:
        return "refused, structural"
    if name in (Rule.ADMITTED.value, Rule.RECRUIT.value):
        if failed:
            return f"admitted ({len(ran)} of {len(checks)} checks run, {len(failed)} failed), contested"
        return f"admitted ({len(ran)} of {len(checks)} checks run, all passed)"
    if name == Rule.ADMISSIBLE.value:
        return f"admissible, unverified{suffix}"
    if name == Rule.CARRY.value:
        return f"carried, below the floor{suffix}"
    return f"{name}{suffix}"


# ---------------------------------------------------------------------------
# the probe: the only test of the law itself
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Probe:
    """An action whose outcome the law predicts, and what the outcome was.

    A forward model is a prediction, and predictions are checked the only way
    predictions can be: do something whose result the model commits to, and
    look. Move a known scatterer to a cell, switch a light, open the window,
    excite a surveyed point. A model with no probe is an assumption and is
    recorded as one (the check is not_run). A model whose probe came out as
    predicted has been tested, once, at that state and those conditions.

    Attributes:
        name: what the probe is called in the record.
        action: what was done, in words a reviewer can picture.
        predicted: the law of the decision statistic the model predicts for
            the state the action creates, under the interval's conditions.
            Anything with `.sample(size, rng)`.
        observed: the statistic actually recorded after the action, as a
            one-dimensional sample. Readings, and the only place readings
            enter an audit: a probe is a reading of the state the action
            MADE, not of the claim.
    """

    name: str
    action: str
    predicted: Law
    observed: Any

    def __post_init__(self) -> None:
        if not hasattr(self.predicted, "sample"):
            raise InvalidInput(
                f"probe {self.name!r}: predicted must be a law with .sample(size, rng), "
                f"got {type(self.predicted).__name__}"
            )
        values = np.asarray(self.observed, dtype=float).ravel()
        if values.size < 8 or not np.all(np.isfinite(values)):
            raise InvalidInput(
                f"probe {self.name!r}: observed must be at least 8 finite readings of the "
                f"statistic, got {values.size}"
            )
        object.__setattr__(self, "observed", values)


def check_probe(
    probes: Sequence[Probe] | None, *, tolerance: float, draws: int, rng: np.random.Generator | int | None
) -> Check:
    """Step 3. The law against what happened when someone acted on it.

    For each probe, the squared Hellinger between the predicted law and the
    observed outcome is set beside the estimator's own floor for a sample of
    that size (the predicted law against a fresh draw of itself). The probe
    passes when the observed outcome sits within `tolerance` of that floor.
    The tolerance is the caller's to set and to defend, like the floor: it
    says how far a prediction may miss before the law is called wrong.
    """
    if not probes:
        return _not_run("probe", 3, "at least one Probe: an action the law predicts the outcome of, and the outcome")
    tolerance = float(tolerance)
    if not (0.0 <= tolerance < 1.0):
        raise InvalidInput(f"probe tolerance must be in [0, 1), got {tolerance}")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    rows = []
    failed = []
    for probe in probes:
        n = int(probe.observed.size)
        predicted = np.asarray(probe.predicted.sample(max(n, draws), generator), dtype=float).ravel()
        again = np.asarray(probe.predicted.sample(n, generator), dtype=float).ravel()
        measured = float(hellinger2_samples(predicted, probe.observed))
        floor = float(hellinger2_samples(predicted, again))
        ok = measured <= floor + tolerance
        rows.append({"name": probe.name, "action": probe.action, "n_observed": n,
                     "h2_observed": measured, "h2_floor": floor, "passed": ok})
        if not ok:
            failed.append(f"{probe.name}: predicted against observed {measured:.4f}, estimator floor {floor:.4f}")
    values = {"probes": rows, "tolerance": tolerance}
    if failed:
        return Check(name="probe", step=3, status=FAILED, values=values,
                     reason=f"{len(failed)} of {len(rows)} probe(s) came out unlike the law predicted: " + "; ".join(failed))
    return Check(name="probe", step=3, status=PASSED, values=values,
                 reason=f"{len(rows)} probe(s) came out as the law predicted, within {tolerance:.3f} of the estimator floor")


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _not_run(name: str, step: int, needed: str) -> Check:
    return Check(name=name, step=step, status=NOT_RUN, reason=f"not run: needs {needed}")


def _base_sigma(ruling: Ruling, result: SigmaResult | None) -> float:
    """The sigma a check compares against.

    A ruling made on a drift Bracket carries the bracket's LOW end as its
    sigma, which is |H_meas - H_null| and not the scored value. Comparing a
    freshly scored sigma against it would flag every refinement as "not a
    refinement" and every reference as explaining nothing. When the caller
    supplies the SigmaResult the ruling was scored from, that value is used.
    """
    if result is not None:
        return float(result.value)
    return float(ruling.sigma)


def _is_zero(result: SigmaResult) -> bool:
    """Whether a scored sigma is zero, allowing a sampled estimate its bias.

    A structural zero from closed forms is exactly 0.0. A sampled estimate of
    a structural zero is a small positive number, the estimator's bias, of the
    order of the reported noise. Reading that as "not zero" would flag every
    sampled perturbation of a structural zero as a flip.
    """
    if result.value == 0.0:
        return True
    if not result.exact and result.noise is not None:
        return float(result.value) <= float(result.noise)
    return False


def vocabulary_of(model: ForwardModel | None, states: Sequence[object] | None) -> tuple[Sequence[object] | None, str]:
    """The states an audit ranges over, and where they came from.

    The model's own `states()` wins when it exists: a vocabulary the model
    declares is the domain of its law, and a list the caller typed is the one
    place a sigma can be inflated without anything inside noticing.
    """
    if model is not None and hasattr(model, "states") and callable(getattr(model, "states")):
        declared = tuple(model.states())
        if states is not None and set(map(str, states)) != set(map(str, declared)):
            raise InvalidInput(
                f"the model declares {len(declared)} states and the caller supplied "
                f"{len(states)} that differ from them. The model's vocabulary is the "
                f"domain of its law and it is not the caller's to edit: drop `states`, "
                f"or fix the model."
            )
        return declared, "model"
    return (None if states is None else tuple(states)), "caller"


def check_vocabulary(
    ruling: Ruling,
    fine: tuple[AnswerSet, ForwardModel] | None,
    conditions: Conditions | None,
    *,
    result: SigmaResult | None = None,
    tolerance: float,
    draws: int,
    rng: np.random.Generator | int | None,
) -> Check:
    """Step 3. Score the same claim under a finer vocabulary and see what moves.

    Adding states to a vocabulary adds candidates to a minimum, so sigma can
    only fall. Two things this catches. A finer model whose sigma is HIGHER is
    not a refinement at all, its laws moved on the states the two share, and it
    is reported as such. And an admission the finer model revokes, sigma at or
    above the floor under the coarse vocabulary and below it under the fine
    one, was manufactured by the state the coarse model forgot: the still
    occupant, the person behind the sofa. That is the one-sided error made
    visible, and it is why a positive sigma admits nothing on its own.
    """
    if fine is None or conditions is None:
        return _not_run("vocabulary", 3, "a finer model with its answer set as `fine=(answers, model)`, and the conditions")
    fine_answers, fine_model = fine
    if len(fine_answers.names) < len(ruling.answers):
        return Check(name="vocabulary", step=3, status=FAILED,
                     values={"answers_coarse": len(ruling.answers), "answers_fine": len(fine_answers.names)},
                     reason=(f"the model offered as finer has {len(fine_answers.names)} answers against "
                             f"{len(ruling.answers)}: it is coarser, and a coarser vocabulary can only inflate"))
    fine_value = float(sigma(fine_answers, fine_model, conditions, draws=draws, rng=rng).value)
    coarse_value = _base_sigma(ruling, result)
    values = {"sigma_coarse": coarse_value, "sigma_fine": fine_value, "floor": ruling.floor,
              "answers_coarse": len(ruling.answers), "answers_fine": len(fine_answers.names)}
    if fine_value > coarse_value + tolerance:
        return Check(name="vocabulary", step=3, status=FAILED, values=values,
                     reason=(f"the finer vocabulary scores {fine_value:.4f} against {coarse_value:.4f}: adding states "
                             f"cannot raise a minimum, so this is not a refinement, its laws moved on the shared states"))
    if coarse_value >= ruling.floor > fine_value:
        return Check(name="vocabulary", step=3, status=FAILED, values=values,
                     reason=(f"the finer vocabulary revokes the admission: {coarse_value:.4f} clears the floor "
                             f"{ruling.floor:.4f} and {fine_value:.4f} does not. The coarse model forgot a state "
                             f"that the answers cannot tell apart, and the admission rested on that omission"))
    return Check(name="vocabulary", step=3, status=PASSED, values=values,
                 reason=(f"the finer vocabulary scores {fine_value:.4f} against {coarse_value:.4f}, "
                         f"same side of the floor {ruling.floor:.4f}"))


def check_scope(
    ruling: Ruling, misses: Sequence[Mapping[str, Any]] | None, vocabulary: Sequence[object] | None
) -> Check:
    """Step 1. The states the claim covers, audited on the misses.

    A miss is an interval where the world held a state the claim covers and the
    verdict said otherwise. Each is given as a mapping with "state" and, when
    known, "sigma", the standing the view was given on that interval. Two
    things are read: a miss whose state is not in the vocabulary at all, which
    is a state the model never described and the claim silently covered; and
    the share of misses on intervals where sigma cleared the floor, which is
    the share the model claimed standing for and did not have.
    """
    if misses is None:
        return _not_run("scope", 1, "the misses as mappings with a state (and sigma when known), against the vocabulary")
    if vocabulary is None:
        return _not_run("scope", 1, "the vocabulary: a model with states(), or states= from the caller")
    known = {str(s) for s in vocabulary}
    unknown: list[str] = []
    above: int = 0
    scored: int = 0
    for index, miss in enumerate(misses):
        if not isinstance(miss, Mapping) or miss.get("state") is None:
            raise InvalidInput(
                f"miss {index} carries no state. A miss is an interval where the world held a "
                f"state the claim covers, so it has one, and a miss without one cannot be "
                f"audited against any vocabulary."
            )
        state = str(miss["state"])
        if state not in known:
            unknown.append(state)
        value = miss.get("sigma")
        if value is not None:
            scored += 1
            if float(value) >= ruling.floor:
                above += 1
    values = {"misses": len(misses), "unknown_states": sorted(set(unknown)),
              "misses_with_sigma": scored, "misses_above_floor": above,
              "share_above_floor": (above / scored) if scored else None}
    if unknown:
        return Check(name="scope", step=1, status=FAILED, values=values,
                     reason=(f"{len(unknown)} miss(es) fell on states the model never described: "
                             f"{', '.join(sorted(set(unknown))[:4])}. The claim covers them and the vocabulary does not"))
    # Misses on intervals where sigma cleared the floor are reported and not judged here: a
    # view with standing still misses at the rate its separation allows, and whether that
    # rate is the predicted one is the certificate's question (step 5), not the scope's.
    return Check(name="scope", step=1, status=PASSED, values=values,
                 reason=(f"{len(misses)} miss(es), all on described states"
                         + (f", {above} of {scored} on intervals where sigma cleared the floor" if scored else "")))


def check_question(
    answers: AnswerSet | None, views: Sequence[ForwardModel] | None, conditions: Conditions | None,
    *, draws: int, rng: np.random.Generator | int | None,
) -> Check:
    """Step 1. A question nobody can ever answer is malformed, not hard.

    If sigma is exactly zero for every view of the question under the same
    conditions, the answers coincide on every law there is: either two answers
    touch with no gap between them (T9), or the question asks for a distinction
    no view was built to see. The verdict is then about the question, and no
    sensor should be blamed for it.
    """
    if answers is None or views is None or conditions is None:
        return _not_run("question", 1, "the answer set, every view of the question as `views`, and the conditions")
    if len(views) < 2:
        return _not_run("question", 1, "at least two views: one view at zero is a refusal, not a malformed question")
    values_by_view = [float(sigma(answers, view, conditions, draws=draws, rng=rng).value) for view in views]
    values = {"views": len(views), "sigmas": values_by_view}
    if all(v == 0.0 for v in values_by_view):
        return Check(name="question", step=1, status=FAILED, values=values,
                     reason=(f"sigma is exactly zero for all {len(views)} views: no law anywhere separates these "
                             f"answers, so the answers touch or the question asks for a distinction no view describes"))
    return Check(name="question", step=1, status=PASSED, values=values,
                 reason=f"{sum(v > 0.0 for v in values_by_view)} of {len(views)} views separate the answers")


def check_quotient(
    answers: AnswerSet | None, policy: Callable[[object], object] | None, states: Sequence[object] | None
) -> Check:
    """Step 1. The answers must be the partition the consumer acts on (T13)."""
    if answers is None or policy is None or states is None:
        return _not_run("quotient", 1, "the answer set, the consumer's policy and the states to run it on")
    report = quotient_check(answers, policy, states)
    return Check(
        name="quotient",
        step=1,
        status=PASSED if report.agrees else FAILED,
        reason=report.reason(),
        values={
            "states_checked": report.states_checked,
            "splits": len(report.splits),
            "merges": len(report.merges),
            "unplaced": len(report.unplaced),
            "ambiguous": len(report.ambiguous),
        },
    )


def check_reference(
    ruling: Ruling,
    model: ForwardModel | None,
    answers: AnswerSet | None,
    conditions: Conditions | None,
    against: Conditions | None,
    *,
    result: SigmaResult | None = None,
    draws: int,
    rng: np.random.Generator | int | None,
) -> Check:
    """Step 2. The same answer against itself across the gap in conditions.

    If the conditions alone move the readings by as much as the two answers
    do, the separation the ruling rests on is explained by the regime and not
    by the answer. This is the label-free form of the inversion diagnostic that
    excluded three ghost-regime sessions from the mesh runs, and it is the
    check that catches a condition instrument that moved.
    """
    if model is None or answers is None or conditions is None or against is None:
        return _not_run("reference", 2, "the model, the answer set, and two sets of conditions (the interval's and a reference)")
    base = _base_sigma(ruling, result)
    if base <= 0.0:
        return Check(name="reference", step=2, status=NOT_RUN,
                     reason="not run: sigma is exactly zero, and a refusal needs no reference to be trusted")
    if against.key() == conditions.key():
        return Check(name="reference", step=2, status=NOT_RUN,
                     reason=("not run: the reference conditions are the interval's own, so there is no gap "
                             "to price. Pass the conditions of a reference period as `against`"))
    calibration = null_calibration(model, answers, conditions, against=against, draws=draws, rng=rng)
    band = calibration.budget_for(max(base, calibration.value))
    values = {"null": calibration.value, "sigma": base, "bracket_low": band.low, "bracket_high": band.high, "answer": calibration.answer}
    if calibration.value >= base:
        return Check(
            name="reference", step=2, status=FAILED, values=values,
            reason=(f"the conditions alone explain the separation: the same answer scores "
                    f"{calibration.value:.4f} against itself across the gap, and sigma is {base:.4f}"),
        )
    return Check(
        name="reference", step=2, status=PASSED, values=values,
        reason=(f"same-answer null {calibration.value:.4f} across the gap, sigma {base:.4f}, "
                f"bracket [{band.low:.4f}, {band.high:.4f}]"),
    )


def check_certificate(
    ruling: Ruling, certificates: Mapping[tuple[str, str], Certificate] | None
) -> Check:
    """Step 5. Admission rests on a rule scored on held-out READINGS (T10).

    A certificate scored on the model's own samples fails this check by name:
    it inherits the model's misspecification, so it is the model checking
    itself. When only the ruling is available, the same case is visible from
    the outside: a certificate that clears the floor on a ruling that is not
    admitted can only have been simulated, since a measured one would have
    admitted.
    """
    bound: float | None = ruling.certificate
    if certificates is not None:
        try:
            # Validate the mapping every time, even when the ruling already
            # carries a bound: a value typed in place of a Certificate must be
            # refused here, as a failed check, and never escape as an exception.
            checked = certificate_bound(certificates, ruling.answers)
            if bound is None:
                bound = checked
            simulated = unmeasured_pairs(certificates, ruling.answers)
        except (CertificateError, StandingError, InvalidInput) as exc:
            return Check(name="certificate", step=5, status=FAILED, reason=str(exc))
        open_pairs = len(ruling.answers) * (len(ruling.answers) - 1) // 2
        if simulated:
            return Check(name="certificate", step=5, status=FAILED,
                         values={"certificate": bound, "floor": ruling.floor, "sigma": float(ruling.sigma),
                                 "simulated_pairs": len(simulated), "pairs": open_pairs},
                         reason=(f"the certificate for {len(simulated)} of {open_pairs} open pair(s) was scored "
                                 f"on samples drawn from the forward model, not on readings. It inherits the "
                                 f"model's errors and tests the library against its own law; it cannot admit"))
    if bound is None:
        return _not_run("certificate", 5, "certificates from certify(), one per open pair, scored on held-out readings")
    values = {"certificate": bound, "floor": ruling.floor, "sigma": float(ruling.sigma)}
    if certificates is None and ruling.certificate is not None and bound >= ruling.floor and ruling.rule is Rule.ADMISSIBLE:
        # The ruling was offered certificates that clear the floor and it did
        # not admit. decide() does that in exactly one case.
        return Check(name="certificate", step=5, status=FAILED, values=values,
                     reason=(f"the ruling carries a certificate of {bound:.4f} that clears the floor "
                             f"{ruling.floor:.4f} and is admissible, not admitted: the certificate was scored "
                             f"on the model's own samples"))
    if bound >= ruling.floor:
        return Check(name="certificate", step=5, status=PASSED, values=values,
                     reason=f"a rule on held-out readings separates at {bound:.4f}, above the floor {ruling.floor:.4f}")
    return Check(name="certificate", step=5, status=FAILED, values=values,
                 reason=(f"the best rule on held-out readings separates at {bound:.4f}, below the floor "
                         f"{ruling.floor:.4f}: admission would rest on the model alone"))


def check_stability(
    ruling: Ruling,
    answers: AnswerSet | None,
    perturbations: Sequence[ForwardModel] | None,
    conditions: Conditions | None,
    *,
    result: SigmaResult | None = None,
    draws: int,
    rng: np.random.Generator | int | None,
) -> Check:
    """Step 3. Recount the verdict under the model author's own perturbations.

    T14: a structural zero holds under every LAW of the statistic that keeps
    the locus map, and a magnitude does not. So the perturbations, alternative
    laws the author is prepared to defend as equally plausible, may move sigma
    as much as they like and may move an admission across the floor, and that
    is reported as a magnitude that is not robust. What they may not do is
    turn a zero into a positive value or a positive value into a zero: a zero
    that flips was never structural, and an admission a plausible law annuls
    was a fit. Measured on the mesh before this existed: 9 of 42 verdicts
    moved, 0 of 13 refusals, with both models reading one locus map.

    What this does NOT test is the map. Perturbations share the ruling's
    loci by construction, so a zero this check protects is a zero under the
    map the author wrote. A merge the world does not make is `check_merge`'s
    question, and it is a different check because it needs a different map.
    """
    if answers is None or perturbations is None or conditions is None:
        return _not_run("stability", 3, "the answer set, the conditions, and `perturbations`: alternative models the author defends as equally plausible")
    if len(perturbations) == 0:
        return _not_run("stability", 3, "at least one perturbed model")
    base = _base_sigma(ruling, result)
    # Judge the base the way the perturbations are judged: a sampled estimate
    # of a structural zero is a small positive number, and reading the base by
    # its rule while reading the perturbations by `_is_zero` flagged an
    # identical model as a zero flip.
    base_zero = _is_zero(result) if result is not None else ruling.rule is Rule.ABSTAIN
    scored = [sigma(answers, m, conditions, draws=draws, rng=rng) for m in perturbations]
    values_by_model = [float(r.value) for r in scored]
    zero_flips = [r.value for r in scored if base_zero != _is_zero(r)]
    floor_flips = [r.value for r in scored if (base >= ruling.floor) != (float(r.value) >= ruling.floor)]
    values = {"sigma": base, "floor": ruling.floor, "perturbed": values_by_model,
              "zero_flips": len(zero_flips), "floor_crossings": len(floor_flips)}
    if zero_flips:
        return Check(name="stability", step=3, status=FAILED, values=values,
                     reason=(f"{len(zero_flips)} of {len(perturbations)} perturbation(s) move sigma across zero "
                             f"(from {base:.4f}): the zero was not structural, or the admission rests on a law "
                             f"an equally plausible one annuls"))
    if floor_flips:
        return Check(name="stability", step=3, status=FAILED, values=values,
                     reason=(f"{len(floor_flips)} of {len(perturbations)} perturbation(s) move sigma across the "
                             f"floor {ruling.floor:.4f} (from {base:.4f}): the disposition rests on a magnitude, "
                             f"and magnitudes are not robust to the model"))
    return Check(name="stability", step=3, status=PASSED, values=values,
                 reason=(f"sigma {base:.4f} stays on its side of zero and of the floor {ruling.floor:.4f} under "
                         f"{len(perturbations)} perturbation(s), range [{min(values_by_model):.4f}, {max(values_by_model):.4f}]"))


def check_merge(
    ruling: Ruling,
    alternatives: Sequence[tuple[AnswerSet, ForwardModel]] | None,
    conditions: Conditions | None,
    *,
    draws: int,
    rng: np.random.Generator | int | None,
) -> Check:
    """Step 3. A zero must survive a change of MAP, not only a change of law.

    The mirror of `check_scope`. Leaving a state out of the vocabulary can
    only raise sigma, so the one-sided error the audit was built around runs
    toward admission. Merging two states at one locus runs the other way: it
    manufactures a zero, the audit grades it "refused, structural", and
    nothing else looks at it, because T14 only ever varied the law. The
    flagship example is exactly this shape. Still occupant against empty room
    is a zero under a locus map with one entry for both; a Fresnel map puts a
    body inside the footprint at its own locus and splits it.

    So this check takes alternative maps, each as an answer set over the same
    question and a model that places the states differently, and scores the
    refused claim under each. A zero that becomes positive under an
    alternative map was authored, not structural, and the refusal rests on
    the map. The check reads refusals only; an admissible or admitted claim
    is not a zero and has nothing here to lose.
    """
    if ruling.rule is not Rule.ABSTAIN:
        return _not_run("merge", 3, f"a zero ruling: the merge check reads refusals only, and this one is {ruling.rule.value}")
    if alternatives is None or conditions is None:
        return _not_run("merge", 3, "alternative locus maps as `alternatives=[(answers, model), ...]`, and the conditions")
    if len(alternatives) == 0:
        return _not_run("merge", 3, "at least one alternative map")
    scored = [sigma(alt_answers, alt_model, conditions, draws=draws, rng=rng)
              for alt_answers, alt_model in alternatives]
    split = [float(r.value) for r in scored if not _is_zero(r)]
    values = {"alternatives": len(alternatives), "split_by": len(split),
              "sigma_under_alternatives": [float(r.value) for r in scored]}
    if split:
        return Check(name="merge", step=3, status=FAILED, values=values,
                     reason=(f"{len(split)} of {len(alternatives)} alternative map(s) split the zero "
                             f"(sigma up to {max(split):.4f}). The refusal rests on a merge the author "
                             f"wrote, not on the world: T14 protects a zero against a change of law, "
                             f"never against a change of map"))
    return Check(name="merge", step=3, status=PASSED, values=values,
                 reason=f"the zero holds under {len(alternatives)} alternative map(s)")


def check_statistic(result: SigmaResult | None, decision_statistic: str | None) -> Check:
    """Step 4. The statistic sigma was scored on must be the one the decision uses (T15).

    The library cannot read the decision code. It can read the name the
    scoring recorded and the name the caller says the decision uses, and
    refuse when they differ, which is the case that moved 9 of 42 verdicts
    when it was tried. Two names that agree are a claim by the caller, and the
    check says so.
    """
    recorded = None if result is None else result.provenance.get("statistic")
    if recorded is None or decision_statistic is None:
        return _not_run("statistic", 4, "a SigmaResult scored with statistic= and `decision_statistic`, the name of the statistic the decision rule reads")
    values = {"scored_on": str(recorded), "decision_reads": str(decision_statistic)}
    if str(recorded) != str(decision_statistic):
        return Check(name="statistic", step=4, status=FAILED, values=values,
                     reason=(f"sigma was scored on {recorded!r} and the decision reads {decision_statistic!r}: "
                             f"a sigma over a statistic the decision does not use bounds nothing the decision does"))
    return Check(name="statistic", step=4, status=PASSED, values=values,
                 reason=f"sigma was scored on the statistic the decision reads, {recorded!r}, by the caller's own account")


def check_seed(
    ruling: Ruling,
    model: ForwardModel | None,
    answers: AnswerSet | None,
    conditions: Conditions | None,
    *,
    draws: int,
    seeds: Sequence[int],
) -> Check:
    """Step 4. How much of the value is Monte Carlo noise, against the floor."""
    if model is None or answers is None or conditions is None:
        return _not_run("seed", 4, "the model, the answer set and the conditions, to recompute sigma under several seeds")
    stability = seed_stability(lambda seed: sigma(answers, model, conditions, draws=draws, rng=seed).value, seeds)
    values = {"mean": stability.mean, "spread": stability.spread, "low": stability.low, "high": stability.high, "floor": ruling.floor}
    if stability.threatens(ruling.floor):
        return Check(name="seed", step=4, status=FAILED, values=values,
                     reason=(f"the seed alone moves sigma across the floor: [{stability.low:.4f}, "
                             f"{stability.high:.4f}] around {ruling.floor:.4f}"))
    return Check(name="seed", step=4, status=PASSED, values=values,
                 reason=f"sigma spread {stability.spread:.4f} over {len(seeds)} seeds, floor {ruling.floor:.4f} not in reach")


# ---------------------------------------------------------------------------
# the one call
# ---------------------------------------------------------------------------
def audit(
    ruling: Ruling,
    *,
    answers: AnswerSet | None = None,
    model: ForwardModel | None = None,
    conditions: Conditions | None = None,
    against: Conditions | None = None,
    policy: Callable[[object], object] | None = None,
    states: Sequence[object] | None = None,
    certificates: Mapping[tuple[str, str], Certificate] | None = None,
    probes: Sequence[Probe] | None = None,
    probe_tolerance: float = 0.05,
    fine: tuple[AnswerSet, ForwardModel] | None = None,
    refinement_tolerance: float = 0.02,
    misses: Sequence[Mapping[str, Any]] | None = None,
    views: Sequence[ForwardModel] | None = None,
    perturbations: Sequence[ForwardModel] | None = None,
    alternatives: Sequence[tuple[AnswerSet, ForwardModel]] | None = None,
    result: SigmaResult | None = None,
    decision_statistic: str | None = None,
    draws: int = 4000,
    rng: np.random.Generator | int | None = 0,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Audit:
    """Run every check the inputs allow, and grade the ruling.

    Every input is optional. A check whose inputs are missing is reported as
    not_run with what it needed, and the grade counts how many ran. A consumer
    that requires more than the floor reads the audit, not the number.

    Pass `result`, the SigmaResult the ruling was scored from, whenever the
    ruling was made on a drift Bracket: the ruling then carries the bracket's
    low end as its sigma, and the reference, vocabulary and stability checks
    compare against the scored value, not against the bound.
    """
    if not isinstance(ruling, Ruling):
        raise StandingError(f"audit() takes a Ruling from decide(), got {type(ruling).__name__}")
    vocabulary, _ = vocabulary_of(model, states)
    checks = (
        check_quotient(answers, policy, vocabulary),
        check_question(answers, views, conditions, draws=draws, rng=rng),
        check_scope(ruling, misses, vocabulary),
        check_reference(ruling, model, answers, conditions, against, result=result, draws=draws, rng=rng),
        check_probe(probes, tolerance=probe_tolerance, draws=draws, rng=rng),
        check_vocabulary(ruling, fine, conditions, result=result, tolerance=refinement_tolerance, draws=draws, rng=rng),
        check_stability(ruling, answers, perturbations, conditions, result=result, draws=draws, rng=rng),
        check_merge(ruling, alternatives, conditions, draws=draws, rng=rng),
        check_statistic(result, decision_statistic),
        check_seed(ruling, model, answers, conditions, draws=draws, seeds=seeds),
        check_certificate(ruling, certificates),
    )
    return Audit(rule=ruling.rule.value, checks=checks, grade=grade_of(ruling.rule, checks, admitted=ruling.admitted))



# ---------------------------------------------------------------------------
# coverage: what refusing costs, over many rulings
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Coverage:
    """How often the view was allowed to speak, over a run of intervals.

    The other direction of step 5. A verdict that refuses too much is safe and
    it costs coverage, and the cost is visible only in aggregate. `refused`
    counts the structural zeros, `withheld` the CARRY rulings below the floor,
    `admissible` the floor cleared without a certificate, `admitted` the rest
    that may be relied on. Shares are of the total.
    """

    total: int
    refused: int
    withheld: int
    admissible: int
    admitted: int
    recruited: int

    @property
    def share_admitted(self) -> float:
        return (self.admitted + self.recruited) / self.total if self.total else 0.0

    @property
    def share_refused(self) -> float:
        return self.refused / self.total if self.total else 0.0

    @property
    def share_uncertified(self) -> float:
        return self.admissible / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "refused": self.refused, "withheld": self.withheld,
                "admissible": self.admissible, "admitted": self.admitted, "recruited": self.recruited,
                "share_admitted": self.share_admitted, "share_refused": self.share_refused,
                "share_uncertified": self.share_uncertified}


def coverage_report(rulings: Sequence[Ruling]) -> Coverage:
    """Count the dispositions over a run of rulings. Raises on an empty run."""
    if not rulings:
        raise InvalidInput("coverage over no rulings is not a number")
    counts = {rule: 0 for rule in Rule}
    for ruling in rulings:
        if not isinstance(ruling, Ruling):
            raise StandingError(f"coverage_report takes Rulings, got {type(ruling).__name__}")
        if ruling.rule is Rule.RECRUIT and not ruling.admitted:
            # Recruited on the model alone: admissible, and not to be relied on.
            counts[Rule.ADMISSIBLE] += 1
        else:
            counts[ruling.rule] += 1
    return Coverage(
        total=len(rulings),
        refused=counts[Rule.ABSTAIN],
        withheld=counts[Rule.CARRY],
        admissible=counts[Rule.ADMISSIBLE],
        admitted=counts[Rule.ADMITTED],
        recruited=counts[Rule.RECRUIT],
    )
