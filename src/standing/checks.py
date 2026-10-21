"""The diagnostics, as functions rather than as a list in a document.

Both of the worst errors a second team made were caught by a null, and neither
would have been caught by reading the formula. A checklist in prose gets read
once; a function gets run every time. So the four checks that matter are here,
each returning a small object that says what it found rather than printing.

    null_calibration    one answer against itself. With one set of conditions
                        that prices the SAMPLER; pass `against` for drift.
    answer_set_ladder   sigma against nested answer sets, because a sigma
                        without the set it was scored against cannot be read.
    quotient_audit      the ladder taken one level DOWN, into the states the
                        quotient grouped together, because a sigma computed over
                        answers can clear a floor while a state inside one answer
                        is indistinguishable from a state inside another.
    quotient_check      the same set against the POLICY rather than the model:
                        is this partition the one the consumer acts on? This is
                        the only check available on the action quotient. It
                        settles whether the two partitions agree, never whether
                        the policy is right.
    seed_stability      how much of the value is Monte Carlo noise.
    redundancy_report   what dependence and saturation do to a combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from standing.answers import Answer, AnswerSet
from standing.divergence import hellinger2_samples
from standing.drift import Bracket, bracket
from standing.errors import InvalidInput
from standing.model import Conditions, ForwardModel
from standing.numerics import (
    check_unit,
    combine_heterogeneous,
    n_eff,
    saturation_count,
)
from standing.sigma import SigmaResult, _pair_value, sigma

__all__ = [
    "NullCalibration",
    "LadderRung",
    "Stability",
    "RedundancyReport",
    "Collapse",
    "QuotientAudit",
    "QuotientCheck",
    "QuotientMerge",
    "QuotientSplit",
    "null_calibration",
    "answer_set_ladder",
    "quotient_audit",
    "quotient_check",
    "seed_stability",
    "redundancy_report",
]


@dataclass(frozen=True)
class NullCalibration:
    """One answer scored against itself. WHAT that prices depends on `against`.

    This class carries `prices` because the two things it can measure were once
    conflated under one name, and the conflation was expensive.

    Called with ONE set of conditions, it draws the same law twice. The
    population separation between a law and itself is exactly zero, so what comes
    back is the estimator's upward bias at the null and nothing else. Measured on
    a law with no drift at all: 0.0127 at 500 draws, 0.0057 at 2000, 0.0005 at
    128000, converging to zero as compute grows. That is a real and useful
    quantity, the noise floor a measured sigma has to clear, but it is NOT drift,
    and reading it as drift makes `bracket` TIGHTEN around whatever number the
    model produced as you spend more compute. On a deliberately misspecified
    model whose truth was 0.1032, the bracket went from [0.6721, 1.0000] at 500
    draws to [0.8517, 0.9375] at 128000: more compute, more confidence, same
    wrong answer.

    Called with `against`, it scores the answer under two different conditions,
    which is what T11 means by drift and the only way to measure it here. That
    value does not vanish with draws, because the two laws really do differ.

    Attributes:
        value: the same-answer separation.
        answer: which answer was used.
        draws: sample size per side.
        prices: "sampler" when the conditions were held fixed, "drift" when two
            sets of conditions were compared. A budget quoted without this cannot
            be read.
    """

    value: float
    answer: str
    draws: int
    prices: str = "sampler"

    @property
    def prices_drift(self) -> bool:
        return self.prices == "drift"

    def budget_for(self, measured: float) -> Bracket:
        """Bracket a measured separation using this null as the budget.

        T11 brackets against DRIFT. If `prices` is "sampler" the bracket is a
        noise bracket instead, which is a weaker and quite different statement:
        it narrows as you add draws, so it can only ever tell you the estimate
        has settled, never that the model is right.
        """
        return bracket(measured, self.value)


@dataclass(frozen=True)
class LadderRung:
    answers: tuple[str, ...]
    value: float


@dataclass(frozen=True)
class Stability:
    """What varying the seed does. Compare `spread` to your margin to the floor.

    A small spread is NOT evidence that the value is right. This measures
    variance, and the estimator's small-sample failure is bias: at two draws two
    identical laws read 0.409 for a true zero with a spread of only 0.031, so
    this reports a settled number for one that is wrong by the whole range. Its
    silence is evidence of nothing on its own. Read it beside a null calibration
    taken at the same draws.
    """

    mean: float
    spread: float
    low: float
    high: float
    seeds: tuple[int, ...]

    def threatens(self, floor: float) -> bool:
        """True when the seed alone can move the answer across the floor."""
        return self.low < floor <= self.high


@dataclass(frozen=True)
class RedundancyReport:
    views: int
    rho: float
    effective: float
    ceiling: float
    combined: float
    saturates_at: float

    @property
    def informative(self) -> bool:
        """False once the combination reports about 1.0 whatever it looks at."""
        return self.effective < self.saturates_at


@dataclass(frozen=True)
class Collapse:
    """Two states in DIFFERENT answers that this view cannot tell apart.

    The quotient is meant to group states that demand the same action. When a
    state inside one answer is indistinguishable from a state inside another, the
    grouping has hidden that fact: the answer-level laws still differ, because
    the other members of the group carry the separation, so sigma clears the
    floor and the claim is admitted. Every verdict the claim then licenses
    against this particular pair is one the view was never entitled to make.

    Attributes:
        left, right: the two state labels.
        left_answer, right_answer: the answers they were grouped into.
        value: their separation, scored exactly as sigma scores an answer pair.
        structural: True when the value is exactly zero and was obtained without
            estimation, so no sample size ever separates them (T1). A structural
            collapse is permanent; one merely below the floor is a policy call at
            the floor you passed.
        mass: the smaller of the two states' within-class shares. This is how
            often the claim can actually be wrong on account of this pair, and it
            is why a collapse needs mass to count at all.
    """

    left: str
    left_answer: str
    right: str
    right_answer: str
    value: float
    structural: bool
    mass: float = 1.0


@dataclass(frozen=True)
class QuotientAudit:
    """What the answer-level sigma is resting on, one level down.

    Attributes:
        answer_sigma: sigma over the answers, the number a ruling would use.
        floor: the admission threshold the audit was run against.
        delta: the mass floor in force. `sound` and `laundered` are relative to
            BOTH numbers and neither may be quoted without the other.
        collapses: cross-answer state pairs below the floor, worst first.
        pairs_scored: how many cross-answer state pairs were compared.
        refines: False when every answer holds a single state, in which case the
            audit is just sigma again and can tell you nothing new.
    """

    answer_sigma: float
    floor: float
    delta: float
    collapses: tuple[Collapse, ...]
    pairs_scored: int
    refines: bool

    @property
    def sound(self) -> bool:
        """True when no state pair straddles an answer boundary below the floor."""
        return not self.collapses

    @property
    def laundered(self) -> bool:
        """The dangerous case: the claim is admitted and a state pair collapses.

        Admitted on the strength of states the view CAN separate, while a pair it
        cannot separate sits inside the same question. This is the one to act on.
        """
        return self.answer_sigma >= self.floor and bool(self.collapses)

    @property
    def worst(self) -> Collapse | None:
        return self.collapses[0] if self.collapses else None

    def reason(self) -> str:
        """One line, in the terms a consumer has to argue with.

        The collapse is reported FIRST. An earlier version checked `refines`
        before `sound`, so on an all-singleton set this line said the audit
        repeated sigma and never mentioned a collapse that `worst`, `sound` and
        `laundered` were all reporting, including when `laundered` was True.
        """
        if self.collapses:
            worst = self.collapses[0]
            kind = (
                "structurally indistinguishable"
                if worst.structural
                else f"separated by only {worst.value:.4f}"
            )
            verdict = "admitted anyway" if self.laundered else "not admitted"
            tail = (
                ""
                if self.refines
                else " (every answer holds one state, so this repeats sigma)"
            )
            return (
                f"sigma {self.answer_sigma:.4f} against floor {self.floor:.4f} at "
                f"delta {self.delta:.2f}, {verdict}, but {worst.left!r} (in "
                f"{worst.left_answer!r}) and {worst.right!r} (in "
                f"{worst.right_answer!r}) are {kind}{tail}"
            )
        if not self.refines:
            return (
                "every answer holds one state, so the quotient groups nothing and "
                "this audit repeats sigma"
            )
        return (
            f"{self.pairs_scored} cross-answer state pairs all separate at or "
            f"above the floor {self.floor:.4f} at delta {self.delta:.2f}"
        )


def null_calibration(
    model: ForwardModel,
    answers: AnswerSet,
    conditions: Conditions,
    *,
    against: Conditions | None = None,
    answer: str | None = None,
    draws: int = 4000,
    rng: np.random.Generator | int | None = None,
) -> NullCalibration:
    """Score one answer against itself, and say what that prices.

    Args:
        conditions: the conditions to score under.
        against: a SECOND set of conditions. Supply it to price DRIFT, which is
            what T11's bracket is about: the same answer under conditions that
            moved. Leave it out to price the estimator's noise floor instead.
            The two are different quantities and the result records which one it
            is, because they were once reported under one name.

    A same-conditions null is positive because the estimator is biased upward at
    zero, and it shrinks as draws grow. Do not read that as the conditions being
    stable, and do not tune until it vanishes: at that point you have only bought
    sample size. A two-conditions null does not shrink with draws, and that is
    how you can tell you measured the world rather than the sampler.
    """
    name = answer or answers.names[0]
    chosen = next((a for a in answers.answers if a.name == name), None)
    if chosen is None:
        raise InvalidInput(f"{name!r} is not in the answer set {answers.names}")
    if against is not None and not isinstance(against, Conditions):
        raise InvalidInput(
            f"against must be a Conditions, got {type(against).__name__}. Drift "
            f"is measured between two sets of conditions; anything else would "
            f"price the sampler again under a different name."
        )
    if against is not None and against.key() == conditions.key():
        raise InvalidInput(
            "against holds the same conditions as the ones being scored, so "
            "there is no drift between them to price and the result would be "
            "the estimator's noise floor wearing the word drift. Omit against "
            "to measure the noise floor deliberately."
        )

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    prices = "drift" if against is not None else "sampler"
    left_law = model.law(chosen, conditions)
    right_law = left_law if against is None else model.law(chosen, against)
    left = np.asarray(left_law.sample(draws, generator), dtype=float).ravel()
    right = np.asarray(right_law.sample(draws, generator), dtype=float).ravel()
    if left.size == 0 or right.size == 0:
        return NullCalibration(value=0.0, answer=name, draws=0, prices=prices)
    return NullCalibration(
        value=hellinger2_samples(left, right),
        answer=name,
        draws=int(draws),
        prices=prices,
    )


def answer_set_ladder(
    sets: Sequence[AnswerSet], compute: Callable[[AnswerSet], float]
) -> list[LadderRung]:
    """Sigma against a ladder of answer sets, coarsest first.

    Report this rather than a single number. Standing is antitone in the set, so
    a value quoted without its set can be made to say almost anything, and the
    ladder shows how much of the answer is the question rather than the view.
    """
    if not sets:
        raise InvalidInput("a ladder needs at least one answer set")
    return [LadderRung(answers=s.names, value=float(compute(s))) for s in sets]


def _canonical(label: object) -> str:
    """A stable name for a state, usable as seed material across processes.

    NOT str(state). The per-pair generator is keyed on the answer names it is
    given, deliberately, so that a pair's value does not depend on iteration
    order. Feeding it str() of an unordered container puts hash order back in:
    with frozenset states, the identity of the worst collapse and its value both
    moved between processes under a fixed seed, while the answer-level sigma it
    is compared against stayed put. Two numbers with different reproducibility
    guarantees cannot be compared, which is the thing this check exists to catch.

    The caller names its own states, so this only has to reject what cannot be a
    name.
    """
    text = str(label)
    if not text:
        raise InvalidInput("a state label is the empty string, so it cannot be named")
    if " object at 0x" in text:
        raise InvalidInput(
            f"the state label {text!r} carries a memory address, which changes "
            f"between runs, so the audit would not be reproducible. Give the "
            f"state a stable label."
        )
    return text


def quotient_audit(
    answers: AnswerSet,
    model: ForwardModel,
    conditions: Conditions,
    *,
    floor: float,
    states: Mapping[str, Mapping[str, object]],
    delta: float,
    weights: Mapping[str, Mapping[str, float]] | None = None,
    result: SigmaResult | None = None,
    draws: int = 2000,
    rng: np.random.Generator | int | None = None,
    force_samples: bool = False,
    max_pairs: int = 2000,
) -> QuotientAudit:
    """Score the states the quotient grouped, and report any that collapse.

    `answer_set_ladder` runs the set coarser and coarser, pricing how much of a
    sigma is the question rather than the view. This runs the other way. Sigma is
    a minimum over ANSWER pairs, so it sees only the law of each group; if one
    answer holds a state this view cannot tell apart from a state in another
    answer, the group's law is a mixture, the separable members carry it away,
    and sigma clears the floor while every verdict resting on that pair is
    unentitled.

    THE CALLER SUPPLIES THE STATES, AND THEIR LABELS. This function will not
    guess what the members of a region are, for two reasons that both bit.
    `Answer.state` is documented as either a frozenset of members or an opaque
    state, and a TUPLE is legal as both: unpacked, a genuine tuple-valued state
    is torn apart; left alone, a tuple region becomes one opaque state and the
    audit silently scores nothing while reporting `sound`. Same answer set, same
    model, opposite verdict, decided by the container the caller happened to
    pick. And rebuilding a state as an Answer named str(state) changed the type
    the model receives, so a model written against the documented contract broke,
    with a refusal message that blamed the model.

    DELTA IS A MASS FLOOR AND IT IS NOT OPTIONAL. Scoring single states walks
    straight into T9: on a state space that is a discretisation of something
    continuous, the two states either side of an action boundary are in different
    answers by construction, and their separation falls like h^2/8 as the grid
    refines. Measured, `laundered` was True at every resolution and got louder
    the better the model was resolved, which is a guaranteed false alarm rather
    than a finding. So a collapse must involve EVENTS carrying real mass: blocks
    of at least `delta` of their answer's within-class prior. Measured, that is
    stable to the fifth decimal across a 25-fold refinement where the singleton
    value fell by 2500.

    Delta is a ladder, not a second magic number. At delta = 1 the audit recovers
    sigma itself; as delta falls it approaches the vacuous state-level minimum.
    Report it as a ladder, the way `answer_set_ladder` is reported, and never
    quote `sound` or `laundered` without both the floor and the delta.

    A within-class prior is admissible input and adds no freedom: `model.law` for
    a grouped answer ALREADY integrates over one, since you cannot produce a
    group's law without weighting its members. Declaring it makes visible what
    was previously undeclared inside the caller's model. A prior ACROSS answers
    is a different object and stays forbidden, because sigma must not depend on
    which answer is true.

    WHAT THIS UNDER-APPROXIMATES, stated plainly. The mass floor is defined over
    EVENTS, so in full generality a collapse is a pair of BLOCKS of states, one
    in each answer, each carrying at least delta. This implementation scores
    single states only, so it sees a collapse only when one individual state
    already carries delta. On a finely gridded state space no single state does,
    and the audit then reports nothing where a block-level search would still
    find a genuine collapse. That error is ONE-SIDED: it misses collapses, it
    never invents them, so `sound` is weaker than it looks while `laundered`
    remains trustworthy when it fires. Choose delta with that in mind, and read
    the ladder rather than one rung.

    Args:
        answers: the set sigma was scored against.
        model: the same forward model, asked about single states as answers
            holding that one state. It receives the state object you supply,
            unchanged.
        conditions: the same conditions.
        floor: the admission threshold the claim faces.
        states: for each answer name, a mapping of label to member state. Labels
            are yours and must be stable across runs.
        delta: mass floor in (0, 1]. A block must carry at least this share of
            its answer's within-class prior to count.
        weights: for each answer name, a mapping of label to relative weight.
            Uniform within each answer when omitted.
        result: a SigmaResult already computed against this same answer set, at
            the same draws and the same force_samples, or it is refused.
        draws, rng, force_samples: passed through, and used for the state blocks,
            so the two levels are obtained the same way.
        max_pairs: refuse rather than run away; the work is quadratic.

    Returns:
        QuotientAudit. Read `laundered` first.
    """
    floor = check_unit(floor, "floor")
    delta = check_unit(delta, "delta")
    if delta <= 0.0:
        raise InvalidInput(
            "delta must be greater than zero. At delta = 0 a collapse needs no "
            "mass at all, which is the state-level minimum T9 says is vacuous on "
            "any discretised state space."
        )
    if not isinstance(answers, AnswerSet):
        raise InvalidInput(f"answers must be an AnswerSet, got {type(answers).__name__}")
    if isinstance(draws, bool) or not isinstance(draws, int):
        raise InvalidInput(f"draws must be an int, got {type(draws).__name__}")
    if draws < 2:
        raise InvalidInput(f"draws must be at least 2, got {draws}")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int):
        raise InvalidInput(f"max_pairs must be an int, got {type(max_pairs).__name__}")
    if max_pairs < 1:
        raise InvalidInput(f"max_pairs must be at least 1, got {max_pairs}")
    if not isinstance(states, Mapping):
        raise InvalidInput(
            f"states must map each answer name to its members, got "
            f"{type(states).__name__}"
        )
    missing = set(answers.names) - set(states)
    if missing:
        raise InvalidInput(
            f"no states given for {sorted(missing)}. Every answer needs its "
            f"members, because an audit that silently skips an answer reports "
            f"sound for a view nobody examined."
        )
    extra = set(states) - set(answers.names)
    if extra:
        raise InvalidInput(f"states names answers that are not open: {sorted(extra)}")

    members: dict[str, list[tuple[str, object, float]]] = {}
    owner: dict[str, list[tuple[str, object, float]]] = {}
    seen: dict[str, object] = {}
    for name in answers.names:
        block = states[name]
        if not isinstance(block, Mapping) or not block:
            raise InvalidInput(
                f"states[{name!r}] must be a non-empty mapping of label to state"
            )
        answer_weights = (weights or {}).get(name, {})
        if not isinstance(answer_weights, Mapping):
            raise InvalidInput(f"weights[{name!r}] must be a mapping of label to weight")
        raw = []
        for label, state in block.items():
            text = _canonical(label)
            if text in seen and seen[text] is not state:
                raise InvalidInput(
                    f"two different states share the label {text!r}, so the audit "
                    f"could not name them apart"
                )
            for other_name, other_state in owner.items():
                if other_name != name and any(
                    st is state or st == state for _, st, _ in other_state
                ):
                    raise InvalidInput(
                        f"the state {text!r} appears in both {other_name!r} and "
                        f"{name!r}. Answers partition the states, so a state in "
                        f"two of them is indistinguishable from itself and would "
                        f"be reported as a structural collapse. Fix the quotient."
                    )
            seen[text] = state
            weight = float(answer_weights.get(label, 1.0))
            if weight < 0.0 or not np.isfinite(weight):
                raise InvalidInput(
                    f"weight for {text!r} is {weight}; within-class weights must "
                    f"be finite and non-negative"
                )
            raw.append((text, state, weight))
        total = sum(w for _, _, w in raw)
        if total <= 0.0:
            raise InvalidInput(
                f"the within-class weights for {name!r} sum to zero, so no state "
                f"in it carries any mass"
            )
        members[name] = [(t, s, w / total) for t, s, w in raw]
        owner[name] = members[name]

    refines = any(len(m) > 1 for m in members.values())

    total_pairs = sum(
        len(members[left.name]) * len(members[right.name])
        for left, right in answers.pairs()
    )
    if total_pairs > max_pairs:
        raise InvalidInput(
            f"this audit would score {total_pairs} cross-answer state pairs, "
            f"above max_pairs={max_pairs}. The work is quadratic in the number "
            f"of states. Raise max_pairs deliberately, or audit coarser states."
        )

    if result is not None:
        if not isinstance(result, SigmaResult):
            raise InvalidInput(f"result must be a SigmaResult, got {type(result).__name__}")
        if set(result.answers) != set(answers.names):
            raise InvalidInput(
                f"the supplied sigma was scored against {result.answers} but this "
                f"audit was given {answers.names}. Comparing a state pair against "
                f"a sigma from a different question says nothing."
            )
        expected_draws = None if force_samples is False and result.exact else draws
        if result.draws is not None and result.draws != draws:
            raise InvalidInput(
                f"the supplied sigma used draws={result.draws} but this audit "
                f"uses draws={draws}. The estimator's bias depends on sample "
                f"size, so the two levels would not be on one scale."
            )
        del expected_draws

    # Derived BEFORE the sigma call. When rng is a Generator, computing sigma
    # here consumes a draw from it, so supplying `result` moved every state-pair
    # value: a documented pure optimisation that changed the report.
    seed = rng if isinstance(rng, int) or rng is None else None
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    base_seed = seed if seed is not None else int(generator.integers(0, 2**62))
    declared = getattr(model, "indistinguishable", None)

    # The name a state is PROBED under is the seed material _pair_rng uses, and
    # it is not always the label. When an answer holds exactly one state, the
    # state pair IS the answer pair, and sigma seeded that pair on the ANSWER
    # names; probing it under the state label would score the same quantity from
    # a different stream. Measured before this: the two levels disagreed on 40 of
    # 40 seeds for one identical pair, which is enough to make `laundered` fire
    # on nothing but Monte Carlo noise. The label is still what gets reported.
    probe_name: dict[str, str] = {}
    for name in answers.names:
        if len(members[name]) == 1:
            probe_name[members[name][0][0]] = name

    # Probing a singleton state under its answer's name merges two namespaces
    # that used to be separate, state labels and answer names, and a clash
    # between them silently hands two DIFFERENT pairs the same seed material. It
    # does not produce a wrong number, but it costs the invariant the probe
    # naming exists to establish: the name a state is probed under must identify
    # the pair. Measured, with a group holding a state labelled 'Q' and a second
    # answer NAMED 'Q', two distinct pairs were both seeded ('Q', 'R'), so their
    # Monte Carlo errors were perfectly correlated and the seed material was no
    # longer a usable provenance key. Reachable whenever answers and states are
    # drawn from one vocabulary, which is exactly the motion-sensor idiom.
    taken: dict[str, tuple[str, str]] = {}
    for name in answers.names:
        for label, _, _ in members[name]:
            probe = probe_name.get(label, label)
            if probe in taken:
                other_answer, other_label = taken[probe]
                raise InvalidInput(
                    f"the state {label!r} in {name!r} and the state "
                    f"{other_label!r} in {other_answer!r} would both be scored "
                    f"under the name {probe!r}, so two different pairs would "
                    f"share one random stream. A state in a single-state answer "
                    f"is scored under that ANSWER's name, so a state label must "
                    f"not collide with another answer's name. Rename one."
                )
            taken[probe] = (name, label)

    singletons: dict[str, Answer] = {}
    for name in answers.names:
        for label, state, _ in members[name]:
            candidate = Answer(name=probe_name.get(label, label), state=state)
            try:
                model.law(candidate, conditions)
            except (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError, RecursionError):
                # Resource and control-flow failures are not model defects and
                # must not be relabelled as one. An earlier version told the
                # caller their model could not answer about a single state when
                # the real event was an out-of-memory.
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised with the contract
                raise InvalidInput(
                    f"the model could not predict for the single state {label!r}: "
                    f"{type(exc).__name__}: {exc}. This audit asks the model "
                    f"about one state at a time, as an answer holding exactly "
                    f"the state object you supplied in `states`."
                ) from exc
            singletons[label] = candidate

    if result is not None:
        answer_sigma = check_unit(result.value, "result.value")
    else:
        answer_sigma = float(
            sigma(answers, model, conditions, draws=draws, rng=rng, force_samples=force_samples)
        .value)

    # A pair carrying delta on both sides and scoring v bounds the answer-level
    # sigma: H2(mix_A, mix_B) <= 1 - sqrt(a*b)*(1 - H2(P,Q)) gives
    # sigma <= 1 - delta*(1 - v). A reportable collapse has v < floor, or v = 0
    # when structural, so the largest sigma that can coexist with one is
    # 1 - delta*(1 - floor). Above that the audit is provably sound and need
    # score nothing.
    #
    # The cutoff was 1 - delta, which is the v = 0 case and is STRICTLY TOO LOW
    # for any positive floor: it discarded pairs that qualified on mass and sat
    # far below the floor. Measured, a pair scoring 0.0308 with mass 0.5 on both
    # sides was skipped at floor 0.30 because sigma came to 0.5148, and reason()
    # then asserted that every pair separated while having scored none of them.
    # It also made `sound` non-monotone in delta in the wrong direction: the same
    # pair was found at delta 0.45 and hidden at delta 0.50, when a stricter mass
    # floor may only ever remove pairs that fail the mass test.
    # STRICTLY greater, and the strictness is load-bearing at floor = 0. The
    # derivation splits on whether the collapse is structural. For a
    # non-structural one, v < floor holds strictly, so sigma < 1 - delta*(1-floor)
    # strictly and equality would be safe. But the `or structural` arm reports a
    # v = 0 pair at ANY floor, and for those the binding cutoff is the v = 0 case,
    # 1 - delta, where equality IS attainable: two answers each holding a
    # collapsed state of mass delta and a far-away state give sigma exactly
    # 1 - delta. At floor = 0 the two cutoffs coincide, and `>=` then skipped a
    # permanent T1 collapse at mass 0.5 while reason() reported that every pair
    # separated. Measured at sigma = 0.5 exactly against sigma = 0.5 - 1e-12.
    #
    # This was documented here as an equivalent mutant. It was not; the reasoning
    # had only considered the non-structural branch.
    if answer_sigma > 1.0 - delta * (1.0 - floor):
        return QuotientAudit(
            answer_sigma=answer_sigma,
            floor=floor,
            delta=delta,
            collapses=(),
            pairs_scored=0,
            refines=refines,
        )

    found: list[Collapse] = []
    scored = 0
    for left_answer, right_answer in answers.pairs():
        for left_label, _, left_mass in members[left_answer.name]:
            for right_label, _, right_mass in members[right_answer.name]:
                left = singletons[left_label]
                right = singletons[right_label]
                try:
                    value, exact = _pair_value(
                        model,
                        left,
                        right,
                        conditions,
                        draws=draws,
                        base_seed=base_seed,
                        force_samples=force_samples,
                        declared=declared,
                    )
                except InvalidInput:
                    raise
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise InvalidInput(
                        f"scoring {left_label!r} against {right_label!r} failed: "
                        f"{type(exc).__name__}: {exc}. A model hook that works "
                        f"for grouped answers must also work for single states, "
                        f"`indistinguishable` included."
                    ) from exc
                scored += 1
                # Only states carrying real mass on BOTH sides can make a claim
                # wrong often enough to matter, and requiring it is what stops
                # the refinement regress.
                if left_mass < delta or right_mass < delta:
                    continue
                structural = bool(exact and value == 0.0)
                # A structural zero is T1 and holds whatever floor was chosen, so
                # it is reported even at floor 0.0, where `value < floor` can
                # never fire and the audit used to announce that every pair
                # separated.
                if value < floor or structural:
                    found.append(
                        Collapse(
                            left=left_label,
                            left_answer=left_answer.name,
                            right=right_label,
                            right_answer=right_answer.name,
                            value=float(value),
                            structural=structural,
                            mass=float(min(left_mass, right_mass)),
                        )
                    )

    # Worst first, and deterministic: labels are stable, so this order is too.
    found.sort(key=lambda c: (c.value, c.left, c.right))
    return QuotientAudit(
        answer_sigma=answer_sigma,
        floor=floor,
        delta=delta,
        collapses=tuple(found),
        pairs_scored=scored,
        refines=refines,
    )



def seed_stability(
    compute: Callable[[int], float], seeds: Sequence[int] = (0, 1, 2, 3, 4)
) -> Stability:
    """Run the same computation under several seeds and report the spread."""
    if len(seeds) < 2:
        raise InvalidInput("stability needs at least two seeds")
    values = [float(compute(int(seed))) for seed in seeds]
    return Stability(
        mean=float(np.mean(values)),
        spread=float(max(values) - min(values)),
        low=float(min(values)),
        high=float(max(values)),
        seeds=tuple(int(s) for s in seeds),
    )


def redundancy_report(
    per_view: Mapping[str, float], rho: float = 0.0, *, eps: float = 0.01
) -> RedundancyReport:
    """What combining these views actually buys, and whether it still informs.

    The combination assumes independence, so with rho above zero the combined
    figure is an upper bound. And past the saturation count it reports about 1.0
    regardless of what it is looking at, which is why a combined number quoted
    without its null is the most impressive and least meaningful thing in a
    report.
    """
    values = list(per_view.values())
    if not values:
        raise InvalidInput("redundancy_report needs at least one view")
    # This report exists to show what dependence costs, so it states the
    # optimistic case and prints rho beside it rather than hiding the choice.
    combined = combine_heterogeneous(values, dependence="independent")
    effective = n_eff(len(values), rho)
    typical = float(np.median(values))
    return RedundancyReport(
        views=len(values),
        rho=float(rho),
        effective=effective,
        ceiling=float("inf") if rho == 0.0 else 1.0 / rho,
        combined=combined,
        saturates_at=saturation_count(typical, eps),
    )


@dataclass(frozen=True)
class QuotientSplit:
    """One place the policy separates two states the answer set grouped.

    This is the dangerous direction. The answer set says these two situations
    are the same question; the policy acts differently in them, so a verdict
    that clears the floor on this answer set was scored on a coarser question
    than the one the deployment actually asks.
    """

    answer: str
    left: str
    right: str
    left_action: str
    right_action: str


@dataclass(frozen=True)
class QuotientMerge:
    """One action the policy takes in states the answer set kept apart.

    This was written as the safe direction and it is not one. The argument for
    safety was that refining adds pairs to a minimum, so sigma can only fall.
    That argument counts the pairs and forgets that the surviving pairs CHANGE:
    a coarse answer's law is the mixture of the fine laws inside it, and a
    mixture can sit closer to a rival than either component does.

    Measured, by quadrature on Gaussians. Two states at N(-2,1) and N(+2,1)
    that the policy treats alike, against a rival at N(0,2). The question the
    consumer actually asks scores 0.0500. The finer declared set scores 0.2677,
    because the minimum is then taken over component pairs and never over the
    mixture the consumer faces. At a floor of 0.25 the real question CARRIES and
    the declared one ADMITS, a 5.35x overstatement in the unsafe direction.

    This is the convex-hull failure in another costume: a situation can be
    separable from every rival individually and still be a blend of them. So
    there is no inequality in either direction and a merge clears `agrees` too.
    """

    action: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class QuotientCheck:
    """What the caller's own policy says about the caller's own answer set.

    The action quotient cannot be checked against the world: whether a policy
    is the right thing to do is a question about the world and not about code.
    It CAN be checked against the policy, which is the half that matters,
    because the failure that has actually happened in the field is a quotient
    coarser than the policy it feeds. Grouping a still occupant with a moving
    one under `occupied` is exactly that failure, and an alarm that treats them
    differently is a witness to it that needs no theory to read.

    Attributes:
        agrees: the two partitions are the same one. Nothing else will do:
            neither direction of disagreement is conservative, and the reasons
            are on `QuotientSplit` and `QuotientMerge` with the numbers.
        states_checked: how many states the policy was run on. A check is only
            as wide as the states it was given, and it says nothing about a
            state nobody sampled.
        splits: the answers the policy tears apart, one witness pair per
            disagreeing pair of actions rather than one per pair of states.
        merges: the actions spanning more than one answer.
        unplaced: states belonging to no answer.
        ambiguous: states belonging to more than one.
    """

    agrees: bool
    states_checked: int
    splits: tuple[QuotientSplit, ...] = ()
    merges: tuple[QuotientMerge, ...] = ()
    unplaced: tuple[str, ...] = ()
    ambiguous: tuple[str, ...] = ()

    def reason(self) -> str:
        """One sentence a caller can put in front of a reviewer."""
        if self.agrees:
            return (
                f"the answer set matches the policy on all {self.states_checked} "
                f"states supplied"
            )
        parts: list[str] = []
        if self.splits:
            worst = self.splits[0]
            parts.append(
                f"{len(self.splits)} split(s): {worst.left} and {worst.right} are "
                f"both {worst.answer!r} but the policy does {worst.left_action} "
                f"in one and {worst.right_action} in the other, so the answer set "
                f"is coarser than the deployment"
            )
        if self.merges:
            worst_merge = self.merges[0]
            parts.append(
                f"{len(self.merges)} merge(s): the policy does "
                f"{worst_merge.action} in {' and '.join(worst_merge.answers)} "
                f"alike, so the answer set is finer than the deployment and "
                f"sigma is a minimum over pairs nobody asked about"
            )
        if self.unplaced:
            parts.append(f"{len(self.unplaced)} state(s) in no answer: {', '.join(self.unplaced[:3])}")
        if self.ambiguous:
            parts.append(f"{len(self.ambiguous)} state(s) in more than one answer")
        return "; ".join(parts)

    def as_dict(self) -> dict:
        """The JSON form a record carries."""
        return {
            "agrees": bool(self.agrees),
            "states_checked": int(self.states_checked),
            "splits": [
                {
                    "answer": s.answer, "left": s.left, "right": s.right,
                    "left_action": s.left_action, "right_action": s.right_action,
                }
                for s in self.splits
            ],
            "merges": [
                {"action": m.action, "answers": list(m.answers)} for m in self.merges
            ],
            "unplaced": list(self.unplaced),
            "ambiguous": list(self.ambiguous),
        }


def _placements(answers: AnswerSet, state: object) -> list[str]:
    """Which answers a state belongs to, resolving the region/point ambiguity.

    A tuple or frozenset is legal both as a region and as one opaque state, and
    `quotient_audit` refuses to guess between them because it has nothing to
    guess WITH. Here there is something: the states were supplied, so equality
    is tried first and membership only after. A state that lands both ways, in
    different answers, is reported as ambiguous rather than resolved by
    preference.
    """
    exact = [a.name for a in answers.answers if _equal(a.state, state)]
    inside = [
        a.name for a in answers.answers
        if isinstance(a.state, (frozenset, set)) and not _equal(a.state, state)
        and _contains(a.state, state)
    ]
    return exact + inside


def _equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except Exception:  # a state with a hostile __eq__ must not take the check down
        return False


def _contains(region: object, state: object) -> bool:
    try:
        return state in region  # type: ignore[operator]
    except TypeError:
        return False


def quotient_check(
    answers: AnswerSet,
    policy: Callable[[object], object],
    states: Sequence[object],
) -> QuotientCheck:
    """Check an answer set against the policy it is supposed to serve.

    Run the policy over the states and compare the partition it induces with
    the one the answer set declares. This is the only check the library can make
    on C1, and it is worth being precise about what it does and does not settle.

    It DOES settle whether the two partitions are the same one. Both ways of
    differing are unsound and neither is conservative, which was not obvious and
    is the reason `agrees` means equality and nothing weaker. Coarser than the
    policy, sigma is scored on a pair the consumer would act differently within.
    Finer than the policy, sigma is a minimum over component pairs while the
    consumer faces the mixture, which overstated a true 0.0500 as 0.2677 across
    a floor of 0.25 in the case on `QuotientMerge`.

    It does NOT settle whether the policy is right. Nothing can. If the alarm
    should distinguish two situations and does not, the policy and the answer set
    agree with each other and are wrong together, and no amount of code will say
    so.

    Args:
        answers: the set a claim will be scored against.
        policy: what the consumer would DO in a state. Any hashable return will
            do; only equality between returns is used.
        states: the states to check on. The check is exactly as wide as this
            list, which is why `states_checked` is reported beside the verdict.

    Raises:
        InvalidInput: if the policy is not callable, if fewer than two states
            were supplied, or if the policy fails or returns something
            unhashable on a state.
    """
    if not callable(policy):
        raise InvalidInput(
            f"policy must be callable, got {type(policy).__name__}. It is the "
            f"action a consumer would take in a state, which is the same "
            f"function AnswerSet.from_actions takes."
        )
    sampled = list(states)
    if len(sampled) < 2:
        raise InvalidInput(
            f"quotient_check needs at least two states, got {len(sampled)}: with "
            f"one state there is no pair to agree or disagree about, and a check "
            f"that cannot fail is not a check"
        )

    acted: list[tuple[object, object]] = []
    for state in sampled:
        try:
            action = policy(state)
        except Exception as exc:
            raise InvalidInput(
                f"the policy failed on state {state!r}: {type(exc).__name__}: {exc}. "
                f"Every state you check must be one the policy can act in."
            ) from exc
        try:
            hash(action)
        except TypeError as exc:
            raise InvalidInput(
                f"the policy returned an unhashable action ({type(action).__name__}) "
                f"for state {state!r}. Actions are compared for equality, so use a "
                f"name, a tuple, or a frozenset rather than a dict or list."
            ) from exc
        acted.append((state, action))

    # placement, and the two ways it can go wrong
    placed: list[tuple[object, object, str]] = []
    unplaced: list[str] = []
    ambiguous: list[str] = []
    for state, action in acted:
        names = _placements(answers, state)
        if not names:
            unplaced.append(_canonical(state))
        elif len(names) > 1:
            ambiguous.append(f"{_canonical(state)} in {', '.join(sorted(names))}")
        else:
            placed.append((state, action, names[0]))

    # splits: inside one answer, the policy does two different things
    by_answer: dict[str, dict[object, list[object]]] = {}
    for state, action, name in placed:
        by_answer.setdefault(name, {}).setdefault(action, []).append(state)
    splits: list[QuotientSplit] = []
    for name in sorted(by_answer):
        groups = by_answer[name]
        if len(groups) < 2:
            continue
        # One witness per pair of actions, not per pair of states: an answer
        # holding a thousand states of two kinds has one disagreement to report,
        # not half a million.
        ordered = sorted(groups, key=lambda a: _canonical(a))
        for i, left_action in enumerate(ordered):
            for right_action in ordered[i + 1:]:
                left = min(groups[left_action], key=lambda s: _canonical(s))
                right = min(groups[right_action], key=lambda s: _canonical(s))
                splits.append(QuotientSplit(
                    answer=name,
                    left=_canonical(left), right=_canonical(right),
                    left_action=_canonical(left_action),
                    right_action=_canonical(right_action),
                ))

    # merges: one action, more than one answer
    spread: dict[object, set[str]] = {}
    for _state, action, name in placed:
        spread.setdefault(action, set()).add(name)
    merges = tuple(
        QuotientMerge(action=_canonical(action), answers=tuple(sorted(names)))
        for action, names in sorted(spread.items(), key=lambda kv: _canonical(kv[0]))
        if len(names) > 1
    )

    return QuotientCheck(
        agrees=not splits and not merges and not unplaced and not ambiguous,
        states_checked=len(sampled),
        splits=tuple(splits),
        merges=merges,
        unplaced=tuple(sorted(unplaced)),
        ambiguous=tuple(sorted(ambiguous)),
    )
