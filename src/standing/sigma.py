"""Sigma: the minimum, over pairs of open answers, of the divergence between the
laws they predict for one view's decision statistic.

Computed per view, never on a fused scalar. Fusion cannot create standing (T6),
with equality only if the fusion is sufficient, and a weighted mean is not. A
system that computes sigma after fusing is measuring a strictly smaller quantity
than the one it wanted, so this function takes one view at a time and combination
lives in `numerics`.

Two numerical facts the result carries rather than hides. A sample-based sigma
holds Monte Carlo noise of roughly +/-0.02 at 2000 draws, shrinking as 1/sqrt(N),
which is not negligible against a floor of 0.25. And the estimator is biased
slightly upward at the null, so a pair that is structurally identical should be
decided structurally, which `exact` does when the laws expose a closed form.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from standing.answers import Answer, AnswerSet
from standing.divergence import hellinger2_samples
from standing.errors import InvalidInput
from standing.model import Conditions, ForwardModel

__all__ = ["SigmaResult", "sigma", "sigma_of_pair"]


def _pair_rng(base_seed: int, left: str, right: str) -> np.random.Generator:
    """A generator determined by the seed and the pair, not by iteration order.

    Sharing one generator across pairs makes each pair's value depend on how many
    pairs were drawn before it, so the number recorded for a pair could not be
    reproduced by computing that pair alone: measured at 0.1384 inside the loop
    against 0.1175 on its own. Deriving a child generator per pair fixes the
    value to the pair, which is what a record has to promise.
    """
    # NUL-separated, because "|" is a character a state label may contain and
    # f"{seed}|a|b|c" cannot be told from f"{seed}|a|b|c" read the other way:
    # _pair_rng(0, "a|b", "c") and _pair_rng(0, "a", "b|c") produced identical
    # draws. Labels are now supplied by the caller, so the exposure is wide.
    material = b"\0".join((str(base_seed).encode(), str(left).encode(), str(right).encode()))
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


@dataclass(frozen=True)
class SigmaResult:
    """Sigma and everything needed to re-derive it.

    Attributes:
        value: the minimum over pairs.
        binding_pair: the pair that attained it, which is what to fix.
        per_pair: every pair's value, so the minimum can be audited.
        answers: the set it was scored against. A sigma without its answer set
            cannot be read.
        exact: whether closed forms were used rather than sampling.
        draws: samples per answer when sampling.
        noise: an estimate of the Monte Carlo standard error, None when exact.
    """

    value: float
    binding_pair: tuple[str, str]
    per_pair: Mapping[tuple[str, str], float]
    answers: tuple[str, ...]
    exact: bool
    draws: int | None = None
    noise: float | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)

    def clears(self, floor: float) -> bool:
        return self.value >= floor


def _pair_divergence(
    law_a, law_b, *, draws: int, rng: np.random.Generator
) -> tuple[float, bool]:
    """Closed form when both laws offer one, otherwise a sample estimate.

    An interval in which NOTHING was in a position to report is handled here and
    only here. When both laws produce no draws at all, the two answers predict
    the same absence of a statistic, so sigma is zero structurally rather than by
    estimation. When only one produces none, that is a defect in the model rather
    than a silent interval, and it raises: silently reading it as zero would
    report "nothing could be told apart" for a model that failed to run.
    """
    exact_a = getattr(law_a, "hellinger2_against", None)
    if exact_a is not None and type(law_a) is type(law_b):
        return float(exact_a(law_b)), True
    return _sample_divergence(law_a, law_b, draws=draws, rng=rng)


def _sample_divergence(law_a, law_b, *, draws: int, rng: np.random.Generator):
    """Sample both laws and estimate, handling a silent interval structurally."""
    sample_a = np.asarray(law_a.sample(draws, rng), dtype=float).ravel()
    sample_b = np.asarray(law_b.sample(draws, rng), dtype=float).ravel()
    if sample_a.size == 0 and sample_b.size == 0:
        return 0.0, True
    if sample_a.size == 0 or sample_b.size == 0:
        raise InvalidInput(
            "one answer produced no draws and the other did. A silent interval "
            "gives neither answer a statistic; one empty side is a model defect, "
            "and reading it as zero would report that nothing could be told "
            "apart when in fact nothing ran."
        )
    return hellinger2_samples(sample_a, sample_b), False


def _pair_value(
    model: ForwardModel,
    left: Answer,
    right: Answer,
    conditions: Conditions,
    *,
    draws: int,
    base_seed: int,
    force_samples: bool,
    declared=None,
) -> tuple[float, bool]:
    """One pair's value and whether it was obtained exactly.

    Extracted so that everything scoring a pair goes through the same code. The
    quotient audit compares a state-level value against an answer-level sigma,
    and that comparison means nothing unless both were obtained the same way:
    same structural shortcut, same per-pair generator, same routing under
    force_samples. Two copies of this logic would be two chances to drift.
    """
    if declared is not None and declared(left, right, conditions):
        return 0.0, True
    law_a = model.law(left, conditions)
    law_b = model.law(right, conditions)
    pair_rng = _pair_rng(base_seed, left.name, right.name)
    if force_samples:
        # Route through the same silence handling as the default path. An
        # earlier version called the estimator directly here, so a silent
        # interval returned 0.0 with closed forms and RAISED with
        # force_samples=True: the same interval, two answers, depending on a
        # flag that is supposed to change only how the number is obtained.
        # Keep the exactness the silence path reports. A pair where NEITHER law
        # produced a draw is a structural zero, and returning False here reported
        # it as an estimate: sigma then claimed exact=False with noise computed
        # from zero draws, and Collapse read the pair as "a policy call at the
        # floor you passed" rather than as T1.
        value, was_exact = _sample_divergence(law_a, law_b, draws=draws, rng=pair_rng)
        return float(value), bool(was_exact)
    value, was_exact = _pair_divergence(law_a, law_b, draws=draws, rng=pair_rng)
    return float(value), was_exact


def sigma(
    answers: AnswerSet,
    model: ForwardModel,
    conditions: Conditions,
    *,
    draws: int = 2000,
    rng: np.random.Generator | int | None = None,
    force_samples: bool = False,
    statistic: str | None = None,
) -> SigmaResult:
    """Standing of one view, under these conditions, over these answers.

    Args:
        answers: the open answers. At least two, and no synonyms; both are
            enforced by AnswerSet itself.
        model: your forward model. Receives answers and conditions only. It may
            also expose `indistinguishable(left, right, conditions) -> bool`,
            which asserts a STRUCTURAL zero for that pair; when it returns True
            sigma is exactly 0 and is not estimated.
        conditions: what was true about the observation, never what was observed.
        draws: samples per answer when a closed form is unavailable.
        rng: seed or generator, recorded so the result can be reproduced.
        force_samples: ignore closed forms and sample anyway, which is how the
            sampler gets checked against the exact answer.
        statistic: a name for the decision statistic this view decides on. Two
            results may only be combined if their statistics differ, so naming
            it here is what lets `combine_views` refuse an invalid combination.

    Returns:
        SigmaResult, carrying the binding pair and the full pair table.
    """
    if draws < 2:
        raise InvalidInput(f"draws must be at least 2, got {draws}")
    seed = rng if isinstance(rng, int) or rng is None else None
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    base_seed = seed if seed is not None else int(generator.integers(0, 2**62))

    per_pair: dict[tuple[str, str], float] = {}
    all_exact = True
    declared = getattr(model, "indistinguishable", None)
    for left, right in answers.pairs():
        # A model that KNOWS a pair is unseparable under these conditions must
        # say so, and the answer is exactly zero rather than an estimate near
        # zero. This is not a nicety: the estimator is biased upward at the null
        # by about 0.006 at 2000 draws, and the boundary between CARRY and
        # ABSTAIN sits at exactly zero. Estimating a structural zero therefore
        # turns "no rule can ever separate these" into "identifiable but not
        # here", which is the wrong disposition and the wrong argument. Found by
        # differential test against a deployment that had the shortcut and a
        # library that did not. Handled inside _pair_value.
        value, was_exact = _pair_value(
            model,
            left,
            right,
            conditions,
            draws=draws,
            base_seed=base_seed,
            force_samples=force_samples,
            declared=declared,
        )
        all_exact = all_exact and was_exact
        per_pair[(left.name, right.name)] = float(value)

    binding = min(per_pair, key=per_pair.get)
    return SigmaResult(
        value=per_pair[binding],
        binding_pair=binding,
        # Read-only: a result that a caller can edit after the fact is not
        # provenance, and this one is meant to be checked by someone else.
        per_pair=MappingProxyType(dict(per_pair)),
        answers=answers.names,
        exact=all_exact,
        draws=None if all_exact else draws,
        noise=None if all_exact else 1.0 / np.sqrt(draws),
        provenance=MappingProxyType({
            "divergence": "hellinger2",
            "statistic": statistic,
            "seed": seed,
            "conditions": conditions.key(),
            "pairs": len(per_pair),
        }),
    )


def sigma_of_pair(
    left: Answer,
    right: Answer,
    model: ForwardModel,
    conditions: Conditions,
    *,
    draws: int = 2000,
    rng: np.random.Generator | int | None = None,
    force_samples: bool = False,
) -> float:
    """Divergence for one pair, for callers auditing a binding pair directly.

    Routed through _pair_value, like everything else that scores a pair. It was
    a third copy of that logic and had drifted: it never consulted the model's
    `indistinguishable` hook, so a pair the model DECLARED unseparable came back
    as 0.675 from here while sigma() and the quotient audit both reported an
    exact zero. Its own docstring promised it reproduced the pair table.
    """
    if isinstance(rng, np.random.Generator):
        base_seed = int(rng.integers(0, 2**62))
    else:
        # rng=None picks 0 by convention, not by correspondence. sigma(rng=None)
        # draws its base seed at random, so there is no pair table for this to
        # reproduce and no value here is more correct than another; what matters
        # is that repeated calls agree with each other. A sweep flags changing
        # the constant as surviving, and it survives because the choice is
        # arbitrary rather than because nothing tests it.
        #
        # Worth knowing when reading this: sigma_of_pair(rng=None) is therefore
        # DETERMINISTIC while sigma(rng=None) is not, which is a small asymmetry
        # between two functions whose docstrings otherwise promise to agree.
        base_seed = 0 if rng is None else int(rng)
    value, _ = _pair_value(
        model,
        left,
        right,
        conditions,
        draws=draws,
        base_seed=base_seed,
        force_samples=force_samples,
        declared=getattr(model, "indistinguishable", None),
    )
    return float(value)
