"""Heuristic weights, kept deliberately apart from standing.

Two independent teams, given the concept, built a weight: a product of factors
like readiness, liveness and audit status. That is a reasonable engineering
instrument and this module supports it. What it must not do is inherit the
theory's guarantees by sharing a name, which happened once and took hours to
notice.

A weight cannot make four statements that standing can:

    it cannot say a view asserts NOTHING as opposed to a little (T1),
    it cannot rise when the question narrows with the data fixed (T2),
    it cannot bound what fusion may do (T6),
    it cannot be exhibited to a stranger as a rule they can apply (T10).

So `Weight` has no `.sigma`, is not accepted by `rules.decide`, and says all of
this if you reach for the attribute anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from standing.errors import StandingError
from standing.numerics import check_unit

__all__ = ["Weight"]


@dataclass(frozen=True)
class Weight:
    """A product of factors in [0, 1]. Useful, and not standing.

    Attributes:
        factors: named multiplicands, each in [0, 1].
        view: which view this weight describes.
    """

    factors: Mapping[str, float]
    view: str = "unnamed"

    def __post_init__(self) -> None:
        if not self.factors:
            raise StandingError("a weight needs at least one factor")
        cleaned = {
            str(name): check_unit(value, f"factor {name!r}")
            for name, value in self.factors.items()
        }
        object.__setattr__(self, "factors", cleaned)

    @property
    def value(self) -> float:
        product = 1.0
        for factor in self.factors.values():
            product *= factor
        return product

    def __getattr__(self, item: str):
        if item in {"sigma", "standing", "value_sigma"}:
            raise StandingError(
                f"a Weight has no {item!r}. This is a product of factors, not an "
                f"f-divergence: none of T1, T2, T6 or T10 applies to it, so it "
                f"cannot say a view asserts nothing, cannot rise when the "
                f"question narrows, cannot bound a fusion and cannot be "
                f"exhibited as a rule. Compute sigma with standing.sigma if that "
                f"is what you need."
            )
        raise AttributeError(item)
