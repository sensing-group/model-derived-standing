"""Conditions drift, so a measured sigma is a bracket rather than a number.

Sigma is defined at FIXED conditions C, and no deployment holds C fixed. Two
windows recorded hours apart differ in more than their label: a reference has
re-adapted, liveness has changed, a baseline has moved. So comparing two recorded
windows does not estimate sigma; it estimates sigma contaminated by the change in
conditions.

The repair is exact, and it is the reason this library uses squared Hellinger and
not KL. The Hellinger DISTANCE is a metric, so with

    H_meas   the separation you measured between the two windows
    H_null   the same-answer null, one answer against itself across the same gap
    H_true   what sigma is defined as, at fixed conditions

the triangle inequality and its reverse give

    | H_meas - H_null |  <=  H_true  <=  H_meas + H_null

The consequence is a change of posture, not a correction factor. The null is not
a pass/fail check to be tuned until it vanishes; it is the BUDGET for how much of
a measured separation the conditions alone could explain. A team expecting zero
reads a working system as broken, or tunes the sensitivity away. One deployment's
per-view null sits at 0.171 and its signal clears it more than fivefold.

KL admits no such bound, having no triangle inequality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from standing.errors import InvalidInput
from standing.numerics import check_unit

__all__ = ["Bracket", "bracket"]


@dataclass(frozen=True)
class Bracket:
    """What a measurement under drifting conditions can honestly claim.

    Attributes:
        low: the certified lower bound on the true squared Hellinger.
        high: the upper bound.
        measured: the raw separation between the two windows.
        null: the same-answer null used as the drift budget.
    """

    low: float
    high: float
    measured: float
    null: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def clears(self, floor: float) -> bool:
        """Whether the claim is entitled even under the worst drift allowed."""
        return self.low >= check_unit(floor, "floor")

    def ratio(self) -> float:
        """The informal 'n times the drift floor', kept for comparison only.

        Reported because deployments quote it, and flagged because it is not the
        rigorous statement: `low` is.
        """
        if self.null == 0.0:
            return math.inf
        return self.measured / self.null


def bracket(measured_h2: float, null_h2: float) -> Bracket:
    """Bound the true sigma of a pair from a measurement and its same-answer null.

    Args:
        measured_h2: squared Hellinger between the two windows being compared.
        null_h2: squared Hellinger between two windows of the SAME answer,
            separated the same way in time and conditions.

    Returns:
        Bracket, whose `low` is the value a claim may be based on.

    Raises:
        InvalidInput: if either argument is not a squared Hellinger distance.
    """
    measured = check_unit(measured_h2, "measured_h2")
    null = check_unit(null_h2, "null_h2")
    h_meas = math.sqrt(measured)
    h_null = math.sqrt(null)
    low = max(0.0, h_meas - h_null) ** 2
    high = min(1.0, (h_meas + h_null) ** 2)
    if low > high + 1e-12:
        raise InvalidInput("bracket is inverted, which is arithmetically impossible")
    return Bracket(low=low, high=min(high, 1.0), measured=measured, null=null)
