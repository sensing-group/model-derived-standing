"""Standing when the decision is a number.

The categorical definition compares the law of the reading under two answers.
For a quantity theta the same comparison between theta and theta + delta, at
the resolution delta the consumer acts on, is standing on a number. Two facts
make it well behaved, and both are checked in tests/test_continuous.py.

1. Where the predicted reading does not depend on theta under the conditions,
   the sensitivity is zero and so is standing, EXACTLY and for every theta. The
   thermometer with the window open is this case. Bands lose that zero because
   integrating over a band mixes boundary points; a sensitivity test keeps it.
2. As delta shrinks, the squared Hellinger distance between the laws at theta
   and theta + delta is delta^2 * I(theta) / 8, with I the Fisher information
   the view carries about theta under the conditions. Standing on a number is
   Fisher information at the consumer's resolution, compared with the floor.

Nothing here takes a reading. A ScalarModel predicts the mean and spread of
the statistic from theta and the conditions only, like every forward model in
this library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Protocol, runtime_checkable

import numpy as np

from standing.answers import Answer, AnswerSet
from standing.divergence import hellinger2_gaussian
from standing.errors import InvalidInput
from standing.model import Conditions, GaussianLaw
from standing.sigma import SigmaResult

__all__ = [
    "ScalarModel",
    "LocalResult",
    "sensitivity",
    "fisher_information",
    "sigma_local",
    "resolution",
    "bands",
    "BandModel",
]

#: A few units in the last place at the reading's own magnitude. Insensitivity
#: is judged relative to the predicted mean, never by an absolute number, so a
#: model in tiny units is not declared a structural zero by its units.
_ULP_TOL = 4.0 * float(np.finfo(float).eps)


def _within_ulp(delta_value: float, *magnitudes: float) -> bool:
    scale = max(*(abs(m) for m in magnitudes), 1e-300)
    return abs(delta_value) <= _ULP_TOL * scale


@runtime_checkable
class ScalarModel(Protocol):
    """A forward model for a statistic that is approximately normal in theta."""

    def mean(self, theta: float, conditions: Conditions) -> float:
        """Predicted mean of the statistic when the quantity equals theta."""

    def sd(self, theta: float, conditions: Conditions) -> float:
        """Predicted spread of the statistic when the quantity equals theta."""

    # Optional. A model that knows its derivative supplies it and the finite
    # difference is skipped:
    #
    #     def sensitivity(self, theta, conditions) -> float


@dataclass(frozen=True)
class LocalResult:
    """Standing between theta and theta + delta, and how it was obtained.

    Attributes:
        value: squared Hellinger distance between the two predicted laws, the
            same quantity `sigma` reports for a categorical pair.
        exact: True always; the laws are Gaussian and the form is closed.
        insensitive: True when the predicted mean does not move with theta at
            the reading's own magnitude, in which case `value` is exactly 0.0 by
            declaration, not by subtraction.
        sensitivity: d mean / d theta under the conditions.
        fisher: Fisher information about theta under the conditions.
        approximation: delta^2 * fisher / 8, the small-delta expansion of
            `value`, reported so a reader can see how local the regime is.
    """

    theta: float
    delta: float
    value: float
    exact: bool
    insensitive: bool
    sensitivity: float
    fisher: float
    approximation: float
    provenance: Mapping[str, object] = field(default_factory=dict)

    def clears(self, floor: float) -> bool:
        return self.value >= floor

    def as_sigma_result(self) -> SigmaResult:
        """Wrap as a SigmaResult so `decide` applies the ordinary rules."""
        left, right = _label(self.theta), _label(self.theta + self.delta)
        return SigmaResult(
            value=self.value,
            binding_pair=(left, right),
            per_pair={(left, right): self.value},
            answers=(left, right),
            exact=True,
            provenance={"continuous": True, **dict(self.provenance)},
        )


def _label(theta: float) -> str:
    return f"{theta:.6g}"


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise InvalidInput(f"{name} must be finite, got {value!r}")
    return value


def sensitivity(
    model: ScalarModel, theta: float, conditions: Conditions, *, step: float | None = None
) -> float:
    """d mean / d theta under the conditions, closed form if the model has one."""
    theta = _finite(theta, "theta")
    own = getattr(model, "sensitivity", None)
    if callable(own):
        return _finite(own(theta, conditions), "sensitivity")
    h = step if step is not None else max(1e-6, 1e-6 * abs(theta))
    up = _finite(model.mean(theta + h, conditions), "mean")
    down = _finite(model.mean(theta - h, conditions), "mean")
    return (up - down) / (2.0 * h)


def fisher_information(model: ScalarModel, theta: float, conditions: Conditions) -> float:
    """Fisher information of a Gaussian statistic about theta.

    I = (m')^2 / s^2 + 2 (s')^2 / s^2. The second term is the spread's own
    dependence on theta, which is zero for most forward models and kept so a
    model whose noise grows with the quantity is not under-credited.
    """
    theta = _finite(theta, "theta")
    s = _finite(model.sd(theta, conditions), "sd")
    if s <= 0.0:
        raise InvalidInput("Fisher information needs a positive spread")
    m1 = sensitivity(model, theta, conditions)
    h = max(1e-6, 1e-6 * abs(theta))
    s1 = (
        _finite(model.sd(theta + h, conditions), "sd")
        - _finite(model.sd(theta - h, conditions), "sd")
    ) / (2.0 * h)
    return (m1 * m1 + 2.0 * s1 * s1) / (s * s)


def sigma_local(
    model: ScalarModel, theta: float, delta: float, conditions: Conditions
) -> LocalResult:
    """Standing between theta and theta + delta under the conditions.

    Args:
        model: predicts mean and spread from theta and conditions only.
        theta: the value at which standing is asked.
        delta: the resolution the consumer acts on. Yours to state; see
            `resolution` for the sensor's own limit at an error level.
        conditions: what was true about the interval, never what was read.

    Returns:
        LocalResult. `value` is exactly 0.0 when the reading is insensitive to
        theta under these conditions, which is the structural zero of the
        categorical case carried over; otherwise it is the closed-form squared
        Hellinger distance between the two Gaussian laws.
    """
    theta = _finite(theta, "theta")
    delta = _finite(delta, "delta")
    if delta <= 0.0:
        raise InvalidInput("delta must be positive")
    m1 = sensitivity(model, theta, conditions)
    fisher = fisher_information(model, theta, conditions)
    approx = min(1.0, delta * delta * fisher / 8.0)
    m_here = _finite(model.mean(theta, conditions), "mean")
    m_there = _finite(model.mean(theta + delta, conditions), "mean")
    insensitive = _within_ulp(m_there - m_here, m_here, m_there) and _within_ulp(
        m1 * delta, m_here, m_there
    )
    if insensitive:
        value = 0.0
    else:
        a = GaussianLaw(model.mean(theta, conditions), model.sd(theta, conditions))
        b = GaussianLaw(
            model.mean(theta + delta, conditions), model.sd(theta + delta, conditions)
        )
        value = hellinger2_gaussian(a.mean, a.sd, b.mean, b.sd)
    return LocalResult(
        theta=theta,
        delta=delta,
        value=float(value),
        exact=True,
        insensitive=insensitive,
        sensitivity=m1,
        fisher=fisher,
        approximation=float(approx),
        provenance={"conditions": conditions.key()},
    )


def resolution(
    model: ScalarModel, theta: float, conditions: Conditions, *, error: float = 0.375
) -> float:
    """The smallest difference in theta the view can resolve at an error level.

    Solves for the delta at which the Bhattacharyya bound on the Bayes error of
    telling theta from theta + delta, one half of the Bhattacharyya coefficient
    with equal priors, equals `error`. The default 0.375 is the bound the floor
    tau = 0.25 implies in the categorical case, so the two constants agree.
    Returns math.inf when the reading is insensitive to theta.
    """
    theta = _finite(theta, "theta")
    if not 0.0 < error < 0.5:
        raise InvalidInput("error must lie strictly between 0 and 0.5")
    target_h2 = 1.0 - 2.0 * error  # H2 = 1 - BC and the bound is BC / 2

    def h2(delta: float) -> float:
        return sigma_local(model, theta, delta, conditions).value

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if h2(hi) >= target_h2:
            break
        hi *= 2.0
    else:
        return math.inf
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if h2(mid) >= target_h2:
            hi = mid
        else:
            lo = mid
    return hi


def bands(edges: tuple[float, ...] | list[float], *, name_of=None) -> AnswerSet:
    """Answers as consecutive bands of a number, for the region formulation.

    Each answer's state is the (low, high) tuple of its band, so a forward
    model can integrate over it. Compare with `sigma_local`: bands always
    overlap at a shared edge, so they never yield an exact zero on their own.
    """
    edges = tuple(float(e) for e in edges)
    if len(edges) < 3:
        raise InvalidInput("bands need at least three edges, hence two bands")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise InvalidInput("band edges must increase strictly")
    label = name_of or (lambda lo, hi: f"[{lo:g}, {hi:g})")
    return AnswerSet(
        tuple(Answer(name=label(lo, hi), state=(lo, hi)) for lo, hi in zip(edges, edges[1:]))
    )


class _MixtureLaw:
    """The law of the statistic when theta is uniform over a band."""

    def __init__(self, model: ScalarModel, band: tuple[float, float], conditions: Conditions):
        self.model, self.band, self.conditions = model, band, conditions

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self.band
        thetas = rng.uniform(lo, hi, int(size))
        means = np.array([self.model.mean(t, self.conditions) for t in thetas])
        sds = np.array([self.model.sd(t, self.conditions) for t in thetas])
        return rng.normal(means, sds)


@dataclass(frozen=True)
class BandModel:
    """Adapter: a ScalarModel seen through `bands`, for the ordinary `sigma`.

    Exists so the two formulations can be compared on one model. The band law
    is a mixture and has no closed form, so `sigma` samples it; that is also
    why the band formulation can only approach zero and never declare it.
    """

    scalar: ScalarModel

    def law(self, answer: Answer, conditions: Conditions) -> _MixtureLaw:
        state = answer.state
        if not (isinstance(state, tuple) and len(state) == 2):
            raise InvalidInput("BandModel expects answers built by `bands`")
        return _MixtureLaw(self.scalar, (float(state[0]), float(state[1])), conditions)
