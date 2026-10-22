"""Standing on a number: the sensitivity zero, the Fisher limit, the resolution.

A failure here is a claim in docs/continuous-standing-research-v1.md being false.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standing.continuous import (
    BandModel,
    bands,
    fisher_information,
    resolution,
    sensitivity,
    sigma_local,
)
from standing.errors import InvalidInput
from standing.model import Conditions
from standing.rules import decide
from standing.sigma import sigma


class Thermometer:
    """Room temperature against heater power p in [0, 1]; the window is a condition."""

    def mean(self, p: float, c: Conditions) -> float:
        if c.get("window") == "open":
            return 14.1
        return 18.0 + 3.5 * p

    def sd(self, p: float, c: Conditions) -> float:
        return 0.1


class Counter:
    """A saturating statistic of a people count n."""

    def mean(self, n: float, c: Conditions) -> float:
        return 1.0 - math.exp(-n / 2.0)

    def sd(self, n: float, c: Conditions) -> float:
        return 0.05


class Wave:
    """A statistic that is not monotone in the quantity."""

    def mean(self, x: float, c: Conditions) -> float:
        return math.sin(x)

    def sd(self, x: float, c: Conditions) -> float:
        return 0.1

    def sensitivity(self, x: float, c: Conditions) -> float:
        return math.cos(x)


CLOSED = Conditions({"window": "closed"})
OPEN = Conditions({"window": "open"})
NONE = Conditions({})


def test_window_open_is_an_exact_zero_for_every_power():
    for p in (0.0, 0.3, 0.7, 1.0):
        r = sigma_local(Thermometer(), p, 0.1, OPEN)
        assert r.insensitive and r.value == 0.0 and r.exact


def test_window_closed_has_standing_and_fisher_matches_closed_form():
    r = sigma_local(Thermometer(), 0.5, 0.1, CLOSED)
    assert not r.insensitive and r.value > 0.25
    assert r.fisher == pytest.approx((3.5 / 0.1) ** 2, rel=1e-6)


def test_small_delta_expansion_matches_exact_hellinger():
    r = sigma_local(Thermometer(), 0.5, 0.002, CLOSED)
    assert r.approximation == pytest.approx(r.value, rel=0.02)


def test_resolution_is_infinite_when_insensitive_and_finite_otherwise():
    assert resolution(Thermometer(), 0.5, OPEN) == math.inf
    d = resolution(Thermometer(), 0.5, CLOSED)
    # Equal spreads: BC = exp(-dm^2 / (8 s^2)); BC / 2 = 0.375 gives dm = s sqrt(8 ln(4/3)).
    expected = 0.1 * math.sqrt(8.0 * math.log(4.0 / 3.0)) / 3.5
    assert d == pytest.approx(expected, rel=1e-3)


def test_resolution_agrees_with_the_floor():
    d = resolution(Thermometer(), 0.5, CLOSED, error=0.375)
    assert sigma_local(Thermometer(), 0.5, d, CLOSED).clears(0.25)
    assert not sigma_local(Thermometer(), 0.5, 0.5 * d, CLOSED).clears(0.25)


def test_decide_abstains_on_the_structural_zero_and_carries_otherwise():
    zero = sigma_local(Thermometer(), 0.5, 0.1, OPEN).as_sigma_result()
    assert not decide(zero, floor=0.25).admitted
    live = sigma_local(Thermometer(), 0.5, 0.1, CLOSED).as_sigma_result()
    assert decide(live, floor=0.25).sigma > 0.25


def test_bands_approach_zero_but_cannot_declare_it():
    answers = bands((0.0, 0.5, 1.0))
    r = sigma(answers, BandModel(Thermometer()), OPEN, draws=4000, rng=1)
    assert not r.exact and r.value < 0.02


def test_bands_lose_standing_at_a_shared_edge_where_points_keep_it():
    answers = bands((0.0, 0.5, 1.0))
    band = sigma(answers, BandModel(Thermometer()), CLOSED, draws=4000, rng=1).value
    point = sigma_local(Thermometer(), 0.25, 0.5, CLOSED).value
    assert 0.0 < band < point


def test_resolution_worsens_as_the_counter_saturates():
    d1 = resolution(Counter(), 1.0, NONE)
    d6 = resolution(Counter(), 6.0, NONE)
    assert d1 < 1.0 < d6


def test_a_stationary_point_is_not_a_structural_zero():
    # At the peak of a sine the sensitivity vanishes, but the reading still
    # moves over a finite step (second order), so standing is tiny and not
    # declared zero. Only a reading that does not move at all earns the zero.
    r = sigma_local(Wave(), math.pi / 2, 1e-4, NONE)
    assert r.sensitivity == pytest.approx(0.0, abs=1e-12)
    assert not r.insensitive and 0.0 < r.value < 1e-12
    assert sigma_local(Wave(), 0.0, 0.1, NONE).value > 0.0


def test_inputs_are_checked():
    with pytest.raises(InvalidInput):
        sigma_local(Thermometer(), 0.5, 0.0, CLOSED)
    with pytest.raises(InvalidInput):
        resolution(Thermometer(), 0.5, CLOSED, error=0.5)
    with pytest.raises(InvalidInput):
        bands((0.0, 1.0))
    with pytest.raises(InvalidInput):
        bands((0.0, 1.0, 1.0))
