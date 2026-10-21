"""The combination laws as executable theorems.

Every assertion here is a proposition from the theory rather than a regression
of whatever the code happened to return. Where a proposition has a direction
(which of two wrong formulas overstates), the direction is asserted too, because
only one of the two is dangerous.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standing.errors import InvalidInput
from standing.numerics import (
    combine_heterogeneous,
    combine_replicates,
    h2_bernoulli,
    independence_inflation,
    n_eff,
    saturation_count,
)


# --- T7 and T7c, and the boundary between them ----------------------------


def test_t7_is_the_equal_value_case_of_t7c():
    for h in (0.0, 0.05, 0.17, 0.5, 0.9, 1.0):
        for n in (1, 2, 4, 16):
            assert combine_replicates(h, n) == pytest.approx(
                combine_heterogeneous([h] * n, dependence='independent'), abs=1e-12
            )


def test_max_overstates_and_mean_understates():
    """AM-GM on the (1 - h_i), asserted in both directions over random sets."""
    rng = np.random.default_rng(20260824)
    for _ in range(20000):
        n = int(rng.integers(2, 8))
        hs = rng.random(n) * 0.9
        truth = combine_heterogeneous(hs, dependence='independent')
        with_mean = combine_replicates(float(hs.mean()), n)
        with_max = combine_replicates(float(hs.max()), n)
        assert with_mean <= truth + 1e-12, "mean form must never overstate"
        assert with_max >= truth - 1e-12, "max form must never understate"


def test_a_blind_view_zeroes_replicates_but_not_a_heterogeneous_set():
    assert combine_replicates(0.0, 50) == 0.0
    assert combine_heterogeneous([0.0, 0.5, 0.3], dependence='independent') == pytest.approx(
        combine_heterogeneous([0.5, 0.3], dependence='independent')
    )


def test_combination_is_monotone_and_bounded():
    rng = np.random.default_rng(11)
    for _ in range(2000):
        h = float(rng.random())
        n = int(rng.integers(1, 40))
        a, b = combine_replicates(h, n), combine_replicates(h, n + 1)
        assert 0.0 <= a <= b <= 1.0


def test_heterogeneous_never_below_its_best_member():
    rng = np.random.default_rng(5)
    for _ in range(5000):
        hs = rng.random(int(rng.integers(1, 9)))
        assert combine_heterogeneous(hs, dependence='independent') >= hs.max() - 1e-12


def test_replicates_refuses_a_sequence():
    """The exact shape of the 2.49x overstatement must not be expressible."""
    for wrong in ([0.1, 0.2], (0.1, 0.2), np.array([0.1, 0.2])):
        with pytest.raises(InvalidInput, match="combine_heterogeneous"):
            combine_replicates(wrong, 2)


# --- T8b, the dependence correction ---------------------------------------


def test_n_eff_identity_and_ceiling():
    assert n_eff(37, 0.0) == 37.0
    assert n_eff(1, 0.9) == 1.0
    for rho in (0.05, 0.2, 0.5, 0.9):
        # the ceiling is approached from below and never attained
        assert n_eff(10_000, rho) < 1.0 / rho
        assert n_eff(1_000_000, rho) == pytest.approx(1.0 / rho, rel=1e-3)


def test_n_eff_is_decreasing_in_rho():
    previous = math.inf
    for rho in (0.0, 0.1, 0.3, 0.6, 1.0):
        current = n_eff(20, rho)
        assert current <= previous + 1e-12
        previous = current


def test_dependence_never_raises_the_combination():
    rng = np.random.default_rng(3)
    for _ in range(5000):
        h = float(rng.random())
        n = int(rng.integers(2, 30))
        rho = float(rng.random())
        assert combine_replicates(h, n, rho) <= combine_replicates(h, n) + 1e-12


# --- T8 and T7b, the two reporting hazards --------------------------------


def test_independence_inflation_matches_the_measured_case():
    assert independence_inflation(0.1534, 4) == pytest.approx(3.17, abs=0.005)


def test_inflation_tends_to_n_as_h_vanishes():
    assert independence_inflation(1e-9, 7) == pytest.approx(7.0, rel=1e-6)
    assert independence_inflation(0.0, 7) == 7.0


def test_saturation_count_matches_its_definition():
    for h in (0.05, 0.17, 0.4):
        for eps in (0.1, 0.01, 0.001):
            n = saturation_count(h, eps)
            assert (1.0 - h) ** n == pytest.approx(eps, rel=1e-9)
    assert saturation_count(0.17, 0.01) == pytest.approx(24.7, abs=0.2)
    assert saturation_count(0.0) == math.inf


# --- the Bernoulli instance behind every certificate ----------------------


def test_h2_bernoulli_basics():
    assert h2_bernoulli(0.3, 0.3) == 0.0
    assert h2_bernoulli(0.0, 1.0) == 1.0
    assert h2_bernoulli(0.2, 0.7) == pytest.approx(h2_bernoulli(0.7, 0.2))
    assert h2_bernoulli(0.5, 0.5000001) < 1e-6


def test_h2_bernoulli_matches_the_direct_definition():
    rng = np.random.default_rng(99)
    for _ in range(5000):
        a, b = float(rng.random()), float(rng.random())
        expected = 1.0 - math.sqrt(a * b) - math.sqrt((1 - a) * (1 - b))
        assert h2_bernoulli(a, b) == pytest.approx(expected, abs=1e-12)


# --- degenerate input raises, never returns -------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -0.1, 1.5])
def test_every_entry_point_rejects_unusable_numbers(bad):
    with pytest.raises(InvalidInput):
        h2_bernoulli(bad, 0.5)
    with pytest.raises(InvalidInput):
        combine_replicates(bad, 3)
    with pytest.raises(InvalidInput):
        combine_heterogeneous([0.4, bad], dependence='independent')


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 1.5])
def test_n_eff_rejects_unusable_correlations(bad):
    with pytest.raises(InvalidInput):
        n_eff(4, bad)


def test_negative_correlation_is_allowed_down_to_the_psd_bound():
    """Anticorrelated views carry more than independent ones, so n_eff exceeds n.

    Refusing this outright would refuse a real deployment: per-link correlations
    there run from -0.205 to +0.266 and a negative mean is possible.
    """
    assert n_eff(10, -0.05) > 10.0
    assert n_eff(4, -0.1) == pytest.approx(4.0 / (1.0 - 0.3))
    with pytest.raises(InvalidInput, match="positive semi-definite"):
        n_eff(4, -0.5)  # below -1/(n-1) = -0.3333
    with pytest.raises(InvalidInput, match="diverges"):
        n_eff(4, -1.0 / 3.0)  # exactly at the bound


def test_counts_must_be_positive_integers():
    for bad in (0, -3):
        with pytest.raises(InvalidInput):
            combine_replicates(0.2, bad)
    for bad in (2.0, "4", True):
        with pytest.raises(InvalidInput):
            combine_replicates(0.2, bad)


def test_empty_heterogeneous_set_raises():
    with pytest.raises(InvalidInput):
        combine_heterogeneous([], dependence='independent')


# --- T3: sigma is not aspect-decomposable, so combination has a precondition --


def test_combine_views_refuses_slices_of_one_statistic():
    """Measured on a deployment: each aspect alone 0.0000, joint 0.4288."""
    from standing.answers import Answer, AnswerSet
    from standing.model import Conditions, GaussianLaw
    from standing.numerics import combine_views
    from standing.sigma import sigma

    class Model:
        def __init__(self, offsets):
            self.offsets = offsets

        def law(self, answer, conditions):
            return GaussianLaw(self.offsets[answer.name], 1.0)

    answers = AnswerSet((Answer("a", ("a",)), Answer("b", ("b",))))
    model = Model({"a": 0.0, "b": 2.0})
    conditions = Conditions({"windows": 96})

    same = [
        sigma(answers, model, conditions, statistic="daily"),
        sigma(answers, model, conditions, statistic="daily"),
    ]
    with pytest.raises(InvalidInput, match="not aspect-decomposable"):
        combine_views(same)

    unnamed = [sigma(answers, model, conditions)]
    with pytest.raises(InvalidInput, match="does not name its decision statistic"):
        combine_views(unnamed)

    distinct = [
        sigma(answers, model, conditions, statistic="speech_daily"),
        sigma(answers, model, conditions, statistic="network_daily"),
    ]
    value = combine_views(distinct)
    assert value > max(r.value for r in distinct)
