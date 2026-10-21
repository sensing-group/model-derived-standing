"""The estimator is scored against closed forms, never against itself.

The bug this suite exists to prevent read 0.506 where the truth was 0.886 and
saturated near 0.535 instead of approaching 1. It passed every self-consistency
check available, because a biased estimator is perfectly self-consistent. Only
an oracle catches it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standing.divergence import (
    bhattacharyya_gaussian,
    hellinger2_gaussian,
    hellinger2_samples,
    jeffreys_gaussian,
    jeffreys_samples,
)
from standing.errors import InvalidInput


# --- the closed form itself ------------------------------------------------


def test_gaussian_closed_form_endpoints():
    assert hellinger2_gaussian(0.0, 1.0, 0.0, 1.0) == 0.0
    assert hellinger2_gaussian(0.0, 0.0, 0.0, 0.0) == 0.0
    assert hellinger2_gaussian(0.0, 0.0, 3.0, 0.0) == 1.0
    far = hellinger2_gaussian(0.0, 1.0, 40.0, 1.0)
    assert far == pytest.approx(1.0, abs=1e-9)


def test_gaussian_closed_form_is_symmetric_and_monotone():
    previous = 0.0
    for delta in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        value = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
        assert value >= previous - 1e-12
        assert value == pytest.approx(hellinger2_gaussian(delta, 1.0, 0.0, 1.0))
        previous = value


def test_bhattacharyya_matches_the_analytic_expression():
    for m, s1, s2 in ((0.7, 1.0, 2.0), (-3.0, 0.5, 0.5), (2.0, 3.0, 1.0)):
        expected = math.sqrt(2 * s1 * s2 / (s1**2 + s2**2)) * math.exp(
            -(m**2) / (4 * (s1**2 + s2**2))
        )
        assert bhattacharyya_gaussian(0.0, s1, m, s2) == pytest.approx(expected)


def test_reparametrisation_invariance_t5():
    """A monotone change of units must not move the divergence."""
    rng = np.random.default_rng(41)
    p = rng.normal(0.0, 1.0, 4000)
    q = rng.normal(1.2, 1.0, 4000)
    base = hellinger2_samples(p, q)
    for scale, shift in ((10.0, 0.0), (0.1, -5.0), (3.0, 2.5)):
        moved = hellinger2_samples(p * scale + shift, q * scale + shift)
        assert moved == pytest.approx(base, abs=0.02)


# --- the sample estimator against the oracle -------------------------------


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
def test_estimator_tracks_the_closed_form_across_the_range(delta):
    rng = np.random.default_rng(1000 + int(delta * 10))
    truth = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
    estimates = [
        hellinger2_samples(rng.normal(0.0, 1.0, 4000), rng.normal(delta, 1.0, 4000))
        for _ in range(5)
    ]
    assert float(np.mean(estimates)) == pytest.approx(truth, abs=0.035)


def test_estimator_does_not_saturate_below_one():
    """The signature of the bug: large separations must approach 1, not 0.535."""
    rng = np.random.default_rng(7)
    value = hellinger2_samples(rng.normal(0.0, 1.0, 4000), rng.normal(12.0, 1.0, 4000))
    assert value > 0.95


def test_estimator_null_is_small_and_biased_upward_only_slightly():
    rng = np.random.default_rng(2026)
    nulls = [
        hellinger2_samples(rng.normal(0.0, 1.0, 2000), rng.normal(0.0, 1.0, 2000))
        for _ in range(20)
    ]
    assert 0.0 <= float(np.mean(nulls)) < 0.02


def test_estimator_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(13)
    p, q = rng.normal(0.0, 1.0, 1500), rng.normal(0.8, 2.0, 1500)
    assert hellinger2_samples(p, q) == pytest.approx(hellinger2_samples(q, p), abs=1e-12)


def test_estimator_stays_in_the_unit_interval():
    rng = np.random.default_rng(5)
    for _ in range(200):
        p = rng.normal(rng.normal(), abs(rng.normal()) + 0.01, 300)
        q = rng.normal(rng.normal(), abs(rng.normal()) + 0.01, 300)
        assert 0.0 <= hellinger2_samples(p, q) <= 1.0


def test_identical_point_masses_and_distinct_ones():
    assert hellinger2_samples(np.full(50, 2.0), np.full(50, 2.0)) == 0.0
    assert hellinger2_samples(np.full(50, 2.0), np.full(50, 9.0)) == 1.0


# --- jeffreys is provided but must not be mistaken for a metric ------------


def test_jeffreys_closed_form_and_unboundedness():
    assert jeffreys_gaussian(0.0, 1.0, 0.0, 1.0) == 0.0
    assert jeffreys_gaussian(0.0, 1.0, 100.0, 1.0) > 1000.0
    assert jeffreys_gaussian(0.0, 0.0, 1.0, 1.0) == math.inf


def test_jeffreys_violates_the_triangle_inequality_somewhere():
    """Why the bracket needs Hellinger: this is a counterexample, not a bug."""
    a = jeffreys_gaussian(0.0, 1.0, 6.0, 1.0)
    b = jeffreys_gaussian(0.0, 1.0, 3.0, 1.0)
    c = jeffreys_gaussian(3.0, 1.0, 6.0, 1.0)
    assert a > b + c


# --- degenerate input raises ----------------------------------------------


def test_non_finite_samples_raise_rather_than_being_dropped():
    good = np.zeros(10)
    for bad in (np.nan, np.inf, -np.inf):
        contaminated = np.concatenate([good, [bad]])
        with pytest.raises(InvalidInput, match="non-finite"):
            hellinger2_samples(contaminated, good)
        with pytest.raises(InvalidInput):
            jeffreys_samples(good, contaminated)


def test_empty_samples_raise():
    with pytest.raises(InvalidInput, match="empty"):
        hellinger2_samples([], [1.0, 2.0])


def test_negative_sigma_raises():
    with pytest.raises(InvalidInput):
        hellinger2_gaussian(0.0, -1.0, 1.0, 1.0)


# --- second-pass regressions: scale, and what an absolute epsilon did ------


@pytest.mark.parametrize("scale", [1e-18, 1e-15, 1e-8, 1.0, 1e8, 1e15, 1e18])
def test_t5_holds_across_thirty_six_orders_of_magnitude(scale):
    """An absolute degeneracy epsilon broke this, and broke it silently.

    Data near 1e-15 has a span near 1e-15, which a fixed 1e-12 floor calls
    degenerate: two different laws were declared identical, sigma came out 0 and
    the rules ABSTAINED. Measured before the fix: 0.2731 at unit scale, 0.0000
    at 1e-15. The strongest refusal the theory offers, produced by a change of
    units.
    """
    rng = np.random.default_rng(0)
    p, q = rng.normal(0.0, 1.0, 3000), rng.normal(1.5, 1.0, 3000)
    base = hellinger2_samples(p, q)
    assert hellinger2_samples(p * scale, q * scale) == pytest.approx(base, abs=1e-9)


def test_degeneracy_is_still_detected_at_small_scales():
    assert hellinger2_samples(np.full(50, 2e-20), np.full(50, 9e-20)) == 1.0
    assert hellinger2_samples(np.full(50, 2e-20), np.full(50, 2e-20)) == 0.0


def test_a_span_below_floating_point_resolution_is_degenerate_at_any_scale():
    for magnitude in (1e-20, 1.0, 1e20):
        base = np.full(50, magnitude)
        nudged = base * (1.0 + 1e-15)
        assert hellinger2_samples(base, nudged) in (0.0, 1.0)


# ---------------------------------------------------------------------------
# T5 under SHIFT, and the two defects that broke it. The suite already checks
# T5 under SCALE across thirty-six orders; a tolerance measured against the
# data's LOCATION passed all of that while failing on a plain offset.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9, 1e10, 1e11, 1e12, 1e13])
def test_t5_holds_under_a_shift_of_the_statistic(offset):
    """A choice of origin must not move sigma, and once did: 0.9556 -> 0.0.

    The old flatness test compared the span to the data's magnitude, so a
    statistic offset by 1e10 with a spread of 1e-3 was read as a point mass and
    the rules ABSTAINED, which asserts that no rule separates the answers at any
    sample size. Nanosecond timestamps, frequencies and byte counters all reach
    these offsets.
    """
    rng = np.random.default_rng(0)
    truth = hellinger2_gaussian(0.0, 1e-3, 5e-3, 1e-3)
    p = offset + rng.normal(0.0, 1e-3, 20_000)
    q = offset + 5e-3 + rng.normal(0.0, 1e-3, 20_000)
    assert hellinger2_samples(p, q) == pytest.approx(truth, abs=0.02)


def test_an_unresolvable_span_is_refused_rather_than_read_as_a_point_mass():
    """Beyond the representable range the honest answer is a refusal.

    Returning zero would say the answers cannot be told apart, which is a claim
    about the world; the truth is a claim about the caller's units, since the
    same statistic recentred separates perfectly.
    """
    base = 1e13
    step = float(np.spacing(base))  # the smallest difference representable here
    # Each sample carries real spread, so neither is a point mass, but the union
    # spans only a couple of representable steps.
    p = np.array([base, base + step])
    q = np.array([base + step, base + 2 * step])
    assert float(p.max() - p.min()) > 0.0
    with pytest.raises(InvalidInput, match="floating point cannot resolve"):
        hellinger2_samples(p, q)


def test_the_grid_never_asks_for_more_bins_than_the_doubles_can_carry():
    """No bin may be zero-width, or the pseudo-mass is spread over a fiction.

    This once guarded a numpy raise as well, "Too many bins for data range",
    which belongs to the bins-as-a-count form. The grid is an explicit array of
    edges now, so that error cannot occur and the assertion has to be about the
    edges themselves rather than about not crashing.
    """
    from standing.divergence import _grids

    rng = np.random.default_rng(0)
    p = 1e12 + rng.normal(0.0, 1e-3, 20_000)
    q = 1e12 + 5e-3 + rng.normal(0.0, 1e-3, 20_000)
    value = hellinger2_samples(p, q)
    assert 0.0 <= value <= 1.0

    for hp, hq in _grids(p, q, 256):
        assert hp.size == hq.size
        assert np.all(hp > 0.0) and np.all(hq > 0.0)   # pseudo-mass on every bin

    # The equal-width grid at this magnitude must not ask for more edges than
    # the doubles can represent: uncapped it wanted 141 and 75 collapsed.
    low, high = float(min(p.min(), q.min())), float(max(p.max(), q.max()))
    span = high - low
    representable = int(span / float(np.spacing(max(abs(low), abs(high)))))
    assert _grids(p, q, 256)[0][0].size <= representable


def test_both_grids_are_scored_and_the_larger_is_returned():
    """A binning is a kernel, so each value is a lower bound and the max is tighter.

    Dropping either grid survives every other test in this suite: equal-mass
    alone still handles a wide support, and equal-width alone is still accurate
    on an ordinary pair. Only the max is right in both regimes.
    """
    from standing.divergence import _grids

    rng = np.random.default_rng(3)
    p, q = rng.normal(0.0, 1.0, 4000), rng.normal(1.1, 1.0, 4000)
    grids = _grids(p, q, 256)
    assert len(grids) == 2, "both an equal-width and an equal-mass grid"

    per_grid = [
        min(1.0, max(0.0, 1.0 - float(np.sqrt(hp * hq).sum()))) for hp, hq in grids
    ]
    assert min(per_grid) < max(per_grid), "the two grids must actually disagree here"
    assert hellinger2_samples(p, q) == pytest.approx(max(per_grid), abs=1e-12)


@pytest.mark.parametrize("gap", [1e-7, 3e-8, 1e-8, 1e-10, 1e-20])
def test_a_small_but_real_separation_is_never_reported_as_a_structural_zero(gap):
    """1 - BC cancels, and exactly zero is read by decide() as ABSTAIN, i.e. T1.

    Below a mean gap of about 1e-8 the subtraction returned a hard 0.0 for laws
    that are genuinely separable, manufacturing a theorem out of rounding. Same
    defect already fixed once in combine_replicates.
    """
    value = hellinger2_gaussian(0.0, 1.0, gap, 1.0)
    assert value > 0.0
    assert value == pytest.approx(-math.expm1(-(gap**2) / 8.0), rel=1e-9)


def test_the_closed_form_still_calls_identical_laws_exactly_zero():
    """The stable form must not trade the underflow bug for a false positive."""
    assert hellinger2_gaussian(0.0, 1.0, 0.0, 1.0) == 0.0
    assert hellinger2_gaussian(2.5, 0.7, 2.5, 0.7) == 0.0


def test_the_closed_form_keeps_its_point_mass_semantics():
    assert hellinger2_gaussian(0.0, 0.0, 0.0, 0.0) == 0.0
    assert hellinger2_gaussian(0.0, 0.0, 7.0, 0.0) == 1.0
    assert hellinger2_gaussian(0.0, 0.0, 0.0, 1.0) == 1.0


def test_the_closed_form_agrees_with_one_minus_bc_where_that_does_not_cancel():
    """The rewrite must change only the cancelling region, nothing else."""
    for m2, s2 in ((3.0, 1.0), (0.5, 2.0), (-4.0, 0.3), (0.0, 1.5)):
        stable = hellinger2_gaussian(0.0, 1.0, m2, s2)
        naive = 1.0 - bhattacharyya_gaussian(0.0, 1.0, m2, s2)
        assert stable == pytest.approx(naive, rel=1e-12)


# ---------------------------------------------------------------------------
# The grid is equal-MASS. With equal-width bins over the union support the bias
# tracked how wide that support happened to be, which understated a true 0.1516
# as 0.0023 and flipped a ruling from ADMIT to CARRY.
# ---------------------------------------------------------------------------


def _widened(rng, mean, far):
    """The same law, inside a union support stretched by one distant draw."""
    sample = rng.normal(mean, 1.0, 4000)
    return np.concatenate([sample, [float(far)]]) if far else sample


@pytest.mark.parametrize("far", [0, 25, 105, 405])
def test_the_estimate_does_not_collapse_when_the_union_support_is_widened(far):
    rng = np.random.default_rng(11)
    truth = hellinger2_gaussian(0.0, 1.0, 1.0, 1.0)
    value = hellinger2_samples(_widened(rng, 0.0, far), _widened(rng, 1.0, far))
    assert value == pytest.approx(truth, abs=0.03)


def test_the_binding_pair_survives_a_wide_union_support():
    """The binding pair is what a deployment is sent to go and fix."""
    rng = np.random.default_rng(5)

    def draw(centre):
        near = rng.normal(centre, 1.0, 20_000)
        far = rng.normal(60.0, 1.0, 20_000)
        pick = rng.random(20_000) < 0.5
        return np.where(pick, near, far)

    p, q, r = draw(0.0), draw(0.9), draw(0.0)
    r = np.where(rng.random(20_000) < 0.5, rng.normal(0.0, 1.0, 20_000), rng.normal(62.1, 1.0, 20_000))
    pairs = {
        "p|q": hellinger2_samples(p, q),
        "p|r": hellinger2_samples(p, r),
        "q|r": hellinger2_samples(q, r),
    }
    assert min(pairs, key=pairs.get) == "p|q"


def test_the_grid_is_symmetric_in_its_two_samples():
    """Quantiles are taken on the pooled sample, so order cannot matter."""
    rng = np.random.default_rng(2)
    p, q = rng.normal(0.0, 1.0, 5000), rng.normal(1.4, 2.0, 5000)
    assert hellinger2_samples(p, q) == hellinger2_samples(q, p)


def test_a_heavily_tied_sample_gets_a_coarser_grid_rather_than_zero_width_bins():
    rng = np.random.default_rng(3)
    p = np.round(rng.normal(0.0, 1.0, 4000))       # many exact ties
    q = np.round(rng.normal(2.0, 1.0, 4000))
    value = hellinger2_samples(p, q)
    assert 0.0 <= value <= 1.0 and value > 0.1
