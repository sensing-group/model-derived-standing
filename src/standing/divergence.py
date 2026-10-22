"""f-divergences between the laws two answers induce on one statistic.

The family is not a free choice. Standing has to be comparable across views
whose units have nothing to do with each other, and an f-divergence is invariant
under reparametrisation of the observation, so it compares channels rather than
scales.

Squared Hellinger is the default for three reasons that all matter downstream: it
is bounded, so one floor means the same thing everywhere; its square root is a
metric, which is what makes the drift bracket possible at all; and it factorises
over independent views, which is what T7 and T7c are.

THE ESTIMATOR SETTINGS ARE NOT FREE PARAMETERS. An earlier version histogrammed
onto a fixed 200-bin grid over [0, 1] and added one pseudo-count per bin. The
statistic occupied a tenth of that range, so most bins were empty under both
answers and the pseudo-counts rivalled the sample itself. Checked against the
closed form afterwards it was badly wrong: a true 0.886 read 0.506, and
everything above about 0.05 of separation saturated near 0.535 instead of
approaching 1. Two corrections, both verified against Gaussian pairs: the grid
follows the union support of the two samples rather than the range the statistic
could in principle occupy, and the total pseudo-mass is one spread over the bins
rather than one per bin, with the bin count growing as the square root of the
sample size.

The residual bias is upward at the null, about 0.006 at 2000 draws, which is why
a pair that no reporting view separates should be decided structurally rather
than estimated.

THAT FIGURE IS THE LARGE-SAMPLE END AND BADLY UNDERSTATES THE OTHER ONE. On two
identical laws, where the truth is exactly zero, over 300 seeds:

    draws       2       3       5      10      25     100     500    2000
    mean   0.4093  0.4619  0.5155  0.4587  0.2110  0.0443  0.0120  0.0062
    range   0.031   0.047   0.057   0.289   0.314   0.085   0.020   0.007

At a handful of draws the estimate is not noisy, it is CONFIDENTLY WRONG: 0.409
for a true zero, inside a band three hundredths wide. A structurally identical
pair would clear a floor of 0.25 comfortably and be admitted.

Worse, the diagnostic that exists to catch trouble reassures you here.
`seed_stability` reports the spread across seeds, and at two draws that spread
is 0.031, which reads as a settled number. The failure is bias, not variance, so
an instrument measuring variance cannot see it and its silence is evidence of
nothing. There is no natural cut point either: the bias is still 0.21 at 25
draws and 0.044 at 100, so no threshold near the bottom of the range would be
anything but arbitrary. Judge a small-sample sigma against a null measured at
the same draws, which is what `null_calibration` is for.
"""

from __future__ import annotations

import math

import numpy as np

from standing.errors import InvalidInput

__all__ = [
    "hellinger2_samples",
    "hellinger2_gaussian",
    "jeffreys_samples",
    "jeffreys_gaussian",
    "bhattacharyya_gaussian",
]

# A few units in the last place of the data's own magnitude. This asks whether a
# spread is REPRESENTABLE at that magnitude, which is the only scale-free
# question there is, and it is deliberately not a round number like 1e-12: see
# _is_flat for what a round number did.
_ULP_TOL = 4.0 * float(np.finfo(float).eps)


def _same_point(left: float, right: float) -> bool:
    """Whether two point masses sit at one place, judged at their own magnitude."""
    scale = max(abs(left), abs(right), 1e-300)
    return abs(left - right) <= _ULP_TOL * scale


def _is_flat(sample: np.ndarray) -> bool:
    """Whether a sample is a point mass: no spread at all, exactly.

    Two bugs lived in the tolerance this used to carry, and fixing the first
    produced the second.

    An ABSOLUTE epsilon is wrong: data living near 1e-15 has a span near 1e-15,
    which any fixed 1e-12 threshold calls degenerate, so two genuinely different
    laws are declared identical, sigma comes out 0, and the rules ABSTAIN, the
    strongest refusal the theory offers, produced by a choice of units. Scaling
    the same data by 1e-15 moved sigma from 0.2731 to 0.0.

    A tolerance RELATIVE TO LOCATION is also wrong, and fails the same way at the
    other end. A statistic offset by 1e10 with a spread of 1e-3 is perfectly
    resolvable, but 1e-12 * 1e10 is 1e-2, so the sample was read as a point mass:
    measured sigma 0.9556 at offset 0 and exactly 0.0 at offset 1e10, ABSTAIN
    again, this time from a choice of ORIGIN rather than of scale. That breaks T5
    for shifts, and the offsets that reach it are ordinary ones: nanosecond
    timestamps, frequencies, byte counters.

    So there is no tolerance here at all. A point mass is a span of exactly zero,
    which is a statement about the law. A span that is positive but below the
    floating-point resolution is a different statement, about the caller's units,
    and it is refused in _grids rather than decided here. Collapsing
    the two is what let a choice of representation manufacture a theorem.
    """
    low, high = float(sample.min()), float(sample.max())
    return high - low == 0.0


def _clean(sample, name: str) -> np.ndarray:
    """Return a finite 1-D float array, or raise.

    NaN is rejected rather than dropped. Dropping it would change the sample
    size silently and make an estimate that is not the one the caller asked for.
    """
    array = np.asarray(sample, dtype=float).ravel()
    if array.size == 0:
        raise InvalidInput(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise InvalidInput(
            f"{name} holds {bad} non-finite value(s); a divergence computed "
            f"over NaN reads as a measurement and is not one"
        )
    return array


def _degenerate_pair(p: np.ndarray, q: np.ndarray) -> float | None:
    """Decide structurally when BOTH samples carry no spread at all.

    Two constant samples are two point masses, and the answer is 0 if they sit
    at the same place and 1 if they do not. The histogram path must not be used
    here: it spreads a pseudo-mass across the gap between them and reports a
    separation strictly below 1 for two laws that are perfectly separable. That
    is how two point masses at 2 and at 9 read 0.913 instead of 1.

    One degenerate sample against a spread one is NOT decided here, because a
    spread sample is genuine evidence about a law that could overlap the point.
    """
    if not (_is_flat(p) and _is_flat(q)):
        return None
    left, right = float(p[0]), float(q[0])
    scale = max(abs(left), abs(right), 1e-300)
    return 0.0 if abs(left - right) <= _ULP_TOL * scale else 1.0


def _grids(
    p: np.ndarray, q: np.ndarray, max_bins: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every binning this estimator scores on, as smoothed probability vectors.

    TWO GRIDS, AND THE CALLER TAKES THE MAX, which is not a hedge. A binning is a
    kernel applied to the statistic, so by T6 every value computed on one is a
    LOWER bound on the true divergence. The maximum of two lower bounds is a
    lower bound too, and a tighter one, so scoring both grids is strictly better
    than picking either and is still sound in the direction the theory needs.

    The two fail in opposite places, which is why both are kept.

    EQUAL-WIDTH over the union support is accurate for an ordinary narrow pair
    and collapses when the support is wide, because the bias tracks how far apart
    the data happens to sit rather than how different the laws are. Measured at
    2000 draws on a true 0.1175, it reads 0.0057 once one distant draw stretches
    the support: a laundering case with a true 0.1516 read 0.0023, 66 times too
    small, which flipped a ruling from ADMIT to CARRY. It also corrupts the
    BINDING PAIR, the one thing a deployment is told to go and fix, and it does
    not converge: past about 65000 draws the bin count is frozen at max_bins, so
    it approaches the divergence of the 256-bin quantisation rather than the
    divergence itself.

    EQUAL-MASS on quantiles of the pooled sample puts resolution where the mass
    is, so it holds up under a wide support (0.1207 against that same 0.1175),
    but it is slightly the worse of the two on ordinary narrow pairs because it
    spends its bins in the dense middle rather than on the shoulders where the
    two densities differ.

    Measured mean absolute error against the closed form, 2000 draws:

        regime                 equal-width   equal-mass   the max
        ordinary narrow pairs      0.01170      0.01429   0.01156
        wide union support         0.00572*     0.12069   0.12069
        null, identical laws       0.00578      0.00552   0.00615

    (* against a truth of 0.11750, so that cell is the failure, not a success.)

    The max is at least as good as the better of the two in every regime, at a
    cost of 0.0004 in null bias.
    """
    low = float(min(p.min(), q.min()))
    high = float(max(p.max(), q.max()))
    scale = max(abs(low), abs(high), 1e-300)
    # A span too small to represent at this magnitude is REFUSED, not decided.
    # Reading it as a point mass returns 0.0, which sigma reports as exact and
    # decide reads as ABSTAIN, asserting that no rule separates the pair at any
    # sample size. That is a claim about the world, and the truth here is a claim
    # about the caller's units: the same statistic recentred separates perfectly.
    if high - low <= _ULP_TOL * scale:
        raise InvalidInput(
            f"the two samples span {high - low:.3e} around a magnitude of "
            f"{scale:.3e}, which floating point cannot resolve here, so no "
            f"divergence between them is measurable. This is a property of the "
            f"units, not of the laws: recentre or rescale the statistic (a "
            f"timestamp or a counter offset from zero is the usual cause). "
            f"Returning zero would report that nothing can tell the answers "
            f"apart, which is a theorem this data cannot support."
        )

    resolved = int(min(max_bins, max(16, round(min(p.size, q.size) ** 0.5))))

    # A grid can be no finer than the floating-point spacing at this magnitude.
    # Far from the origin that spacing is coarse: at 1e12 a double resolves about
    # 1.2e-4, so a span of 8e-3 admits only 66 distinct edges, and asking for 141
    # collapses 75 of them to zero width. Those bins can never hold a draw, yet
    # each still receives its share of the pseudo-mass, so the smoothing is
    # spread over a grid that is mostly fictional.
    #
    # This used to be a crash guard as well: with `bins=count, range=(lo, hi)`
    # numpy raised "Too many bins for data range" straight through the library.
    # It does not raise any more, because the grid is now built as an explicit
    # array of edges and that error belongs to the count form. The cap is
    # therefore doing quieter work than its first comment claimed, and is kept
    # for the degenerate-bin reason rather than the crash.
    spacing = float(np.spacing(scale))
    representable = int((high - low) / spacing) if spacing > 0.0 else resolved
    resolved = max(2, min(resolved, representable))

    grids: list[tuple[np.ndarray, np.ndarray]] = []

    pad = 0.02 * (high - low)
    edges_width = np.linspace(low - pad, high + pad, resolved + 1)

    # Quantiles of the pooled sample, symmetric in p and q as the estimator must
    # be. Ties collapse edges, so duplicates are dropped rather than producing
    # zero-width bins; a heavily tied sample simply gets a coarser grid, which is
    # the honest resolution for it.
    pooled = np.concatenate((p, q))
    edges_mass = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, resolved + 1)))

    for edges in (edges_width, edges_mass):
        if edges.size < 2:
            continue
        edges = edges.copy()
        # np.histogram is right-closed on the last bin only, so nudge the outer
        # edges outward to keep every draw inside the grid.
        edges[0] = np.nextafter(edges[0], -np.inf)
        edges[-1] = np.nextafter(edges[-1], np.inf)
        bins = edges.size - 1
        alpha = 1.0 / bins  # total pseudo-mass of one, spread over the grid
        hp, _ = np.histogram(p, bins=edges)
        hq, _ = np.histogram(q, bins=edges)
        sp = hp.astype(float) + alpha
        sq = hq.astype(float) + alpha
        grids.append((sp / sp.sum(), sq / sq.sum()))
    return grids


def hellinger2_samples(p, q, max_bins: int = 256) -> float:
    """Squared Hellinger in [0, 1] between two samples of one statistic."""
    p = _clean(p, "p")
    q = _clean(q, "q")
    structural = _degenerate_pair(p, q)
    if structural is not None:
        return structural
    grids = _grids(p, q, max_bins)
    if not grids:
        return 0.0
    # Each grid is a kernel, so each value is a LOWER bound on the truth (T6).
    # The tightest of them is still a lower bound.
    return max(
        min(1.0, max(0.0, 1.0 - float(np.sqrt(hp * hq).sum()))) for hp, hq in grids
    )


def jeffreys_samples(p, q, max_bins: int = 256) -> float:
    """Symmetric KL in nats. Unbounded, and no triangle inequality.

    Provided for the discrimination-exponent reading only. It cannot support the
    drift bracket, which needs a metric.
    """
    p = _clean(p, "p")
    q = _clean(q, "q")
    structural = _degenerate_pair(p, q)
    if structural is not None:
        return 0.0 if structural == 0.0 else math.inf
    grids = _grids(p, q, max_bins)
    if not grids:
        return 0.0
    # Same reasoning as the Hellinger path: a kernel can only lose separation.
    return max(
        float(((hp - hq) * (np.log(hp) - np.log(hq))).sum()) for hp, hq in grids
    )


def bhattacharyya_gaussian(m1: float, s1: float, m2: float, s2: float) -> float:
    """Bhattacharyya coefficient between two Gaussians, closed form."""
    for value, name in ((m1, "m1"), (s1, "s1"), (m2, "m2"), (s2, "s2")):
        if not math.isfinite(float(value)):
            raise InvalidInput(f"{name} is not finite")
    if s1 < 0.0 or s2 < 0.0:
        raise InvalidInput("standard deviations must be non-negative")
    v1, v2 = s1 * s1, s2 * s2
    total = v1 + v2
    # Point masses are decided by an EXACT zero spread, never by an absolute
    # epsilon: a spread of 1e-7 in the caller's units is a law with spread, and
    # an epsilon of 1e-6 read it as a point mass, returning 0 (BC) or 1 (H2) for
    # two laws that overlap heavily. Same defect as the one _is_flat documents.
    if total == 0.0:
        return 1.0 if _same_point(m1, m2) else 0.0
    if s1 == 0.0 or s2 == 0.0:
        return 0.0
    return math.sqrt(2.0 * s1 * s2 / total) * math.exp(-((m1 - m2) ** 2) / (4.0 * total))


def hellinger2_gaussian(m1: float, s1: float, m2: float, s2: float) -> float:
    """Squared Hellinger between two Gaussians, closed form.

    The oracle every sample-based estimate is checked against. Two point masses
    at one place are the same law, so zero; at different places they are
    perfectly separable, so one.

    NOT computed as 1 - BC, which cancels. BC approaches 1 as the laws approach
    each other, so the subtraction loses every significant digit exactly where
    the answer matters most, and below a mean gap of about 1e-8 it returns a hard
    0.0 for laws that are genuinely separable. That is not a rounding nuisance:
    `sigma` reports 0.0 as EXACT, `decide` reads exactly zero as ABSTAIN, and
    ABSTAIN asserts T1, that no rule separates the pair at any sample size. So
    floating-point cancellation was manufacturing a theorem. It is the same
    defect already fixed once in `combine_replicates`, whose comment records that
    a small but real standing was reported as a structural zero and the rules
    abstained where they should have carried.

    Writing it through expm1 and log1p keeps the small values instead:

        H2 = -expm1( 0.5 * log1p(-(s1 - s2)^2 / T) - (m1 - m2)^2 / (4T) )

    using 2*s1*s2/T = 1 - (s1 - s2)^2 / T with T = s1^2 + s2^2. Measured at a
    mean gap of 1e-8 this gives 1.25e-17 where the subtraction gave 0.0, and it
    agrees with the old form everywhere the old form did not cancel. It moves the
    threshold from 1e-8 to about 1e-154 rather than removing it, so a structural
    zero from a closed form is only ever as sound as that closed form.
    """
    for value, name in ((m1, "m1"), (s1, "s1"), (m2, "m2"), (s2, "s2")):
        if not math.isfinite(float(value)):
            raise InvalidInput(f"{name} is not finite")
    if s1 < 0.0 or s2 < 0.0:
        raise InvalidInput("standard deviations must be non-negative")

    v1, v2 = s1 * s1, s2 * s2
    total = v1 + v2
    # Point masses keep the semantics stated above, decided rather than computed,
    # and decided by an EXACT zero spread: an absolute epsilon on the variance
    # made the answer depend on the caller's units (0.1175 at scale 1, exactly
    # 1.0 at scale 1e-6 for the same pair), which breaks T5 for scalings.
    if total == 0.0:
        return 0.0 if _same_point(m1, m2) else 1.0
    if s1 == 0.0 or s2 == 0.0:
        return 1.0

    spread_term = math.log1p(-((s1 - s2) ** 2) / total)
    log_bc = 0.5 * spread_term - ((m1 - m2) ** 2) / (4.0 * total)
    return min(1.0, max(0.0, -math.expm1(log_bc)))


def jeffreys_gaussian(m1: float, s1: float, m2: float, s2: float) -> float:
    """Symmetric KL between two Gaussians, in nats."""
    for value, name in ((m1, "m1"), (s1, "s1"), (m2, "m2"), (s2, "s2")):
        if not math.isfinite(float(value)):
            raise InvalidInput(f"{name} is not finite")
    v1, v2 = s1 * s1, s2 * s2
    if v1 + v2 == 0.0:
        return 0.0 if _same_point(m1, m2) else math.inf
    if s1 == 0.0 or s2 == 0.0:
        return math.inf
    d2 = (m1 - m2) ** 2
    return float((v1 + d2) / (2.0 * v2) + (v2 + d2) / (2.0 * v1) - 1.0)
