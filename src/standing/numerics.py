"""The combination laws, with the distinction the API exists to enforce.

Two views combine one way and N copies of one view combine another, and the
formulas are close enough to substitute for each other silently. They must not
share a function:

    combine_replicates(h, n)     T7    N conditionally independent REPLICATES of
                                       one view, all with the same value h.
                                       1 - (1 - h)^n.

    combine_heterogeneous(hs, dependence=...)  T7c   views that are NOT replicates,
                                       each with its own h_i.
                                       1 - prod_i (1 - h_i).

T7 is the equal-value case of T7c. Substituting a summary value into the
exponent form is invalid in a known direction: with the maximum it OVERSTATES,
with the mean it UNDERSTATES, by AM-GM on the (1 - h_i). A deployment applied
the replicate law to heterogeneous views and overstated by 2.49x, which only a
controlled null caught.

The corollary that trips people is the reverse of the one they remember. A view
with h = 0 forces the whole REPLICATE combination to zero, because redundancy
cannot lift a structural zero. It does NOT zero a HETEROGENEOUS combination: its
factor is 1 and the other views carry the result. Both statements are true and
they are about different functions.

Dependence is priced only for replicates, where T8b gives an effective count.
There is no analogue for heterogeneous views yet, so combine_heterogeneous under
assumed independence is an UPPER BOUND and callers should report it as one.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from standing.errors import InvalidInput

__all__ = [
    "combine_views",
    "h2_bernoulli",
    "combine_replicates",
    "combine_heterogeneous",
    "n_eff",
    "saturation_count",
    "independence_inflation",
    "check_unit",
]


def check_unit(value: float, name: str, *, upper: float = 1.0) -> float:
    """Return value as a float, or raise if it is not a usable probability.

    NaN and infinity are rejected here rather than downstream, because both
    survive the arithmetic below and produce output that reads as a result.
    """
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"{name} is not a number: {value!r}") from exc
    if math.isnan(out):
        raise InvalidInput(f"{name} is NaN")
    if math.isinf(out):
        raise InvalidInput(f"{name} is infinite")
    if not (0.0 <= out <= upper):
        raise InvalidInput(f"{name} must lie in [0, {upper}], got {out}")
    return out


def h2_bernoulli(a: float, b: float) -> float:
    """Squared Hellinger between Bernoulli(a) and Bernoulli(b).

    This is the divergence a threshold test induces: `a` and `b` are the two
    answers' probabilities of crossing it. It is the computable instance behind
    every certificate.
    """
    a = check_unit(a, "a")
    b = check_unit(b, "b")
    # Half the sum of squared differences of square roots, which equals
    # 1 - BC exactly and does not cancel: 1 - BC returned a hard 0.0 for rates
    # 0.5 and 0.5 + 1e-9, whose true distance is 5e-19.
    h2 = 0.5 * ((math.sqrt(a) - math.sqrt(b)) ** 2 + (math.sqrt(1.0 - a) - math.sqrt(1.0 - b)) ** 2)
    return _clip_unit(h2)


def n_eff(n: int, rho: float) -> float:
    """Effective number of independent views among n with mean correlation rho.

    T8b. The ceiling is the point of the function: as n grows this tends to
    1/rho, so at rho = 0.9 a hundred correlated views are worth 1.11.

    NEGATIVE rho is legitimate and is accepted. Views that anticorrelate carry
    more joint information than independent ones, so n_eff exceeds n. An earlier
    version rejected it outright, which would have refused a real deployment: the
    wireless per-link correlations run from -0.205 at the tenth percentile to
    +0.266 at the ninetieth, and a mean below zero is perfectly possible in a
    room where movement raises some links while it lowers others.

    The floor is not -1: an exchangeable correlation matrix is only positive
    semi-definite for rho >= -1/(n-1), and at that boundary n_eff diverges. Below
    it there is no such set of views, so it raises.
    """
    n = _check_count(n)
    try:
        rho = float(rho)
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"rho is not a number: {rho!r}") from exc
    if math.isnan(rho) or math.isinf(rho):
        raise InvalidInput(f"rho must be finite, got {rho}")
    if rho > 1.0:
        raise InvalidInput(f"rho must be at most 1, got {rho}")
    if n > 1:
        lower = -1.0 / (n - 1)
        if rho < lower - 1e-12:
            raise InvalidInput(
                f"rho = {rho} is below -1/(n-1) = {lower:.6f} for n = {n}, so no "
                f"exchangeable set of {n} views has that mean correlation: the "
                f"correlation matrix would not be positive semi-definite"
            )
        if abs(1.0 + (n - 1) * rho) <= 1e-12:
            raise InvalidInput(
                f"rho = {rho} sits exactly at -1/(n-1) for n = {n}, where the "
                f"effective count diverges and the combination is unbounded"
            )
    if rho == 0.0:
        return float(n)
    return float(n) / (1.0 + (n - 1) * rho)


def combine_replicates(h: float, n: int, rho: float = 0.0) -> float:
    """T7, with T8b's dependence correction: N replicates of ONE view.

    Args:
        h: the per-view squared Hellinger, identical across replicates by
            hypothesis. Passing a sequence here is an error rather than a
            convenience, because it is the shape of the 2.49x overstatement.
        n: number of replicates.
        rho: mean pairwise correlation; 0 for independent replicates.

    Raises:
        InvalidInput: if h is a sequence, or any argument is out of domain.
    """
    if isinstance(h, (list, tuple, set, frozenset)) or hasattr(h, "__len__"):
        raise InvalidInput(
            "combine_replicates takes ONE per-view value; several different "
            "values are not replicates. Use combine_heterogeneous(hs, dependence=...)."
        )
    h = check_unit(h, "h")
    effective = n_eff(n, rho)
    if h == 0.0:
        return 0.0  # redundancy cannot lift a structural zero
    if h == 1.0:
        return 1.0
    # expm1, not 1 - exp. Below about h = 1e-16 the subtraction cancels to
    # exactly 0.0, so a small but real standing was reported as a STRUCTURAL
    # zero and the rules ABSTAINED where they should have CARRIED. Found by
    # property search, not by inspection: 1 - exp(log1p(-1e-20)) is 0.0.
    return _clip_unit(-math.expm1(effective * math.log1p(-h)))


def combine_heterogeneous(hs: Iterable[float], *, dependence: str) -> float:
    """T7c: views that are not replicates.

    Computed in log space so that many views do not underflow the product, with
    a saturating view handled before the logarithm rather than by it.

    Args:
        dependence: "independent" if you have established that the views fail
            for unrelated reasons, or "unknown" if you have not. The argument is
            REQUIRED and has no default, because the returned number means
            different things in the two cases and looks identical either way.

    Under "unknown" the result is an UPPER BOUND, and usually a loose one, since
    dependence between views can only lower the combination. Ten views sharing a
    single blindness combine to exactly zero while a naive independent product
    manufactures standing out of nothing, which is the 2.49x overstatement a
    second team produced. The previous version of this function documented that
    responsibility in prose and returned a bare float, so nothing recorded
    whether the caller had discharged it.
    """
    if dependence not in ("independent", "unknown"):
        raise InvalidInput(
            f"dependence must be 'independent' or 'unknown', got {dependence!r}. "
            f"There is no default: a combination assuming independence and one "
            f"admitting it does not know are the same number and different "
            f"claims, so the caller has to say which it is."
        )
    values = _as_sequence(hs)
    if not values:
        raise InvalidInput("combine_heterogeneous needs at least one view")
    checked = [check_unit(v, f"hs[{i}]") for i, v in enumerate(values)]
    if any(v == 1.0 for v in checked):
        return 1.0
    total = math.fsum(math.log1p(-v) for v in checked)
    return _clip_unit(-math.expm1(total))


def saturation_count(h: float, eps: float = 0.01) -> float:
    """T7b: the N_eff past which a combined figure is within eps of 1.

    Beyond this the combination reports about 1.0 whatever it is looking at, so
    the per-view certificate is the quantity to report. Returns infinity when
    h = 0, where the combination never moves.
    """
    h = check_unit(h, "h")
    eps = check_unit(eps, "eps")
    if eps <= 0.0:
        raise InvalidInput("eps must be positive")
    if h == 0.0:
        return math.inf
    if h == 1.0:
        return 1.0
    return math.log(eps) / math.log1p(-h)


def independence_inflation(h: float, n: int) -> float:
    """T8: the factor by which assuming independence overstates n duplicates.

    (1 - (1 - h)^n) / h, which is 3.17 for four duplicates at the value we
    measured. Tends to n as h tends to 0.
    """
    h = check_unit(h, "h")
    n = _check_count(n)
    if h == 0.0:
        return float(n)
    return combine_replicates(h, n) / h


def _clip_unit(value: float) -> float:
    """Clamp a divergence into [0, 1], guarding float error only.

    A value outside by more than rounding means the caller computed something
    that is not a squared Hellinger, and silently clipping it would hide that.
    """
    if math.isnan(value):
        raise InvalidInput("computation produced NaN")
    if value < -1e-9 or value > 1.0 + 1e-9:
        raise InvalidInput(f"value {value} is not a squared Hellinger distance")
    return min(1.0, max(0.0, value))


def _check_count(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise InvalidInput(f"count must be an int, got {type(n).__name__}")
    if n < 1:
        raise InvalidInput(f"count must be at least 1, got {n}")
    return n


def _as_sequence(hs: Iterable[float]) -> Sequence[float]:
    if isinstance(hs, (str, bytes)):
        raise InvalidInput("hs must be a sequence of numbers")
    try:
        return list(hs)
    except TypeError as exc:
        raise InvalidInput(
            "combine_heterogeneous takes a sequence of per-view values; for N "
            "replicates of one view use combine_replicates(h, n)."
        ) from exc


def combine_views(results) -> float:
    """Combine independent views by T7c, refusing the combination T3 forbids.

    `combine_heterogeneous` is arithmetic and will happily combine anything. This
    wrapper takes SigmaResult objects and enforces the precondition that makes
    the arithmetic mean something: each view must decide on its OWN statistic.

    Slices of one fused statistic are not independent views. Measured on a
    deployment with two aspects feeding one daily statistic: each aspect alone
    scores 0.0000 while the joint value is 0.4288, so combining the per-aspect
    values gives 0.0000 and misses everything. Sigma is aspect-relative and NOT
    aspect-decomposable (T3), and no combination law repairs that.

    Args:
        results: SigmaResult objects, each carrying a distinct `statistic` name.

    Raises:
        InvalidInput: if any two results share a statistic, or any is unnamed.
    """
    items = list(results)
    if not items:
        raise InvalidInput("combine_views needs at least one view")
    names = []
    for index, result in enumerate(items):
        name = getattr(result, "provenance", {}).get("statistic")
        if not name:
            raise InvalidInput(
                f"view {index} does not name its decision statistic. Pass "
                f"statistic=... to sigma(): combining views is only valid when "
                f"each decides on its own statistic, and an unnamed one cannot "
                f"be shown to."
            )
        names.append(name)
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise InvalidInput(
            f"views share the decision statistic(s) {sorted(duplicates)}, so they "
            f"are slices of one statistic rather than independent views. Sigma is "
            f"not aspect-decomposable (T3): on a real deployment each aspect alone "
            f"scored 0.0000 while the joint value was 0.4288. Compute sigma jointly "
            f"instead of combining."
        )
    # combine_views has already refused any two results sharing a statistic,
    # which is what independence means operationally here.
    return combine_heterogeneous([r.value for r in items], dependence="independent")
