"""Certificates: admitting a claim without asking anyone to trust the model.

Any statistic you can write induces a lower bound on sigma, because a statistic
is a deterministic kernel and processing cannot add standing (T6 read backwards).
So scanning a threshold over the decision statistic and taking the separation it
achieves gives a value that sigma cannot be below, and the threshold that attains
it is an explicit rule: a third party can apply it to fresh data using no part of
our machinery.

THE HYPOTHESIS IS NOT OPTIONAL. The statistic must be fixed independently of the
sample it is scored on. Maximising over a sample maximises over its noise: a
deployment followed the procedure literally and obtained 0.189 per view and
0.9896 combined on a null where BOTH SIDES WERE THE SAME ANSWER, which is
near-perfect standing between an answer and itself. This module therefore takes
two samples per answer, one to choose the threshold and one to score it, and
refuses to do anything else.

THE SOURCE IS NOT OPTIONAL EITHER. A certificate scored on samples drawn from
the forward model inherits every misspecification of that model: if the law
overstates sigma 8.66 times, so does the certificate, to within threshold-scan
looseness. The only run that ever issued a certificate did exactly this, and
the deck called it "held-out data". So every certificate now says what it was
scored on, `readings` or `model`, and only a certificate on readings can admit.
A model-sourced certificate is a consistency check of the library against its
own law; it is kept, recorded and never allowed to admit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from standing.answers import AnswerSet
from standing.errors import CertificateError, InvalidInput
from standing.numerics import h2_bernoulli

__all__ = ["Certificate", "certify_pair", "certify", "split", "SOURCES"]

SOURCES = ("readings", "model")
"""What a certificate may have been scored on. Only `readings` can admit."""


def _check_source(source: object) -> str:
    if not isinstance(source, str) or source not in SOURCES:
        raise InvalidInput(
            f"source must be one of {SOURCES}, got {source!r}. A certificate has "
            f"to say whether it was scored on readings or on the model's own "
            f"samples, because only the first can admit a claim: a certificate "
            f"on model samples is as wrong as the model is."
        )
    return source


@dataclass(frozen=True)
class Certificate:
    """An explicit rule that separates two answers, and the rate it achieves.

    Attributes:
        pair: the two answers separated.
        threshold: the rule is "statistic >= threshold".
        value: squared Hellinger between the two crossing rates, ON THE SCORING
            SAMPLE. This is a lower bound on sigma for this pair.
        rate_left, rate_right: the crossing probabilities that produced it.
        n_select, n_score: sample sizes, so the bound's precision is visible.
        source: `readings` when both samples are recorded data, `model` when
            they were drawn from the forward model. Only `readings` can admit.
    """

    pair: tuple[str, str]
    threshold: float
    value: float
    rate_left: float
    rate_right: float
    n_select: int
    n_score: int
    source: str = "model"

    def __post_init__(self) -> None:
        _check_source(self.source)

    @property
    def measured(self) -> bool:
        """Whether this certificate was scored on readings rather than the model."""
        return self.source == "readings"

    def decide(self, statistic) -> np.ndarray:
        """Apply the rule to fresh data. This is what makes it a certificate.

        Non-finite input raises rather than comparing. A NaN compares False
        against any threshold, so a third party auditing dirty data would get a
        quiet "below threshold" for every unusable sample and no indication that
        anything was wrong.
        """
        values = np.asarray(statistic, dtype=float)
        if not np.all(np.isfinite(values)):
            bad = int(np.count_nonzero(~np.isfinite(values)))
            raise InvalidInput(
                f"{bad} non-finite value(s) in the data this rule was applied to. "
                f"NaN compares False against any threshold, so the verdict would "
                f"read as 'below' rather than as 'unusable'."
            )
        return values >= self.threshold


def split(sample, rng: np.random.Generator | int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and halve a sample into a selection half and a scoring half."""
    array = np.asarray(sample, dtype=float).ravel()
    if array.size < 4:
        raise InvalidInput(
            f"need at least 4 draws to split into selection and scoring halves, got {array.size}"
        )
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    shuffled = generator.permutation(array)
    middle = shuffled.size // 2
    return shuffled[:middle], shuffled[middle:]


def _rates(sample: np.ndarray, threshold: float) -> float:
    return float(np.count_nonzero(sample >= threshold)) / float(sample.size)


def _check_disjoint(select: np.ndarray, score: np.ndarray, label: str) -> None:
    """Refuse the one mistake that turns a certificate into a fit."""
    if select.size == 0 or score.size == 0:
        raise InvalidInput(f"{label}: both halves must be non-empty")
    if np.shares_memory(select, score):
        raise CertificateError(
            f"{label}: the selection and scoring samples overlap in memory, so "
            f"the threshold would be scored partly on the data that chose it. "
            f"Split first: split() returns two disjoint copies."
        )
    if select.shape == score.shape and np.array_equal(select, score):
        raise CertificateError(
            f"{label}: the selection and scoring samples are identical. A "
            f"threshold chosen and scored on one sample maximises over its "
            f"noise and is a fit, not a certificate: that is how a null where "
            f"both sides were the same answer scored 0.9896. Use split(), or "
            f"pass genuinely separate data."
        )


def certify_pair(
    name_left: str,
    select_left,
    score_left,
    name_right: str,
    select_right,
    score_right,
    *,
    source: str,
    grid: int = 200,
) -> Certificate:
    """Choose a threshold on one sample, score it on another.

    Args:
        select_*: the halves used to CHOOSE the threshold.
        score_*: the halves used to SCORE it. Must not be the selection halves.
        source: `readings` or `model`, what the samples are. Required, because
            the two look identical to this function and mean opposite things
            to `decide`.
        grid: how many candidate thresholds to scan over the union support.
    """
    source = _check_source(source)
    sl = _finite(select_left, "select_left")
    sr = _finite(select_right, "select_right")
    zl = _finite(score_left, "score_left")
    zr = _finite(score_right, "score_right")
    _check_disjoint(sl, zl, name_left)
    _check_disjoint(sr, zr, name_right)
    if grid < 2:
        raise InvalidInput("grid must be at least 2")

    low = float(min(sl.min(), sr.min()))
    high = float(max(sl.max(), sr.max()))
    if high - low <= 0.0:
        candidates = np.array([low])
    else:
        candidates = np.linspace(low, high, int(grid))

    # Choose on the selection halves only. The scoring halves are not consulted
    # until the threshold is fixed.
    best_threshold = float(candidates[0])
    best_selected = -1.0
    for threshold in candidates:
        value = h2_bernoulli(_rates(sl, threshold), _rates(sr, threshold))
        if value > best_selected:
            best_selected, best_threshold = value, float(threshold)

    rate_left = _rates(zl, best_threshold)
    rate_right = _rates(zr, best_threshold)
    return Certificate(
        pair=(name_left, name_right),
        threshold=best_threshold,
        value=h2_bernoulli(rate_left, rate_right),
        rate_left=rate_left,
        rate_right=rate_right,
        n_select=int(sl.size + sr.size),
        n_score=int(zl.size + zr.size),
        source=source,
    )


def certify(
    answers: AnswerSet,
    selection: dict[str, np.ndarray],
    scoring: dict[str, np.ndarray],
    *,
    source: str,
    grid: int = 200,
) -> tuple[float, dict[tuple[str, str], Certificate]]:
    """Certify every pair, and return the minimum as a lower bound on sigma.

    The minimum is the bound because sigma is itself a minimum over pairs: a
    claim is only as entitled as its worst confusion.

    `source` says what the samples are and is required: `readings` for
    recorded data, `model` for draws from the forward model. See the module
    docstring for why the second cannot admit.
    """
    source = _check_source(source)
    missing = [n for n in answers.names if n not in selection or n not in scoring]
    if missing:
        raise InvalidInput(f"no samples supplied for {missing}")
    certificates: dict[tuple[str, str], Certificate] = {}
    for left, right in answers.pairs():
        certificates[(left.name, right.name)] = certify_pair(
            left.name,
            selection[left.name],
            scoring[left.name],
            right.name,
            selection[right.name],
            scoring[right.name],
            source=source,
            grid=grid,
        )
    floor = min(c.value for c in certificates.values())
    return floor, certificates


def _finite(sample, label: str) -> np.ndarray:
    array = np.asarray(sample, dtype=float).ravel()
    if array.size == 0:
        raise InvalidInput(f"{label} is empty")
    if not np.all(np.isfinite(array)):
        raise InvalidInput(
            f"{label} holds non-finite values; a certificate computed over NaN "
            f"reads as a rule a third party could apply, and it is not one"
        )
    return array
