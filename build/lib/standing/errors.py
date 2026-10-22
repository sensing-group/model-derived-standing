"""Every way this library refuses to answer.

A library that returns a float for degenerate input is worse than no library,
because the caller gets a plausible number and no signal. Every function here
validates at its boundary and raises. Nothing propagates a NaN.

The taxonomy is deliberately small. `InvalidInput` covers anything numerically
unusable, `AnswerSetError` anything wrong with the set of answers, and
`CertificateError` anything that would make a certificate a fit rather than a
witness.
"""

from __future__ import annotations

__all__ = [
    "StandingError",
    "InvalidInput",
    "AnswerSetError",
    "CertificateError",
]


class StandingError(Exception):
    """Base for everything this library raises."""


class InvalidInput(StandingError, ValueError):
    """A value outside the domain a formula is defined on.

    Includes NaN and infinity, which are the dangerous cases: they flow through
    arithmetic without complaint and emerge as a number that looks like a
    measurement. A second implementation produced a plausible certificate from a
    NaN and did not notice for hours.
    """


class AnswerSetError(StandingError, ValueError):
    """The set of open answers cannot support a standing computation.

    Fewer than two answers means nothing was asked, and returning zero for that
    case says "nothing could be told apart" when the truth is "nothing was
    asked". The two must not share an output.
    """


class CertificateError(StandingError, ValueError):
    """A certificate was requested in a way that would make it a fit.

    Chiefly: selecting a statistic on the same sample it is scored on. That is
    not a weaker certificate, it is not one at all, and it reported 0.9896 on a
    null where both sides were the same answer.
    """
