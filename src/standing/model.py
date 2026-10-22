"""What you supply: the predictive law of your own decision statistic.

This is the part no library can provide, and the part most expensive to get
wrong. The model must predict the statistic your pipeline ACTUALLY decides on,
not the raw sensor values and not the statistic you wish it used. If the pipeline
decides on a maximum over a sliding window, the model simulates that. We once
simulated a linear ramp where the deployment used a sigmoid, and every number for
two days was on the wrong curve.

The signature carries T4 structurally: `law` receives the answer and the
conditions, and there is nowhere to pass a reading. That is what lets standing
exist for an interval in which nothing was recorded.

`Conditions` is a mapping of the things that decide what a view could have seen:
which views were in a position to report, with what weight, which were filled in.
Values do not belong here. A field whose name looks like a reading is rejected on
construction, which is a lint and not a proof, but it catches the common slip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Protocol, runtime_checkable

import numpy as np

from standing.divergence import hellinger2_gaussian
from standing.errors import InvalidInput

__all__ = ["Conditions", "Law", "GaussianLaw", "ForwardModel", "canonical_repr"]


def canonical_repr(value: object) -> str:
    """A representation that does not depend on hash randomisation.

    `repr` of a set or frozenset lists members in hash order, which differs
    between processes for strings, so a key or fingerprint built from it
    changed on every run. Members are rendered in sorted order instead,
    recursively, and mappings by sorted key.
    """
    if isinstance(value, (set, frozenset)):
        return "{" + ", ".join(sorted(canonical_repr(v) for v in value)) + "}"
    if isinstance(value, (tuple, list)):
        inner = ", ".join(canonical_repr(v) for v in value)
        return ("(" + inner + ")") if isinstance(value, tuple) else ("[" + inner + "]")
    if isinstance(value, Mapping):
        items = sorted((canonical_repr(k), canonical_repr(v)) for k, v in value.items())
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    return repr(value)

_READING_WORDS = ("reading", "value", "measurement", "observation", "sample")
_PLAIN = (str, int, float, bool, type(None))


def _is_plain(value: object) -> bool:
    """Whether a condition value has a representation that survives a machine."""
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, (tuple, list, frozenset, set)):
        return all(_is_plain(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(k, _PLAIN) and _is_plain(v) for k, v in value.items()
        )
    return False


@dataclass(frozen=True)
class Conditions:
    """What was true about the observation, never what was observed.

    Attributes:
        fields: names and values describing the conditions of the interval:
            liveness, weights, fill flags, geometry, time since a reference
            rebuild. Anything the forward model needs that is not a reading.
    """

    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping):
            raise InvalidInput("Conditions.fields must be a mapping")
        for key, value in self.fields.items():
            if not _is_plain(value):
                raise InvalidInput(
                    f"field {key!r} holds a {type(value).__name__}, which has no "
                    f"stable representation. Conditions go into a record that "
                    f"must re-derive the same key on another machine, and the "
                    f"default repr of an object embeds its memory address, so "
                    f"two identical runs would disagree. Use plain data: numbers, "
                    f"strings, booleans, None, or tuples of those."
                )
            lowered = str(key).lower()
            if any(word in lowered for word in _READING_WORDS):
                raise InvalidInput(
                    f"field {key!r} looks like a reading. Standing is derived "
                    f"from the model and the conditions and never from the "
                    f"values, which is what lets it exist for an interval where "
                    f"nothing was recorded. Rename the field if it really is a "
                    f"condition."
                )
        object.__setattr__(self, "fields", dict(self.fields))

    def get(self, key: str, default: object = None) -> object:
        return self.fields.get(key, default)

    def key(self) -> Hashable:
        """A stable hashable summary, for records and caching."""
        return tuple(sorted((str(k), canonical_repr(v)) for k, v in self.fields.items()))


@runtime_checkable
class Law(Protocol):
    """The predicted distribution of the decision statistic under one answer."""

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `size` values of the decision statistic."""


@dataclass(frozen=True)
class GaussianLaw:
    """A normal law, with an exact divergence so tests need no sampling.

    Supplied because a closed form is the only honest way to check a sampler,
    and because many pipelines really do decide on an approximately normal
    statistic.
    """

    mean: float
    sd: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.sd):
            raise InvalidInput("a Gaussian law needs finite parameters")
        if self.sd < 0.0:
            raise InvalidInput("standard deviation must be non-negative")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if self.sd == 0.0:
            return np.full(int(size), float(self.mean))
        return rng.normal(self.mean, self.sd, int(size))

    def hellinger2_against(self, other: "GaussianLaw") -> float:
        return hellinger2_gaussian(self.mean, self.sd, other.mean, other.sd)


@runtime_checkable
class ForwardModel(Protocol):
    """Your model: what the decision statistic does under each answer.

    There is deliberately no reading parameter. If you find yourself wanting one,
    what you are computing is a posterior and not standing.
    """

    def law(self, answer, conditions: Conditions) -> Law:
        """The predicted law of the decision statistic under this answer."""

    # Optional, and worth having. A model that also defines
    #
    #     def states(self) -> Sequence[Hashable]
    #
    # owns its vocabulary: every state of the world its law can describe, which
    # is the domain of the law's own arguments. `audit` reads it in place of a
    # list the caller typed, because a list a caller typed is the one place a
    # sigma can be quietly inflated: leaving a state out removes a candidate
    # from a minimum, so it can only raise sigma, and nothing inside sigma can
    # tell. A model that says what it can describe closes that door.
