"""The answers a claim ranges over, built as a quotient rather than written as a list.

Standing is a property of a view relative to the answers still open, so the answer
set is a first-class object here and never an implicit pair. Two consequences are
load-bearing:

    sigma is a MINIMUM over pairs. A view that separates `absent` from `present`
    beautifully still has sigma zero if it cannot separate `present` from
    `context_confounded`, because a claim is not entitled to ignore the answer it
    cannot rule out.

    sigma is therefore ANTITONE in the set. Removing an answer removes the pairs
    that contain it, so the minimum can only rise. That is Recruit: an anchor
    whose own claim rules an answer out raises standing with no change in the data.

Answers are REGIONS, not points. A minimum over a continuum of point-answers is
identically zero (T9), so `from_actions` groups the states your model can describe
by the action they demand and makes each group one answer. The groups are the
answers. That construction is the reason the criterion is not vacuous, and it is
why this module asks for an action map rather than for a list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Hashable, Iterable, Iterator

from standing.errors import AnswerSetError
from standing.model import canonical_repr

__all__ = ["Answer", "AnswerSet", "QuotientOrigin"]


_canon = canonical_repr  # one rendering for keys, fingerprints and records


def _fingerprint(pairs: Iterable[tuple[Hashable, Hashable]]) -> str:
    """A short digest of a state -> action map, stable across runs.

    It fingerprints the MAP rather than the function that produced it, which is
    the part a third party can act on: two records naming the same digest
    grouped their states the same way, whatever code did the grouping. Fields
    are NUL separated for the same reason the pair seeds are, so that ('a', 'bc')
    and ('ab', 'c') cannot collide.
    """
    digest = hashlib.blake2b(digest_size=8)
    for state, action in sorted((_canon(s), _canon(a)) for s, a in pairs):
        digest.update(state.encode())
        digest.update(b"\0")
        digest.update(action.encode())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class QuotientOrigin:
    """Where an answer set came from, carried so that a record can say.

    The action quotient is the one condition the library cannot check against
    the world, and it used to be the only one leaving no trace either: a set
    derived from a policy and a set typed by hand were indistinguishable
    everywhere downstream. They are different objects and a consumer auditing
    from outside is entitled to know which one it is holding.

    Attributes:
        kind: "derived" when `from_actions` built it from an action map,
            "declared" when the answers were written out by hand.
        states: how many states were grouped; None when declared.
        action: the fingerprint of the map, None when declared. It describes
            the map the set was BUILT from, so it does not change when the set
            is later narrowed by a recruitment.
    """

    kind: str
    states: int | None = None
    action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "states": self.states, "action": self.action}


@dataclass(frozen=True)
class Answer:
    """One answer still open, named, with the state or region it stands for.

    Attributes:
        name: identifier used in records and messages.
        state: whatever your forward model needs in order to predict under this
            answer. Two answers whose `state` compares equal are synonyms under
            two names, which is a defect the set refuses to carry silently.
    """

    name: str
    state: Hashable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AnswerSetError("an answer needs a non-empty name")


@dataclass(frozen=True)
class AnswerSet:
    """The answers still open for one claim.

    Construction refuses two things outright. A set with fewer than two answers
    is not a question, and returning zero for it would say "nothing could be told
    apart" when the truth is "nothing was asked". A set holding the same state
    under two names drags the minimum to zero and refuses every claim: measured
    at 0.1534 -> 0.0000 in a deployment before it was caught.
    """

    answers: tuple[Answer, ...]
    # Excluded from comparison deliberately. Two sets holding the same answers
    # ARE the same question whichever way they were built, and making equality
    # depend on provenance would break every comparison in the tests and, worse,
    # would let a caller "fix" a disagreement by rebuilding the set.
    origin: QuotientOrigin | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.origin is None:
            # Absent evidence of derivation, the set WAS declared. This is not
            # the "None means not declared" convention the record uses: there
            # the fact is unknown, here it is known by construction, because
            # anything built through `from_actions` says so.
            object.__setattr__(self, "origin", QuotientOrigin(kind="declared"))
        elif not isinstance(self.origin, QuotientOrigin):
            raise AnswerSetError(
                f"origin must be a QuotientOrigin, got {type(self.origin).__name__}"
            )
        if len(self.answers) < 2:
            raise AnswerSetError(
                f"standing needs at least two open answers, got "
                f"{len(self.answers)}: with one answer nothing was asked, which "
                f"is not the same as nothing being distinguishable"
            )
        names = [a.name for a in self.answers]
        if len(set(names)) != len(names):
            raise AnswerSetError(f"answer names must be distinct, got {names}")
        seen: dict[Hashable, str] = {}
        for answer in self.answers:
            try:
                previous = seen.get(answer.state)
            except TypeError as exc:
                raise AnswerSetError(
                    f"answer {answer.name!r} has an unhashable state "
                    f"({type(answer.state).__name__}). States are compared to "
                    f"catch synonyms, so they must be hashable: use a tuple of "
                    f"sorted items, or a frozenset, rather than a dict or list."
                ) from exc
            if previous is not None:
                raise AnswerSetError(
                    f"{answer.name!r} and {previous!r} stand for the same state, "
                    f"so they are one answer under two names. A synonym forces "
                    f"sigma to zero and refuses every claim. Merge them, or give "
                    f"them the distinct states you meant."
                )
            seen[answer.state] = answer.name

    @classmethod
    def from_actions(
        cls,
        states: Iterable[Hashable],
        action_of: Callable[[Hashable], Hashable],
        *,
        name_of: Callable[[Hashable], str] | None = None,
    ) -> "AnswerSet":
        """Build the quotient: group states by the action they demand.

        This is the construction, and using it rather than writing a list is what
        keeps answers regions rather than points. Each answer's `state` is the
        frozenset of the states that demand its action, so the forward model can
        integrate over the region.

        Args:
            states: every state your model can describe.
            action_of: the action a consumer would take in that state.
            name_of: how to name the answer for an action; defaults to `str`.
        """
        grouped: dict[Hashable, list[Hashable]] = {}
        seen: list[tuple[Hashable, Hashable]] = []
        for state in states:
            action = action_of(state)
            grouped.setdefault(action, []).append(state)
            seen.append((state, action))
        if not grouped:
            raise AnswerSetError("no states were supplied, so no answers exist")
        label = name_of or (lambda action: str(action))
        return cls(
            tuple(
                Answer(name=label(action), state=frozenset(members))
                for action, members in grouped.items()
            ),
            origin=QuotientOrigin(
                kind="derived", states=len(seen), action=_fingerprint(seen)
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.answers)

    def pairs(self) -> Iterator[tuple[Answer, Answer]]:
        """Every unordered pair of distinct open answers."""
        yield from combinations(self.answers, 2)

    def narrowed(self, *, ruled_out: str) -> "AnswerSet":
        """This set with one answer ruled out, which is what Recruit does.

        Raises:
            AnswerSetError: if the answer is not open, or if removing it would
                leave fewer than two.
        """
        if ruled_out not in self.names:
            raise AnswerSetError(f"{ruled_out!r} is not open; open are {self.names}")
        remaining = tuple(a for a in self.answers if a.name != ruled_out)
        if len(remaining) < 2:
            raise AnswerSetError(
                "cannot narrow below two answers: with one answer left there is "
                "nothing to tell apart and sigma is undefined, not infinite"
            )
        # The origin travels with the narrowed set. A recruitment removes an
        # answer from a quotient; it does not build a different quotient, and a
        # record of the narrowed claim should still name the map it came from.
        return AnswerSet(remaining, origin=self.origin)

    def __len__(self) -> int:
        return len(self.answers)
