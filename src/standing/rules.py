"""The five dispositions, as the only things a consumer may do with a claim.

    ADMISSIBLE  sigma clears the floor on the model alone. The claim may be
                carried and may not be relied on: no reading has exhibited the
                separation the laws predict. This is what the README always
                said, "unrefused means not-yet-refused, never entitled", and
                what the code did not do until this disposition existed.
    ADMITTED    sigma clears the floor AND a certificate (an explicit rule
                scored on held-out readings, see `certify`) clears it too. The
                claim may be used as evidence, and a third party can check why.
    RECRUIT   standing was below the floor, an anchor with standing of its own
              ruled an answer out, and with that answer gone it clears. NOTHING
              ABOUT THE DATA CHANGED; the question narrowed (T2).
    CARRY     identifiable but not here. The claim travels downstream WITH its
              standing rather than being deleted, so a later stage that narrows
              the set can still use it.
    ABSTAIN   sigma is exactly zero. Nothing that is a function of the readings
              separates the answers, at any sample size (T1).

Keeping CARRY and ABSTAIN distinct is the point. Abstention at a threshold is a
policy you must defend and a reviewer may always ask why the floor sits where it
does. Abstention at exactly zero is a theorem you need not defend.

RECRUIT enforces the recursion that keeps this honest: an anchor may narrow the
set only if the anchor's own claim has standing. A claim without standing cannot
lend standing to another claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Mapping, Sequence

from standing.answers import AnswerSet
from standing.certify import Certificate
from standing.drift import Bracket
from standing.errors import CertificateError, StandingError
from standing.numerics import check_unit
from standing.sigma import SigmaResult

__all__ = ["Rule", "Anchor", "Ruling", "decide", "certificate_bound", "unmeasured_pairs"]


class Rule(str, Enum):
    ADMISSIBLE = "admissible"
    ADMITTED = "admitted"
    RECRUIT = "recruit"
    CARRY = "carry"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class _Established:
    """The standing of a claim that was already ruled on.

    Not a SigmaResult: that would claim a pair table this does not have. It
    exists so `Anchor.from_ruling` never has to hand a bare number to the
    constructor, which the constructor refuses for good reason.
    """

    value: float
    answers: tuple[str, ...]


@dataclass(frozen=True)
class Anchor:
    """A claim from elsewhere that can rule an answer out.

    `standing` is a SigmaResult or a drift Bracket, NEVER a bare number, and that
    is the whole point of this class. RECRUIT is the disposition nothing else in
    the library can do: it admits a claim that failed its own floor, on the
    strength of another claim narrowing the question. If the other claim's
    standing can be asserted rather than exhibited, the recursion has no base
    case and the mechanism is decorative.

    It was decorative. A hand-typed `sigma=0.99`, with no model, no data and no
    provenance, turned a CARRY at 0.0371 into an admitted RECRUIT at 0.4539, and
    the resulting Record passed `verify` with nothing recording where 0.99 came
    from. Requiring the object means the anchor's own standing was computed the
    same way as everything else, and the ruling can say so.

    If your anchor's standing crossed a process boundary as a Record, do not
    rebuild a SigmaResult by hand: pass the drift Bracket the record carries, or
    re-score the anchor here. A reconstructed result is a typed float again.
    """

    name: str
    rules_out: str
    standing: "SigmaResult | Bracket"
    floor: float
    rank: int = 0
    asserts: str | None = None
    """Which answer, in the ANCHOR's own set, this anchor is asserting.

    Ruling an answer out is an assertion, so it is governed by the per-answer
    argument rather than by sigma against a floor. Supply it together with
    `forbidden` and the constructor will check them against each other.
    """
    forbidden: frozenset = frozenset()
    """The answers the anchor's view may NOT assert, from an audit of its states.

    This is the object that should cross a module boundary in place of a scalar.
    Leave it empty only when nothing was audited, and know that an empty set is
    the absence of a check rather than a clean bill of health.
    """

    @classmethod
    def from_ruling(
        cls, name: str, rules_out: str, ruling: "Ruling", floor: float | None = None
    ) -> "Anchor":
        """Build an anchor from the ruling that established it, carrying its rank.

        This is the honest path and it is the one that makes a cycle
        unconstructible: to build it you must already hold the ruling, so two
        claims cannot each be recruited on the other.
        """
        if not isinstance(ruling, Ruling):
            raise StandingError(
                f"from_ruling needs a Ruling, got {type(ruling).__name__}"
            )
        if not ruling.admitted:
            raise StandingError(
                f"{name!r} was not admitted ({ruling.rule.value}), so it cannot "
                f"lend standing. A claim without standing cannot lend standing."
            )
        value = ruling.sigma_after if ruling.sigma_after is not None else ruling.sigma
        return cls(
            name=name,
            rules_out=rules_out,
            standing=_Established(float(value), ruling.answers),
            floor=ruling.floor if floor is None else floor,
            rank=ruling.rank,
        )

    def __post_init__(self) -> None:
        if isinstance(self.standing, (int, float)) and not isinstance(self.standing, bool):
            raise StandingError(
                f"anchor {self.name!r} was given a bare number ({self.standing}) "
                f"as its standing. An anchor lends its entitlement to another "
                f"claim, so it must exhibit that entitlement: pass the "
                f"SigmaResult from scoring the anchor, or the drift Bracket it "
                f"was bounded by. A claim without standing cannot lend standing."
            )
        if not isinstance(self.standing, (SigmaResult, Bracket, _Established)):
            raise StandingError(
                f"anchor {self.name!r} needs a SigmaResult or a drift Bracket as "
                f"its standing, got {type(self.standing).__name__}"
            )
        check_unit(self.floor, f"floor of anchor {self.name!r}")
        if self.forbidden and not isinstance(self.forbidden, (frozenset, set)):
            raise StandingError(
                f"anchor {self.name!r}: forbidden must be a set of answer names, "
                f"got {type(self.forbidden).__name__}"
            )
        if self.forbidden and self.asserts is None:
            raise StandingError(
                f"anchor {self.name!r} carries a forbidden set but does not say "
                f"which answer it asserts, so the two cannot be checked against "
                f"each other. Set `asserts`, or drop `forbidden` and accept that "
                f"nothing was checked."
            )
        if self.asserts is not None and self.asserts in self.forbidden:
            raise StandingError(
                f"anchor {self.name!r} asserts {self.asserts!r}, which its own "
                f"audit lists as forbidden. Ruling an answer out is an assertion, "
                f"so a view that may not assert its conclusion cannot lend "
                f"standing on the strength of it, whatever its sigma reads."
            )
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise StandingError(
                f"anchor {self.name!r} needs a non-negative integer rank, got "
                f"{self.rank!r}. Rank 0 ASSERTS that this anchor's standing was "
                f"established without recruiting anything, and is a claim about "
                f"provenance. Use Anchor.from_ruling to carry it honestly."
            )

    @property
    def sigma(self) -> float:
        """The value this anchor may be judged on.

        A Bracket contributes its LOW end, for the same reason a claim does: it
        is the value the anchor may be relied on for under drifting conditions.
        """
        if isinstance(self.standing, Bracket):
            return self.standing.low
        return self.standing.value

    @property
    def has_standing(self) -> bool:
        """Whether this anchor may lend its entitlement.

        NOT simply sigma against a floor. Ruling an answer out is an ASSERTION,
        and an assertion is governed by the per-answer argument: an answer every
        one of whose member states has an impostor outside it cannot be asserted
        by any rule, at any prior, whatever the answer-level sigma reads.

        Measured, the case this exists to refuse: a view whose asserted answer is
        the even mixture of its own two rivals scores 0.2889 at the answer level,
        clears a floor of 0.25, and is provably unassertable. RECRUIT used to
        fire on it.
        """
        if self.asserts is not None and self.asserts in self.forbidden:
            return False
        # Exactly zero is ABSTAIN, and no floor may override a theorem. sigma is
        # non-negative by construction, so `sigma >= floor` is UNCONDITIONALLY
        # TRUE at floor 0: an anchor whose own view could separate nothing at all
        # lent its standing, and a claim at 0.0003 was admitted at 0.6753 on the
        # strength of it. The module docstring above says a claim without
        # standing cannot lend standing; this is the line that makes it so.
        if self.sigma <= 0.0:
            return False
        return self.sigma >= self.floor


@dataclass(frozen=True)
class Ruling:
    """What was decided, and everything needed to argue with it."""

    rule: Rule
    admitted: bool
    sigma: float
    floor: float
    answers: tuple[str, ...]
    binding_pair: tuple[str, str] | None
    sigma_after: float | None = None
    recruited: str | None = None
    ruled_out: str | None = None
    exact: bool = False
    """Whether sigma was computed in closed form rather than estimated.

    This used to be called `certified`, and the name was a lie the code told
    itself: it marked the EXACTNESS of a virtual number, computed from laws
    nobody measured, and a reader took it for a data certificate. A certificate
    is a rule exhibited on held-out readings (`certify`), and it lives in its
    own field. Exactness of a prediction is not evidence for it.
    """
    certificate: float | None = None
    """The certified lower bound on sigma, when certificates were supplied.

    Present on an ADMITTED ruling, where it clears the floor, and on an
    ADMISSIBLE ruling that was offered certificates which did not, so the record
    shows the data was consulted and what it said. None means no certificate was
    offered, which is the absence of a check and not a clean bill of health.
    """
    reason: str = ""
    rank: int = 0
    """How many recruitments deep this ruling is.

    0 means nothing was recruited. A RECRUIT ruling is one deeper than the
    deepest anchor it used, which is what makes the justification well founded:
    rank strictly increases along a chain, so a cycle would need a rank greater
    than itself and cannot be built.
    """


def _value_of(
    standing, answers: AnswerSet | None
) -> tuple[float, bool, tuple[str, str] | None, tuple[str, ...]]:
    """Accept a SigmaResult or a drift Bracket, and refuse anything else.

    A Bracket contributes its LOW end, because that is the value a claim may be
    based on under drifting conditions. It carries no answer set of its own, so
    one must be supplied: a sigma without the set it was scored against cannot be
    read, and a record built from an answer-less ruling fails its own verifier.
    That was a real defect, and it sat on the drift path, which is the path every
    deployment with moving conditions has to use.

    A Bracket is also NOT reported as exact. It bounds the true value using a
    null; it does not exhibit a rule anyone can apply. Only an exact sigma or a
    certificate does that.
    """
    if isinstance(standing, SigmaResult):
        return standing.value, standing.exact, standing.binding_pair, standing.answers
    if isinstance(standing, Bracket):
        if answers is None:
            raise StandingError(
                "a drift Bracket carries no answer set, so pass answers= to "
                "decide(). Without it the ruling cannot name what it was scored "
                "against, and the record built from it will not verify."
            )
        return standing.low, False, None, answers.names
    raise StandingError(
        f"decide() takes a SigmaResult or a drift Bracket, got "
        f"{type(standing).__name__}. A weight is not standing: none of T1, T2, "
        f"T6 or T10 applies to it, so it cannot support a disposition."
    )


def certificate_bound(
    certificates: Mapping[tuple[str, str], Certificate], names: Sequence[str]
) -> float:
    """The certified lower bound on sigma over EVERY open pair, or refuse.

    Sigma is a minimum over pairs, so a certificate that skips a pair certifies
    nothing about the set: the pair it skipped may be the binding one. Each
    entry must be a Certificate from `certify`, which is the object that can be
    applied to fresh data; a number typed in its place is a fit, not a rule.
    """
    if not isinstance(certificates, Mapping) or not certificates:
        raise CertificateError(
            "certificates must be the mapping certify() returns, pair -> "
            "Certificate. An empty or absent mapping is not a certificate of "
            "anything: leave the argument out and the claim stays admissible."
        )
    values: list[float] = []
    missing: list[tuple[str, str]] = []
    for left, right in combinations(names, 2):
        cert = certificates.get((left, right))
        if cert is None:
            cert = certificates.get((right, left))
        if cert is None:
            missing.append((left, right))
            continue
        if not isinstance(cert, Certificate):
            raise CertificateError(
                f"the certificate for {(left, right)} is a "
                f"{type(cert).__name__}, not a Certificate from certify(). A "
                f"certificate is a rule a stranger can apply to fresh readings; "
                f"a value typed by hand is neither."
            )
        if not isinstance(getattr(cert, "source", None), str):
            raise CertificateError(
                f"the certificate for {(left, right)} does not say what it was "
                f"scored on. A certificate on the model's own samples is as wrong "
                f"as the model, so a certificate without a source cannot be read."
            )
        values.append(check_unit(float(cert.value), f"certificate for {(left, right)}"))
    if missing:
        raise CertificateError(
            f"no certificate for {missing}. Sigma is a minimum over pairs, so a "
            f"certificate that skips a pair certifies nothing about the set: "
            f"the pair it skipped may be the one that binds."
        )
    return min(values)


def unmeasured_pairs(
    certificates: Mapping[tuple[str, str], Certificate], names: Sequence[str]
) -> list[tuple[str, str]]:
    """The open pairs whose certificate was scored on the model, not on readings.

    Such a certificate is a consistency check of the library against its own
    law. It is kept and recorded, and it cannot admit: it inherits whatever the
    law got wrong, which was 8.66 times on the one occasion it was measured.

    Reads exactly the pairs `certificate_bound` reads, over `names` and in
    either key order, so a mapping built for a wider set and reused after
    narrowing is judged on the open pairs only. A missing pair is not this
    function's complaint; `certificate_bound` refuses it first.
    """
    if not isinstance(certificates, Mapping):
        raise CertificateError("certificates must be the mapping certify() returns")
    out: list[tuple[str, str]] = []
    for left, right in combinations(names, 2):
        cert = certificates.get((left, right))
        if cert is None:
            cert = certificates.get((right, left))
        if cert is None:
            continue
        if not isinstance(cert, Certificate):
            raise CertificateError(
                f"the certificate for {(left, right)} is a {type(cert).__name__}, "
                f"not a Certificate from certify(), so it cannot say what it was "
                f"scored on"
            )
        if not cert.measured:
            out.append((left, right))
    return out


def decide(
    standing,
    floor: float,
    *,
    answers: AnswerSet | None = None,
    anchors: Sequence[Anchor] = (),
    recompute=None,
    max_rank: int | None = None,
    certificates: Mapping[tuple[str, str], Certificate] | None = None,
) -> Ruling:
    """Apply the five rules.

    Args:
        standing: a SigmaResult, or a Bracket when conditions drifted.
        floor: the threshold for admission. Yours to set and to defend.
        certificates: the mapping `certify` returns, one Certificate per open
            pair, scored on readings the threshold was not chosen on. Without
            it a claim that clears the floor is ADMISSIBLE and nothing more.
            With it, and with every pair's certificate clearing the floor, the
            claim is ADMITTED. A certificate below the floor leaves the claim
            admissible and is recorded, so the ruling shows the data was asked.
        answers: the open set, needed only if recruitment may happen.
        anchors: claims that could narrow the set.
        recompute: callable(AnswerSet) -> SigmaResult, used after an anchor rules
            an answer out. Required for RECRUIT, since narrowing changes sigma.
        max_rank: refuse to recruit on an anchor deeper than this. A long chain
            is not unsound by itself, but its coverage decays with length, so a
            deployment that cares about how far a verdict is from a grounded
            observation sets this and gets told rather than discovering it.

    WELL-FOUNDEDNESS. A recruitment is one deeper than the deepest anchor it
    uses, so rank strictly increases along a chain. A cycle would need a rank
    greater than itself, and since `Anchor.from_ruling` can only be called on a
    ruling that already exists, two claims cannot be recruited on each other.
    Before this, they could: two claims at sigma 0.0003 were each admitted at
    0.6753 on the strength of the other and nothing objected, because an anchor
    held a SigmaResult and a SigmaResult carries no trace of having itself come
    from a recruitment.

    Rank 0 remains an ASSERTION by the caller that the anchor was established
    without recruiting anything. The library cannot check that, in the same way
    it cannot check a forward model; what it can do is make the honest path the
    easy one and make the dishonest one a written falsehood rather than an
    oversight.
    """
    floor = check_unit(floor, "floor")
    value, exact, binding, names = _value_of(standing, answers)

    if value >= floor:
        # The floor is cleared on the model alone. That makes the claim
        # ADMISSIBLE. Admission needs a certificate, because sigma here is a
        # prediction from laws nobody measured, and omitting a state from the
        # model can only inflate it (results v7, the one-sided error): a
        # refusal survives every correction of the model, an admission does
        # not, so an admission must rest on readings.
        if certificates is None:
            return Ruling(
                rule=Rule.ADMISSIBLE,
                admitted=False,
                sigma=value,
                floor=floor,
                answers=names,
                binding_pair=binding,
                exact=exact,  # a bracket bounds; it does not exhibit a rule
                reason=(
                    f"sigma {value:.4f} >= floor {floor:.4f} on the model alone. "
                    f"Admissible, not admitted: no certificate exhibits the "
                    f"separation on readings, so this may be carried and may "
                    f"not be relied on"
                ),
            )
        bound = certificate_bound(certificates, names)
        simulated = unmeasured_pairs(certificates, names)
        if simulated:
            # The bound clears or misses the floor, and either way it was
            # scored on samples the model produced. That tests the library
            # against its own law and says nothing about the world. Recorded,
            # named, and not an admission.
            return Ruling(
                rule=Rule.ADMISSIBLE,
                admitted=False,
                sigma=value,
                floor=floor,
                answers=names,
                binding_pair=binding,
                exact=exact,
                certificate=bound,
                reason=(
                    f"sigma {value:.4f} >= floor {floor:.4f}, and the certificate "
                    f"of {bound:.4f} was scored on the model's own samples for "
                    f"{len(simulated)} of {len(names) * (len(names) - 1) // 2} open pair(s). A simulated "
                    f"certificate inherits the model's errors, so this is "
                    f"admissible and not admitted"
                ),
            )
        if bound >= floor:
            return Ruling(
                rule=Rule.ADMITTED,
                admitted=True,
                sigma=value,
                floor=floor,
                answers=names,
                binding_pair=binding,
                exact=exact,
                certificate=bound,
                reason=(
                    f"sigma {value:.4f} >= floor {floor:.4f}, and a certificate "
                    f"of {bound:.4f} on held-out readings clears the floor on "
                    f"every open pair"
                ),
            )
        return Ruling(
            rule=Rule.ADMISSIBLE,
            admitted=False,
            sigma=value,
            floor=floor,
            answers=names,
            binding_pair=binding,
            exact=exact,
            certificate=bound,
            reason=(
                f"sigma {value:.4f} >= floor {floor:.4f} but the certificate "
                f"reaches only {bound:.4f}. A certificate is a lower bound, so "
                f"this does not contradict the model. It means admission would "
                f"rest on the model alone, and that is not admission"
            ),
        )

    if anchors and answers is not None and recompute is not None:
        for anchor in anchors:
            if not anchor.has_standing or anchor.rules_out not in answers.names:
                continue
            if max_rank is not None and anchor.rank >= max_rank:
                continue
            narrowed = answers.narrowed(ruled_out=anchor.rules_out)
            after = recompute(narrowed)
            if not isinstance(after, SigmaResult):
                raise StandingError(
                    f"recompute must return a SigmaResult, got "
                    f"{type(after).__name__}. Recruitment re-scores sigma against "
                    f"the narrowed set, and the ruling records what that scoring "
                    f"was, so a bare number cannot stand in for it."
                )
            if after.value >= floor:
                return Ruling(
                    rule=Rule.RECRUIT,
                    admitted=True,
                    sigma=value,
                    sigma_after=after.value,
                    floor=floor,
                    answers=narrowed.names,
                    binding_pair=after.binding_pair,
                    recruited=anchor.name,
                    ruled_out=anchor.rules_out,
                    exact=after.exact,
                    rank=anchor.rank + 1,
                    reason=(
                        f"sigma rose {value:.4f} -> {after.value:.4f} when "
                        f"{anchor.name} ruled out {anchor.rules_out} on its own "
                        f"standing of {anchor.sigma:.4f} against floor "
                        f"{anchor.floor:.4f}, with no change in the data"
                    ),
                )

    if value > 0.0:
        return Ruling(
            rule=Rule.CARRY,
            admitted=False,
            sigma=value,
            floor=floor,
            answers=names,
            binding_pair=binding,
            exact=exact,
            reason=(
                f"sigma {value:.4f} < floor {floor:.4f}: identifiable but not "
                f"here, carried with its standing rather than deleted"
            ),
        )

    pair = " and ".join(binding) if binding else "the open answers"
    return Ruling(
        rule=Rule.ABSTAIN,
        admitted=False,
        sigma=0.0,
        floor=floor,
        answers=names,
        binding_pair=binding,
        exact=exact,
        reason=f"nothing that is a function of the readings could tell {pair} apart",
    )
