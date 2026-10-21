"""The four dispositions, and the boundary a weight must not cross."""

from __future__ import annotations

import pytest

from standing.answers import Answer, AnswerSet
from standing.drift import bracket
from standing.errors import StandingError
from standing.model import Conditions, GaussianLaw
from standing.rules import Anchor, Rule, decide
from standing.sigma import sigma
from standing.weight import Weight


class Shifted:
    def __init__(self, offsets):
        self.offsets = offsets

    def law(self, answer, conditions):
        return GaussianLaw(self.offsets[answer.name], 1.0)


C = Conditions({"spread": 1.0})


def build(offsets):
    return AnswerSet(tuple(Answer(n, (n,)) for n in offsets)), Shifted(offsets)


def test_the_floor_alone_makes_a_claim_admissible_and_not_admitted():
    """A virtual sigma can be inflated by a state the model forgot (one-sided).

    So clearing the floor on the model admits nothing. It makes the claim
    admissible: it may be carried, and a consumer may not rely on it.
    """
    answers, model = build({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert ruling.rule is Rule.ADMISSIBLE and not ruling.admitted
    assert ruling.certificate is None
    assert "not admitted" in ruling.reason


def _certificates(offsets, n=2000, seed=0):
    """Certificates from readings the laws themselves generate, split honestly."""
    import numpy as np
    from standing.certify import certify, split

    answers, model = build(offsets)
    rng = np.random.default_rng(seed)
    selection, scoring = {}, {}
    for answer in answers.answers:
        selection[answer.name], scoring[answer.name] = split(
            model.law(answer, C).sample(n, rng), rng
        )
    _, certs = certify(answers, selection, scoring, source="readings")
    return answers, model, certs


def test_a_certificate_that_clears_the_floor_admits():
    answers, model, certs = _certificates({"a": 0.0, "b": 3.0})
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.rule is Rule.ADMITTED and ruling.admitted
    assert ruling.certificate is not None and ruling.certificate >= 0.25
    assert ruling.certificate <= ruling.sigma + 0.06  # a lower bound (T10)


def test_a_certificate_below_the_floor_leaves_the_claim_admissible():
    """Offsets 1.6 apart: sigma 0.274 clears 0.25, the best threshold rule reaches ~0.18.

    A certificate is a lower bound, so this is not a contradiction. It is the
    case where admission would rest on the model alone, and that is refused.
    """
    answers, model, certs = _certificates({"a": 0.0, "b": 1.6})
    ruling = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert ruling.sigma >= 0.25
    assert ruling.rule is Rule.ADMISSIBLE and not ruling.admitted
    assert ruling.certificate is not None and ruling.certificate < 0.25
    assert "rest on the model alone" in ruling.reason


def test_a_certificate_exactly_at_the_floor_admits_and_a_hair_below_does_not():
    """The gate is >=, and a mutation to > survived the suite until this test."""
    answers, model, certs = _certificates({"a": 0.0, "b": 3.0})
    bound = min(c.value for c in certs.values())
    at = decide(sigma(answers, model, C), floor=bound, certificates=certs)
    assert at.rule is Rule.ADMITTED and at.certificate == pytest.approx(bound)
    below = decide(sigma(answers, model, C), floor=min(bound + 1e-6, 1.0), certificates=certs)
    assert below.rule is Rule.ADMISSIBLE


def test_a_certificate_that_skips_a_pair_certifies_nothing():
    from standing.errors import CertificateError

    answers, model, certs = _certificates({"a": 0.0, "b": 3.0, "c": 6.0})
    partial = {pair: cert for pair, cert in certs.items() if pair == ("a", "b")}
    with pytest.raises(CertificateError, match="skips a pair"):
        decide(sigma(answers, model, C), floor=0.25, certificates=partial)


def test_a_certificate_is_looked_up_in_either_order():
    answers, model, certs = _certificates({"a": 0.0, "b": 3.0})
    flipped = {(b, a): cert for (a, b), cert in certs.items()}
    assert decide(sigma(answers, model, C), floor=0.25, certificates=flipped).rule is Rule.ADMITTED


def test_a_number_in_place_of_a_certificate_is_refused():
    from standing.errors import CertificateError

    answers, model = build({"a": 0.0, "b": 3.0})
    with pytest.raises(CertificateError, match="not a Certificate"):
        decide(sigma(answers, model, C), floor=0.25, certificates={("a", "b"): 0.9})
    with pytest.raises(CertificateError, match="not a certificate of anything"):
        decide(sigma(answers, model, C), floor=0.25, certificates={})


def test_an_admissible_claim_cannot_lend_standing():
    """D3 for the direct case: only an admitted claim may anchor another."""
    answers, model = build({"a": 0.0, "b": 3.0})
    admissible = decide(sigma(answers, model, C), floor=0.25)
    with pytest.raises(StandingError, match="cannot lend standing"):
        Anchor.from_ruling("first", "b", admissible)
    answers, model, certs = _certificates({"a": 0.0, "b": 3.0})
    admitted = decide(sigma(answers, model, C), floor=0.25, certificates=certs)
    assert Anchor.from_ruling("first", "b", admitted).has_standing


def test_carry_keeps_a_small_but_nonzero_standing_alive():
    answers, model = build({"a": 0.0, "b": 0.4})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert ruling.rule is Rule.CARRY
    assert not ruling.admitted and ruling.sigma > 0.0
    assert "carried with its standing" in ruling.reason


def test_abstain_only_at_exactly_zero():
    answers, model = build({"a": 0.0, "b": 0.0})
    ruling = decide(sigma(answers, model, C), floor=0.25)
    assert ruling.rule is Rule.ABSTAIN and ruling.sigma == 0.0


def _standing(separation):
    """A real SigmaResult for an anchor, since a bare number is refused.

    An anchor lends its entitlement to another claim, so it has to exhibit it.
    """
    answers, model = build({"seen": 0.0, "unseen": separation})
    return sigma(answers, model, C)


def test_recruit_raises_standing_with_the_data_fixed():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    before = sigma(answers, model, C)
    ruling = decide(
        before,
        floor=0.25,
        answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda narrowed: sigma(narrowed, model, C),
    )
    assert ruling.rule is Rule.RECRUIT and ruling.admitted
    assert ruling.sigma_after > ruling.sigma
    assert ruling.ruled_out == "confounded"


def test_an_anchor_without_standing_cannot_lend_any():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    ruling = decide(
        sigma(answers, model, C),
        floor=0.25,
        answers=answers,
        anchors=[Anchor("weak", "confounded", standing=_standing(0.02), floor=0.25)],
        recompute=lambda narrowed: sigma(narrowed, model, C),
    )
    assert ruling.rule is Rule.CARRY


def test_a_bracket_is_judged_on_its_low_end():
    answers, _ = build({"a": 0.0, "b": 3.0})
    assert decide(bracket(0.9019, 0.1710), floor=0.25, answers=answers).rule is Rule.ADMISSIBLE
    assert decide(bracket(0.9019, 0.1710), floor=0.30, answers=answers).rule is Rule.CARRY


def test_a_bracket_must_be_given_the_answer_set_it_was_scored_against():
    """Without it the ruling is answer-less and its record fails verification."""
    with pytest.raises(StandingError, match="carries no answer set"):
        decide(bracket(0.9019, 0.1710), floor=0.25)


def test_a_bracket_admission_is_not_reported_as_exact():
    """A bracket bounds the true value; it does not exhibit a rule."""
    answers, _ = build({"a": 0.0, "b": 3.0})
    ruling = decide(bracket(0.9019, 0.1710), floor=0.25, answers=answers)
    assert ruling.rule is Rule.ADMISSIBLE and not ruling.exact
    assert ruling.answers == answers.names


def test_a_weight_cannot_support_a_disposition():
    heuristic = Weight({"ceiling": 1.0, "context": 0.6}, view="csi_z_roll")
    assert heuristic.value == pytest.approx(0.6)
    with pytest.raises(StandingError, match="not standing"):
        decide(heuristic, floor=0.25)


def test_a_weight_has_no_sigma_and_says_why():
    heuristic = Weight({"ceiling": 1.0, "context": 0.6})
    with pytest.raises(StandingError, match="T1, T2, T6 or T10"):
        _ = heuristic.sigma


def test_a_bare_float_is_refused():
    with pytest.raises(StandingError):
        decide(0.9, floor=0.25)


def test_an_anchor_cannot_lend_standing_it_never_had():
    """RECRUIT once admitted a claim on a hand-typed float.

    A bare 0.99, with no model and no data, turned a CARRY into an admitted
    RECRUIT and the resulting record verified clean. An anchor lends its
    entitlement, so it has to exhibit it.
    """
    with pytest.raises(StandingError, match="bare number"):
        Anchor("fabricated", "confounded", standing=0.99, floor=0.25)
    with pytest.raises(StandingError, match="SigmaResult or a drift Bracket"):
        Anchor("nonsense", "confounded", standing="0.99", floor=0.25)


def test_an_anchor_may_exhibit_its_standing_as_a_bracket():
    """Under drift a claim rests on the bracket's low end, anchors included."""
    anchor = Anchor("clock", "confounded", standing=bracket(0.9019, 0.1710), floor=0.25)
    assert anchor.has_standing
    assert anchor.sigma == pytest.approx(bracket(0.9019, 0.1710).low)


def test_the_ruling_records_where_the_anchor_got_its_standing():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    ruling = decide(
        sigma(answers, model, C),
        floor=0.25,
        answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda narrowed: sigma(narrowed, model, C),
    )
    assert "on its own standing of" in ruling.reason


def _recruitable():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    return answers, model, sigma(answers, model, C)


def test_a_recruitment_is_one_deeper_than_the_anchor_it_used():
    """Rank strictly increases along a chain, which is what forbids a cycle."""
    answers, model, before = _recruitable()
    ruling = decide(
        before, floor=0.25, answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda n: sigma(n, model, C),
    )
    assert ruling.rule is Rule.RECRUIT
    assert ruling.rank == 1


def test_an_anchor_built_from_a_ruling_carries_that_ruling_s_depth():
    answers, model, before = _recruitable()
    first = decide(
        before, floor=0.25, answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda n: sigma(n, model, C),
    )
    onward = Anchor.from_ruling("first_claim", "confounded", first)
    assert onward.rank == first.rank == 1
    assert onward.has_standing

    second = decide(
        before, floor=0.25, answers=answers, anchors=[onward],
        recompute=lambda n: sigma(n, model, C),
    )
    assert second.rank == 2


def test_two_claims_cannot_be_recruited_on_each_other():
    """The cycle that used to admit both at 0.6753 from a base of 0.0003.

    It is unconstructible rather than detected: building Q's anchor needs P's
    ruling and building P's needs Q's, so neither can be made first. Before
    this, an Anchor held a SigmaResult and a SigmaResult carries no trace of
    having itself come from a recruitment.
    """
    answers, model, before = _recruitable()
    with pytest.raises(StandingError, match="needs a Ruling"):
        Anchor.from_ruling("Q", "confounded", None)

    # A claim that was not admitted cannot lend standing onward either.
    refused = decide(before, floor=0.99, answers=answers)
    assert refused.admitted is False
    with pytest.raises(StandingError, match="cannot lend standing"):
        Anchor.from_ruling("Q", "confounded", refused)


def test_a_chain_can_be_capped_by_depth():
    """A long chain is not unsound, but its coverage decays with length."""
    answers, model, before = _recruitable()
    first = decide(
        before, floor=0.25, answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda n: sigma(n, model, C),
    )
    deep = Anchor.from_ruling("first_claim", "confounded", first)
    assert decide(before, floor=0.25, answers=answers, anchors=[deep],
                  recompute=lambda n: sigma(n, model, C),
                  max_rank=1).rule is Rule.CARRY
    assert decide(before, floor=0.25, answers=answers, anchors=[deep],
                  recompute=lambda n: sigma(n, model, C),
                  max_rank=2).rule is Rule.RECRUIT


@pytest.mark.parametrize("bad", [-1, 1.5, True, "0"])
def test_a_rank_that_is_not_a_natural_number_is_refused(bad):
    with pytest.raises(StandingError, match="non-negative integer rank"):
        Anchor("x", "confounded", standing=_standing(4.0), floor=0.25, rank=bad)


def test_rank_zero_is_the_default_and_means_grounded():
    anchor = Anchor("sensor", "confounded", standing=_standing(4.0), floor=0.25)
    assert anchor.rank == 0


def test_an_anchor_may_not_assert_what_its_own_audit_forbids():
    """Ruling an answer out is an ASSERTION, not a magnitude question.

    The case this exists to refuse, measured: a view whose asserted answer is the
    even mixture of its own two rivals scores 0.2889 at the answer level, clears
    a floor of 0.25, and is provably unassertable at every prior. RECRUIT fired
    on it, because has_standing was sigma against a floor and nothing else.
    """
    with pytest.raises(StandingError, match="lists as forbidden"):
        Anchor(
            "mid_view", "confounded", standing=_standing(4.0), floor=0.25,
            asserts="c", forbidden=frozenset({"c"}),
        )


def test_an_anchor_asserting_something_permitted_still_lends_standing():
    anchor = Anchor(
        "mid_view", "confounded", standing=_standing(4.0), floor=0.25,
        asserts="a", forbidden=frozenset({"c"}),
    )
    assert anchor.has_standing


def test_a_forbidden_set_without_a_declared_assertion_is_refused():
    """Otherwise the two cannot be checked against each other and the set is decoration."""
    with pytest.raises(StandingError, match="does not say"):
        Anchor(
            "mid_view", "confounded", standing=_standing(4.0), floor=0.25,
            forbidden=frozenset({"c"}),
        )


def test_an_unaudited_anchor_still_works_and_says_so():
    """An empty forbidden set is the absence of a check, not a clean bill of health."""
    anchor = Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)
    assert anchor.has_standing
    assert anchor.forbidden == frozenset()
    assert anchor.asserts is None


def test_a_forbidden_assertion_removes_standing_even_above_the_floor():
    """has_standing is not sigma against a floor once an audit is present."""
    from standing.rules import Anchor as A

    permitted = A("v", "confounded", standing=_standing(4.0), floor=0.25,
                  asserts="ok", forbidden=frozenset({"bad"}))
    assert permitted.sigma >= permitted.floor and permitted.has_standing
    # the same standing, asserting the forbidden answer, cannot be built at all
    with pytest.raises(StandingError):
        A("v", "confounded", standing=_standing(4.0), floor=0.25,
          asserts="bad", forbidden=frozenset({"bad"}))


def test_an_anchor_with_no_standing_of_its_own_cannot_lend_any():
    """sigma exactly zero is ABSTAIN, and no floor may override a theorem.

    sigma is non-negative by construction, so `sigma >= floor` is
    unconditionally true at floor 0. An anchor whose own view separated nothing
    at all lent its standing, and a claim at 0.0003 was admitted at 0.6753 on
    the strength of it.
    """
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})

    class Blind:
        def law(self, answer, conditions):
            return GaussianLaw(0.0, 1.0)

    nothing = sigma(answers, Blind(), C)
    assert nothing.value == 0.0

    for floor in (0.0, 0.25):
        assert Anchor("blind", "confounded", standing=nothing, floor=floor).has_standing is False

    ruling = decide(
        sigma(answers, model, C), floor=0.25, answers=answers,
        anchors=[Anchor("blind", "confounded", standing=nothing, floor=0.0)],
        recompute=lambda n: sigma(n, model, C),
    )
    assert ruling.rule is Rule.CARRY and ruling.admitted is False


def test_a_positive_anchor_at_floor_zero_still_lends():
    """The fix must not make floor 0 useless, only make it honest."""
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    anchor = Anchor("clock", "confounded", standing=sigma(answers, model, C), floor=0.0)
    assert anchor.sigma > 0.0 and anchor.has_standing


def test_every_disposition_carries_whether_sigma_was_exact():
    """CARRY and ABSTAIN rulings dropped `exact` (then `certified`) on the floor; the sweep found it."""
    answers, model = build({"a": 0.0, "b": 0.4})
    assert decide(sigma(answers, model, C), floor=0.25).exact is True          # CARRY, closed form
    answers, model = build({"a": 0.0, "b": 0.0})
    assert decide(sigma(answers, model, C), floor=0.25).exact is True          # ABSTAIN, closed form

    class Sampled:
        class _Law:
            def sample(self, size, rng):
                return rng.normal(0.0, 1.0, size)

        def law(self, answer, conditions):
            return self._Law()

    sampled = decide(sigma(answers, Sampled(), C, draws=200, rng=0), floor=0.99)
    assert sampled.rule is Rule.CARRY and sampled.exact is False


def test_an_anchor_exactly_at_its_floor_lends_standing():
    answers, model = build({"a": 0.0, "b": 3.0})
    result = sigma(answers, model, C)
    anchor = Anchor("edge", "b", standing=result, floor=result.value)
    assert anchor.has_standing


def test_anchors_without_an_answer_set_or_a_recompute_cannot_recruit_and_do_not_crash():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    anchor = Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)
    no_recompute = decide(sigma(answers, model, C), floor=0.25, answers=answers, anchors=[anchor])
    assert no_recompute.rule is Rule.CARRY
    no_answers = decide(sigma(answers, model, C), floor=0.25, anchors=[anchor],
                        recompute=lambda n: sigma(n, model, C))
    assert no_answers.rule is Rule.CARRY


def test_a_recruitment_that_lands_exactly_on_the_floor_is_a_recruitment():
    answers, model = build({"absent": 0.0, "present": 3.0, "confounded": 0.05})
    narrowed_value = sigma(answers.narrowed(ruled_out="confounded"), model, C).value
    ruling = decide(
        sigma(answers, model, C), floor=narrowed_value, answers=answers,
        anchors=[Anchor("clock", "confounded", standing=_standing(4.0), floor=0.25)],
        recompute=lambda n: sigma(n, model, C),
    )
    assert ruling.rule is Rule.RECRUIT and ruling.sigma_after == pytest.approx(narrowed_value)
