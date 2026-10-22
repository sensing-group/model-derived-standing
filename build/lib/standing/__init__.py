"""Model-derived standing: whether a reading was entitled to count as evidence.

Standing is computed from the model and the conditions, never from the values, so
it exists for an interval in which nothing was recorded. It asks, before a
reading counts, whether that view could have told apart the answers still open
under the conditions of that interval.

The short path:

    from standing import AnswerSet, Conditions, sigma, decide

    answers = AnswerSet.from_actions(states, action_for)   # the quotient
    result  = sigma(answers, my_model, Conditions({...}))  # per view, pre-fusion
    ruling  = decide(result, floor=0.25)                   # admit/recruit/carry/abstain

Most of what this library does is refuse. `combine_replicates` will not take a
sequence, `certify_pair` will not score a threshold on the sample that chose it,
`Conditions` will not hold a reading, `AnswerSet` will not hold a synonym, and a
`Weight` has no `.sigma`. Each refusal exists because a deployment made that
mistake and the result was a plausible number nobody questioned.
"""

from standing.answers import Answer, AnswerSet, QuotientOrigin
from standing.audit import Audit, Check, Coverage, Probe, audit, coverage_report
from standing.certify import Certificate, certify, certify_pair, split
from standing.checks import (
    Collapse,
    QuotientAudit,
    QuotientCheck,
    QuotientMerge,
    QuotientSplit,
    answer_set_ladder,
    null_calibration,
    quotient_audit,
    quotient_check,
    redundancy_report,
    seed_stability,
)
from standing.divergence import (
    hellinger2_gaussian,
    hellinger2_samples,
    jeffreys_gaussian,
    jeffreys_samples,
)
from standing.drift import Bracket, bracket
from standing.errors import (
    AnswerSetError,
    CertificateError,
    InvalidInput,
    StandingError,
)
from standing.model import Conditions, ForwardModel, GaussianLaw, Law
from standing.numerics import (
    combine_heterogeneous,
    combine_replicates,
    combine_views,
    h2_bernoulli,
    independence_inflation,
    n_eff,
    saturation_count,
)
from standing.record import SCHEMA, Record, Statistic, verify
from standing.rules import Anchor, Rule, Ruling, decide
from standing.sigma import SigmaResult, sigma, sigma_of_pair
from standing.weight import Weight

__version__ = "0.0.1"

__all__ = [
    "Audit",
    "Check",
    "Coverage",
    "Probe",
    "audit",
    "coverage_report",
    # the answers a claim ranges over
    "Answer",
    "AnswerSet",
    "QuotientOrigin",
    # what you supply
    "Conditions",
    "ForwardModel",
    "GaussianLaw",
    "Law",
    # the quantity
    "sigma",
    "sigma_of_pair",
    "SigmaResult",
    # exhibiting it
    "Certificate",
    "certify",
    "certify_pair",
    "split",
    # conditions that drift
    "Bracket",
    "bracket",
    # combining views
    "combine_heterogeneous",
    "combine_replicates",
    "combine_views",
    "h2_bernoulli",
    "independence_inflation",
    "n_eff",
    "saturation_count",
    # divergences
    "hellinger2_gaussian",
    "hellinger2_samples",
    "jeffreys_gaussian",
    "jeffreys_samples",
    # what a consumer may do
    "Anchor",
    "Rule",
    "Ruling",
    "decide",
    # crossing the boundary
    "Record",
    "SCHEMA",
    "Statistic",
    "verify",
    # the diagnostics
    "answer_set_ladder",
    "Collapse",
    "null_calibration",
    "quotient_audit",
    "QuotientAudit",
    "quotient_check",
    "QuotientCheck",
    "QuotientMerge",
    "QuotientSplit",
    "redundancy_report",
    "seed_stability",
    # deliberately not standing
    "Weight",
    # refusals
    "AnswerSetError",
    "CertificateError",
    "InvalidInput",
    "StandingError",
    "__version__",
]
