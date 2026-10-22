"""Do per-aspect values combine into the joint one? T3 says no, and this checks.

The deployment has two aspects, speech and network, feeding ONE fused day
statistic. It is tempting to compute standing per aspect and combine the results
with T7c, and this measures what that gives against the joint value computed
properly.

T3 says sigma is aspect-relative but not aspect-decomposable, so the combination
should not reproduce the joint value. If it does not, `combine_heterogeneous`
needs a precondition stated in the library and not only in a paper: it combines
INDEPENDENT VIEWS THAT EACH PRODUCE THEIR OWN STATISTIC, and per-aspect slices of
one fused statistic are not that.

Run inside the container, from /app.
"""

from __future__ import annotations

import sys

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.model import Conditions
from standing.numerics import combine_heterogeneous
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.sigma import load_standing_config

try:
    import standing_properties as props  # type: ignore
except ModuleNotFoundError:
    from experiments import standing_properties as props  # type: ignore


class CmlLaw:
    def __init__(self, day, answer, seed=0):
        self.day, self.answer, self.seed = day, answer, seed

    def sample(self, size, rng):
        return cml_channel.simulate_day_statistic(self.day, self.answer, int(size), self.seed)


class CmlModel:
    """Optionally restrict every answer to one aspect, holding the rest at baseline."""

    def __init__(self, day, loci, aspect=None):
        self.day, self.loci, self.aspect = day, loci, aspect

    def _cml(self, answer):
        locus = dict(self.loci[answer.name])
        if self.aspect is not None:
            locus = {k: v for k, v in locus.items() if k == self.aspect}
        return CmlAnswer(name=answer.name, locus=locus)

    def law(self, answer, conditions):
        return CmlLaw(self.day, self._cml(answer))

    def indistinguishable(self, left, right, conditions):
        return cml_channel.structurally_identical(self.day, self._cml(left), self._cml(right))


def main() -> int:
    cfg = load_standing_config()
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    aspects = sorted({aspect for locus in loci.values() for aspect in locus})
    answers = AnswerSet(tuple(LibAnswer(n, tuple(sorted(loci[n].items()))) for n in loci))
    day = props.day(96)
    conditions = Conditions({"windows": day.observed_windows})

    joint = lib_sigma(answers, CmlModel(day, loci), conditions, draws=8000, rng=0).value
    per_aspect = {
        aspect: lib_sigma(
            answers, CmlModel(day, loci, aspect=aspect), conditions, draws=8000, rng=0
        ).value
        for aspect in aspects
    }

    print(f"aspects        {aspects}")
    for aspect, value in per_aspect.items():
        print(f"  sigma via {aspect:>8} alone   {value:.4f}")
    # `independent` is the OPTIMISTIC assumption and is deliberate: the point
    # below is that the combination fails even when granted the most generous
    # dependence a caller could claim. `dependence` has no default, because
    # independent and unknown are the same number and different claims.
    combined = combine_heterogeneous(list(per_aspect.values()), dependence="independent")
    print(f"\nT7c combination of the per-aspect values   {combined:.4f}")
    print(f"joint sigma, computed properly            {joint:.4f}")
    print(f"difference                                {abs(joint - combined):.4f}")

    if combined < joint - 1e-6:
        print(
            "\nT3 holds and the combination UNDERSTATES: the aspects are not "
            "independent views with their own statistics, they are slices of one "
            "fused statistic, so combining their standings is not the joint "
            "standing. combine_heterogeneous must not be used this way."
        )
    elif combined > joint + 1e-6:
        print("\nThe combination OVERSTATES, which would be the dangerous direction.")
    else:
        print("\nThey agree, which would make sigma aspect-decomposable here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
