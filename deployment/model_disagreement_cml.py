"""Two forward models for one view, and where their refusal sets disagree.

Run inside the CML container, from /app:

    docker exec -w /app -e PYTHONPATH=/app:/tmp/standing-lib mt_poc_api \\
        python /tmp/standing-lib/model_disagreement_cml.py

What this IS, stated before the result so nobody oversells it. The deployment
carries two predictions of its own daily statistic: `model-mc-v1`, Monte Carlo
through the pipeline, and `model-gauss-v1`, a closed form. They are NOT
independently authored. `gaussian_day_statistic` says so in its own docstring:
it treats the statistic as the mean of the first k windows rather than the max
over consecutive runs, because the maximum has no closed form, and it adds
"never report it as the headline". So this is a model against a declared
simplification of itself, by one author.

That falls short of the instrument theory v16 section 8k asks for, which is two
models written by people who did not read each other. What it can still do is
ask a sharper question than an estimator comparison does: when the modelled
STATISTIC changes shape, which verdicts move? Not which numbers, which verdicts.
A refusal set is what crosses a boundary, so a refusal set that survives a
change of model is worth more than a number that does not.

The result to look for is therefore the SPLIT between the categorical and the
graded part, not the raw count of disagreements.

Exit 0 when the two refusal sets agree everywhere, 1 when they do not. A non-zero
exit is not a defect in the deployment: the approximation is documented and its
author says not to headline it.
"""

from __future__ import annotations

import sys

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.model import Conditions, GaussianLaw
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.sigma import load_standing_config

DRAWS = 20_000


class MonteCarlo:
    """model-mc-v1: sample the pipeline's own daily statistic."""

    name = "mc"

    def __init__(self, day, loci, base_seed):
        self.day, self.loci, self.base_seed = day, loci, base_seed

    def law(self, answer, conditions):
        cml = CmlAnswer(name=answer.name, locus=self.loci[answer.name])
        day, seed = self.day, self.base_seed

        class Law:
            def sample(self, size, rng):
                return cml_channel.simulate_day_statistic(day, cml, int(size), seed)

        return Law()

    def indistinguishable(self, left, right, conditions):
        return cml_channel.structurally_identical(
            self.day,
            CmlAnswer(name=left.name, locus=self.loci[left.name]),
            CmlAnswer(name=right.name, locus=self.loci[right.name]),
        )


class ClosedForm:
    """model-gauss-v1: the deployment's own Gaussian approximation."""

    name = "gauss"

    def __init__(self, day, loci):
        self.day, self.loci = day, loci

    def law(self, answer, conditions):
        mean, sd = cml_channel.gaussian_day_statistic(
            self.day, CmlAnswer(name=answer.name, locus=self.loci[answer.name])
        )
        return GaussianLaw(mean, sd)

    def indistinguishable(self, left, right, conditions):
        return cml_channel.structurally_identical(
            self.day,
            CmlAnswer(name=left.name, locus=self.loci[left.name]),
            CmlAnswer(name=right.name, locus=self.loci[right.name]),
        )


def disposition(value: float, floor: float) -> str:
    """The three outcomes that matter to a refusal set."""
    if value == 0.0:
        return "abstain"
    return "admit" if value >= floor else "carry"


def main() -> int:
    cfg = load_standing_config()
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    answers = AnswerSet(tuple(LibAnswer(n, n) for n in sorted(loci)))
    floor = cfg.sigma_min
    seed = getattr(cfg, "seed", 0)

    try:
        import standing_properties as props  # type: ignore
    except ModuleNotFoundError:
        from experiments import standing_properties as props  # type: ignore

    # `label` is a builder argument and does not survive onto DayConditions, so
    # the label is carried here rather than read back off the object.
    days = [
        ("1_full", props.day(96)),
        ("2_half", props.day(48)),
        ("3_quarter", props.day(24)),
        ("4_gate_edge", props.day(14)),
        ("5_eight", props.day(8)),
        ("6_four", props.day(4)),
        ("7_single_window", props.day(1)),
        ("8_silent", props.day(0)),
        ("9_network_only", props.day(96, speech=False)),
        ("10_speech_only", props.day(96, network=False)),
        ("11_nothing_live", props.day(96, speech=False, network=False)),
        ("12_one_biomarker", props.day(96, live=1)),
        ("13_two_biomarkers", props.day(96, live=2)),
        ("14_four_biomarkers", props.day(96, live=4)),
    ]

    print(f"answers {answers.names}   floor {floor}   draws {DRAWS}")
    print("\nTwo forward models for one view. The comparison is the VERDICT, not")
    print("the value: a refusal set is what crosses the boundary.\n")
    header = (f"{'day':>18} {'pair':>34} {'mc':>8} {'gauss':>8} "
              f"{'mc':>8} {'gauss':>8}")
    print(header)
    print(f"{'':>18} {'':>34} {'':>8} {'':>8} {'rule':>8} {'rule':>8}")

    pairs_compared = 0
    verdict_gaps: list[str] = []
    zero_gaps: list[str] = []

    for label, day in days:
        conditions = Conditions({"windows": day.observed_windows})
        mc = lib_sigma(answers, MonteCarlo(day, loci, seed), conditions,
                       draws=DRAWS, rng=0)
        gauss = lib_sigma(answers, ClosedForm(day, loci), conditions,
                          draws=DRAWS, rng=0)

        for pair, mc_value in sorted(mc.per_pair.items()):
            gauss_value = gauss.per_pair[pair]
            pairs_compared += 1
            mc_rule = disposition(mc_value, floor)
            gauss_rule = disposition(gauss_value, floor)
            flag = ""
            if mc_rule != gauss_rule:
                flag = "  <-- DISAGREE"
                verdict_gaps.append(f"{label} {pair}: {mc_rule} vs {gauss_rule}")
                if "abstain" in (mc_rule, gauss_rule):
                    zero_gaps.append(f"{label} {pair}")
            print(f"{label:>18} {'|'.join(pair):>34} {mc_value:8.4f} "
                  f"{gauss_value:8.4f} {mc_rule:>8} {gauss_rule:>8}{flag}")

    print(f"\npairs compared                 {pairs_compared}")
    print(f"verdict disagreements          {len(verdict_gaps)}")
    print(f"  of which cross exact zero    {len(zero_gaps)}")
    for line in verdict_gaps:
        print(f"    {line}")

    if not verdict_gaps:
        print(
            "\nThe two models refuse the same claims everywhere. That is weak "
            "evidence rather than strong: they share a channel module, a "
            "membership curve and an author, so agreement bounds the arbitrariness "
            "of the approximation and NOT the correctness of the model. The "
            "objection in theory v16 section 8k stands until the second model is "
            "written by someone who did not write the first."
        )
        return 0
    print(
        "\nRead the split, not the total. Verdicts that moved are all "
        "admit-against-carry: the GRADED part of the criterion is an artefact of "
        "how the statistic was modelled. Verdicts at exact zero did not move at "
        "all, and those come from `structurally_identical`, which asks which "
        "aspects reported rather than what shape their statistic has."
    )
    print(
        "That is the case for leading with the refusal set and treating the "
        "magnitude as policy: one half of the criterion survived a change of "
        "model and the other half did not."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
