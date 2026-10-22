"""C1a against the deployment: is the answer set the quotient of the policy?

Run inside the CML container, from /app:

    docker exec -w /app -e PYTHONPATH=/app:/tmp/standing-lib mt_poc_api \\
        python /tmp/standing-lib/quotient_cml.py

Every other instrument in this repo scores the library against its own model.
This one asks a question only a real system can answer: are the answers the
deployment declares the answers it ACTS on? The deployment supplies both halves
and neither was written to be compared with the other, which is the whole point.

The policy is not invented here. `DayConditions.theta` is documented in the
deployment's own channel module as "the threshold the gate applies to the daily
statistic", and `dsm_gate.py` applies it: a day counts toward an indicator when
its statistic is at or above theta and does not otherwise. So the consumer's
action on a day-level claim is binary, and the action under an answer is read
off the deployment's own closed-form statistic for that answer.

Exit 0 when the two partitions agree, 1 when they do not.
"""

from __future__ import annotations

import sys

import numpy as np

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.checks import quotient_check
from standing.model import Conditions
from standing.rules import decide
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.sigma import compute_sigma, load_standing_config


class _Mixture:
    """The law of a GROUP: what an answer predicts once states are grouped.

    This is the object the merge case turns on. A coarse answer's law is the
    mixture over its members, and the consumer faces that mixture rather than
    any component of it.
    """

    def __init__(self, laws):
        self.laws = laws

    def sample(self, size, rng):
        picks = rng.integers(0, len(self.laws), size)
        out = np.empty(size)
        for index, law in enumerate(self.laws):
            mask = picks == index
            count = int(mask.sum())
            if count:
                out[mask] = law.sample(count, rng)
        return out


class CmlLaw:
    def __init__(self, day, answer, base_seed):
        self.day, self.answer, self.base_seed = day, answer, base_seed

    def sample(self, size, rng):
        return cml_channel.simulate_day_statistic(
            self.day, self.answer, int(size), self.base_seed
        )


class CmlModel:
    """Delegates to the deployment's model, and mixes over a grouped answer."""

    def __init__(self, day, loci, base_seed):
        self.day, self.loci, self.base_seed = day, loci, base_seed

    def _members(self, answer):
        if isinstance(answer.state, frozenset):
            return sorted(answer.state)
        return [answer.state]

    def law(self, answer, conditions):
        laws = [
            CmlLaw(self.day, CmlAnswer(name=name, locus=self.loci[name]), self.base_seed)
            for name in self._members(answer)
        ]
        return laws[0] if len(laws) == 1 else _Mixture(laws)


def policy_of(day, loci):
    """What the deployment DOES under an answer: count the day, or drop it.

    Read from the deployment's own closed form rather than from a simulation,
    so the action is a function of the state and not of a seed.
    """
    def act(state: str) -> str:
        mean, _sd = cml_channel.gaussian_day_statistic(
            day, CmlAnswer(name=state, locus=loci[state])
        )
        return "count" if mean >= day.theta else "drop"

    return act


def _knife_edge(day, loci) -> str:
    """Why `present` lands where it does, which is the second finding.

    standing.yaml sets the present locus to 0.405 and says why: it is logit(0.6),
    the directed z at which a biomarker's membership equals the gate threshold
    theta = 0.6. So `present` is DEFINED to sit exactly on the gate's boundary.
    Rounded to three places it sits just under: sigmoid(0.405) = 0.599870 rather
    than 0.600000, and the gate tests `>= theta`.
    """
    import math

    exact = math.log(day.theta / (1.0 - day.theta))
    mean, _ = cml_channel.gaussian_day_statistic(
        day, CmlAnswer(name="present", locus=loci["present"])
    )
    return (
        f"\nwhy `present` lands there, and it is not this check's arithmetic:\n"
        f"  standing.yaml sets the present locus to logit(theta), so `present` is\n"
        f"  DEFINED to sit exactly on the gate boundary rather than above it.\n"
        f"  logit({day.theta}) = {exact:.6f}, rounded in the config to "
        f"{loci['present'].get('speech')}\n"
        f"  sigmoid of the rounded value = {1/(1+math.exp(-loci['present'].get('speech'))):.6f}, "
        f"and the gate tests `>= {day.theta}`\n"
        f"  so the modelled statistic under `present` is {mean:.6f}, short by "
        f"{day.theta - mean:.6f}\n"
        f"  A day that is exactly present does not pass the deployment's own gate."
    )


def _repaired(day, loci, states):
    """The same loci with `present` at logit(theta) exactly, not rounded.

    Separating the two findings. The rounding is an arithmetic slip and is easy
    to fix; whether the answer set is the action quotient is a design question
    and survives the fix.
    """
    import math

    exact = math.log(day.theta / (1.0 - day.theta))
    out = {name: dict(locus) for name, locus in loci.items()}
    out["present"] = {aspect: exact for aspect in out["present"]}
    return out, policy_of(day, out), states


def main() -> int:
    cfg = load_standing_config()
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    states = sorted(loci)

    # The deployment's own day builder, the same one differential_cml uses.
    try:
        import standing_properties as props  # type: ignore
    except ModuleNotFoundError:
        from experiments import standing_properties as props  # type: ignore

    day = props.day(96, label="full")
    act = policy_of(day, loci)

    print(f"declared answers   {tuple(states)}")
    print(f"floor              {cfg.sigma_min}")
    print(f"gate threshold     theta = {day.theta}")
    print("\nwhat the deployment does under each declared answer, from its own")
    print("closed form for the statistic the gate thresholds:")
    for state in states:
        mean, sd = cml_channel.gaussian_day_statistic(
            day, CmlAnswer(name=state, locus=loci[state])
        )
        print(f"  {state:<20} statistic {mean:.4f} (sd {sd:.4f})  ->  {act(state)}")

    declared = AnswerSet(tuple(LibAnswer(name, name) for name in states))
    report = quotient_check(declared, act, states)
    print(f"\nquotient_check     {report.reason()}")
    if report.agrees:
        print("\nThe declared answer set IS the action quotient. C1a holds here.")
        return 0

    blocks = {act(state) for state in states}
    if len(blocks) < 2:
        print(f"\nThe action quotient has ONE block, {blocks}. Every declared "
              f"answer demands the same thing of the consumer, so on the "
              f"deployment's own numbers there is no question here to have "
              f"standing about, and the library refuses to build the set.")
        print(_knife_edge(day, loci))
        loci, act, states = _repaired(day, loci, states)
        blocks = {act(state) for state in states}
        print(f"\n== the same check with the rounding repaired ==")
        print("Set the present locus to logit(theta) exactly and ask again. This "
              "is a different forward model, so the numbers below describe the "
              "configuration the config comment intended, not the one deployed.")
        for state in states:
            mean, _sd = cml_channel.gaussian_day_statistic(
                day, CmlAnswer(name=state, locus=loci[state])
            )
            print(f"  {state:<20} statistic {mean:.6f}  ->  {act(state)}")
        declared = AnswerSet(tuple(LibAnswer(name, name) for name in states))
        report = quotient_check(declared, act, states)
        print(f"\nquotient_check     {report.reason()}")
        if report.agrees or len(blocks) < 2:
            return 1

    # It does not agree, so T13 applies and the size of it is measurable: score
    # the question the deployment asks beside the question it declared.
    quotient = AnswerSet.from_actions(states, act)
    model = CmlModel(day, loci, base_seed=getattr(cfg, "seed", 0))
    conditions = Conditions({"windows": day.observed_windows})
    asked = lib_sigma(quotient, model, conditions, draws=20_000, rng=0).value
    stated = lib_sigma(declared, model, conditions, draws=20_000, rng=0).value

    print(f"\naction quotient    {quotient.names}")
    print(f"  sigma over the question the deployment ASKS      {asked:.4f}  "
          f"-> {decide(lib_sigma(quotient, model, conditions, draws=20_000, rng=0), floor=cfg.sigma_min).rule.value}")
    print(f"  sigma over the answer set it DECLARES            {stated:.4f}  "
          f"-> {decide(lib_sigma(declared, model, conditions, draws=20_000, rng=0), floor=cfg.sigma_min).rule.value}")
    # Printed for orientation only, and NOT a differential comparison: it is
    # computed on the deployed loci and the deployed draw count, so it answers a
    # slightly different question from either line above. differential_cml.py is
    # where the two implementations are compared at matched settings.
    print(f"  the deployment's own sigma, deployed loci        "
          f"{compute_sigma(day, cfg, cfg.answers).value:.4f}  (not comparable above)")

    if (asked < cfg.sigma_min) != (stated < cfg.sigma_min):
        print("\nThe two land on OPPOSITE sides of the floor. The declared set is "
              "not merely a different question, it is a different verdict.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
