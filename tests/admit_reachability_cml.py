"""How often can a claim be ADMITTED once the certificate is composed with drift?

The open question the theory could not answer. Sigma itself is not measurable at
fixed conditions, so what a deployment can actually stand on is the drift-charged
certificate: T10 loses some of the true value by scanning one statistic, and T11
then subtracts the same-answer null. If that composition almost never clears a
floor, the ADMIT branch is decorative and the four rules are really three.

This measures it on the deployed CML pipeline across its own day conditions.
Run inside the container, from /app.
"""

from __future__ import annotations

import sys

import numpy as np

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.certify import certify, split
from standing.drift import bracket
from standing.model import Conditions
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.sigma import load_standing_config

try:
    import standing_properties as props  # type: ignore
except ModuleNotFoundError:
    from experiments import standing_properties as props  # type: ignore

DRAWS = 8000


class CmlLaw:
    def __init__(self, day, answer, seed):
        self.day, self.answer, self.seed = day, answer, seed

    def sample(self, size, rng):
        return cml_channel.simulate_day_statistic(self.day, self.answer, int(size), self.seed)


class CmlModel:
    def __init__(self, day, loci, seed=0):
        self.day, self.loci, self.seed = day, loci, seed

    def _cml(self, answer):
        return CmlAnswer(name=answer.name, locus=self.loci[answer.name])

    def law(self, answer, conditions):
        return CmlLaw(self.day, self._cml(answer), self.seed)

    def indistinguishable(self, left, right, conditions):
        return cml_channel.structurally_identical(self.day, self._cml(left), self._cml(right))


def main() -> int:
    cfg = load_standing_config()
    floor = float(cfg.sigma_min)
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    answers = AnswerSet(tuple(LibAnswer(n, tuple(sorted(loci[n].items()))) for n in loci))

    days = [
        ("96 windows", props.day(96)),
        ("72 windows", props.day(72)),
        ("48 windows", props.day(48)),
        ("24 windows", props.day(24)),
        ("14 windows", props.day(14)),
        ("8 windows", props.day(8)),
        ("4 windows", props.day(4)),
        ("2 windows", props.day(2)),
        ("1 window", props.day(1)),
        ("96w, 6 live", props.day(96, live=6)),
        ("96w, 4 live", props.day(96, live=4)),
        ("96w, 2 live", props.day(2 * 48, live=2)),
        ("96w, 1 live", props.day(96, live=1)),
        ("network only", props.day(96, speech=False)),
        ("speech only", props.day(96, network=False)),
    ]

    print(f"floor {floor}   draws {DRAWS}\n")
    print(f"{'conditions':>14} {'sigma':>7} {'cert':>7} {'null':>7} {'low':>7}  "
          f"{'by sigma':>9} {'by low':>8}  {'downgrade':>9}")

    counts = {"admit_sigma": 0, "admit_low": 0, "rows": 0}
    for label, day in days:
        model = CmlModel(day, loci)
        conditions = Conditions({"windows": day.observed_windows, "peak_k": day.peak_k})
        rng = np.random.default_rng(7)

        value = lib_sigma(answers, model, conditions, draws=DRAWS, rng=0).value

        selection, scoring, silent = {}, {}, False
        for answer in answers.answers:
            draws = np.asarray(model.law(answer, conditions).sample(DRAWS, rng)).ravel()
            if draws.size < 4:
                silent = True
                break
            selection[answer.name], scoring[answer.name] = split(draws, rng)
        if silent:
            print(f"{label:>14} {value:7.4f}  silent: no statistic exists to certify")
            continue

        cert = certify(answers, selection, scoring, source="model")[0]

        first = answers.answers[0]
        null_draws = np.asarray(model.law(first, conditions).sample(DRAWS, rng)).ravel()
        left, right = split(null_draws, rng)
        null = certify(
            AnswerSet((LibAnswer("n0", ("n", 0)), LibAnswer("n1", ("n", 1)))),
            {"n0": left, "n1": right},
            {"n0": right, "n1": left},
            source="model",
        )[0]

        low = bracket(max(cert, null), null).low
        by_sigma = "admit" if value >= floor else ("carry" if value > 0 else "abstain")
        by_low = "admit" if low >= floor else ("carry" if low > 0 else "abstain")
        counts["rows"] += 1
        counts["admit_sigma"] += by_sigma == "admit"
        counts["admit_low"] += by_low == "admit"
        flag = "  <-- lost" if by_sigma == "admit" and by_low != "admit" else ""
        print(
            f"{label:>14} {value:7.4f} {cert:7.4f} {null:7.4f} {low:7.4f}  "
            f"{by_sigma:>9} {by_low:>8}{flag}"
        )

    print(
        f"\nadmitted on sigma: {counts['admit_sigma']}/{counts['rows']}   "
        f"admitted on the drift-charged certificate: {counts['admit_low']}/{counts['rows']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
