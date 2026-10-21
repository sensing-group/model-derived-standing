"""One modality, many sensors: what pooling costs, swept.

Theory v18 §8m says two things about a homogeneous array, and this measures
both. Write J for the joint reading of the live sensors and R for the scalar
the pipeline thresholds.

    J   a sensor that reads the same under both answers contributes a
        Bhattacharyya factor of exactly 1 and drops out (T15), so the joint
        value should not move at all as non-contributing sensors are added,
        and should be exactly 0 when none contributes.

    R   pooling puts the non-contributors' noise into the law of the statistic,
        so the difference should dilute: strictly positive for every finite N
        and heading to zero.

This docstring said the diluted value was "the worst reading available because
it looks like standing", which is what the theory said before this script was
run. The run refutes it. At a floor of 0.25 a diluted 0.0025 looks like a
REFUSAL, and T6 bounds the pooled value above by the joint one, so pooling can
only ever understate. The array regime is conservative and wasteful, not
unsound: it refuses claims the hardware could have supported. Left in the file
as a correction rather than deleted, because the wrong direction was believed
here first and the sweep is what settled it.

The setup is the simplest one that can show it. N unit-variance Gaussian
sensors. k of them are in a position to contribute: mean 0 under `quiet` and
`delta` under `loud`. The other N-k read the same under both answers. R is the
mean over all N, which is what a deployment computes when it does not know
which sensors are in position.

J is computed through the library's own combination law rather than by a
formula written here, because `combine_heterogeneous` IS the tensorisation T15
rests on: 1 - prod(1 - h_i). If a non-contributing sensor did not drop out
exactly, this is where it would show.

    python examples/array_sweep.py
"""

from __future__ import annotations

import math

from standing import (
    Answer,
    AnswerSet,
    Conditions,
    GaussianLaw,
    combine_heterogeneous,
    decide,
    sigma,
)

FLOOR = 0.25
DELTA = 2.0            # separation, in sd, at a sensor that can hear
C = Conditions({"live": True})


class OneSensor:
    """A single sensor: it either hears the difference or it does not."""

    def __init__(self, separation: float):
        self.separation = separation

    def law(self, answer, conditions):
        return GaussianLaw(self.separation if answer.name == "loud" else 0.0, 1.0)


class PooledArray:
    """The mean over N sensors, k of which hear the difference.

    Under `quiet` every sensor is centred at 0. Under `loud` the k contributing
    sensors shift by delta, so the MEAN shifts by k*delta/N while the noise of
    the mean falls only as 1/sqrt(N). That ratio is the dilution.
    """

    def __init__(self, n: int, k: int, separation: float):
        self.n, self.k, self.separation = n, k, separation

    def law(self, answer, conditions):
        shift = (self.k * self.separation / self.n) if answer.name == "loud" else 0.0
        return GaussianLaw(shift, 1.0 / math.sqrt(self.n))


PAIR = AnswerSet((Answer("quiet", "quiet"), Answer("loud", "loud")))


def joint(n: int, k: int) -> float:
    """sigma over J, through the library's own combination law."""
    per_sensor = [
        sigma(PAIR, OneSensor(DELTA if i < k else 0.0), C).value for i in range(n)
    ]
    return combine_heterogeneous(per_sensor, dependence="independent")


def pooled(n: int, k: int) -> float:
    """sigma over R, the statistic a deployment that pools everything computes."""
    return sigma(PAIR, PooledArray(n, k, DELTA), C).value


def pooled_rule(n: int, k: int) -> str:
    """The disposition for R, from the library rather than from a copy of it."""
    return decide(sigma(PAIR, PooledArray(n, k, DELTA), C), floor=FLOOR).rule.value


def joint_note(value: float) -> str:
    """How J compares to the floor, WITHOUT calling it a ruling.

    `combine_heterogeneous` returns a float and the library will not let a
    combination be read as a SigmaResult, which is a refusal worth respecting
    here rather than working around: a combined value is a ceiling on what the
    array could do, not a claim any view is entitled to make. So this reports
    the comparison and does not pretend it is a disposition.
    """
    if value == 0.0:
        return "zero"
    return "over" if value >= FLOOR else "under"


def main() -> int:
    print(f"delta {DELTA} sd at a contributing sensor, floor {FLOOR}\n")

    print("=== k = 1 contributing sensor, N growing ===")
    print(f"{'N':>6} {'k':>3} {'joint J':>10} {'pooled R':>10} {'J vs floor':>10} "
          f"{'R rule':>8} {'thrown away':>12}")
    for n in (1, 2, 5, 10, 25, 50, 100, 200, 500, 1000):
        j, r = joint(n, 1), pooled(n, 1)
        print(f"{n:6} {1:3} {j:10.4f} {r:10.4f} {joint_note(j):>8} {pooled_rule(n, 1):>8} "
              f"{j - r:12.4f}")

    print("\n=== k = 0: nothing in position, which is the categorical case ===")
    print(f"{'N':>6} {'k':>3} {'joint J':>10} {'pooled R':>10} {'J vs floor':>10} "
          f"{'R rule':>8}")
    for n in (1, 10, 100, 1000):
        j, r = joint(n, 0), pooled(n, 0)
        print(f"{n:6} {0:3} {j:10.4f} {r:10.4f} {joint_note(j):>8} {pooled_rule(n, 0):>8}")

    print("\n=== N = 200 fixed, k growing: does the joint see each one? ===")
    print(f"{'N':>6} {'k':>3} {'joint J':>10} {'pooled R':>10} {'J vs floor':>10} "
          f"{'R rule':>8}")
    for k in (0, 1, 2, 5, 10, 25, 50, 100, 200):
        j, r = joint(200, k), pooled(200, k)
        print(f"{200:6} {k:3} {j:10.4f} {r:10.4f} {joint_note(j):>8} {pooled_rule(200, k):>8}")

    # The two claims, checked rather than asserted.
    flat = {round(joint(n, 1), 12) for n in (1, 10, 100, 1000)}
    zeros = {joint(n, 0) for n in (1, 10, 100, 1000)}
    decays = [pooled(n, 1) for n in (10, 100, 1000)]
    print("\nT15, checked:")
    print(f"  joint is unchanged by non-contributing sensors   {len(flat) == 1}"
          f"   (values seen: {sorted(flat)})")
    print(f"  joint is EXACTLY zero when none contributes      {zeros == {0.0}}")
    print(f"  pooled is positive at every N, and falling       "
          f"{all(v > 0.0 for v in decays)} {[round(v, 6) for v in decays]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
