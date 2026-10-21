"""A complete worked example: a pipeline, its answers, and what it may claim.

Run it:

    PYTHONPATH=src python3 examples/toy_pipeline.py

The pipeline is deliberately small and deliberately realistic in the one way that
matters: it decides on a statistic that is NOT the raw sensor value. A room
sensor reports a level per window; the pipeline takes the mean over the last k
windows and calls the room occupied above a threshold. Standing asks whether that
statistic, under the conditions of this interval, could have told apart the
answers a consumer actually acts on differently.

Everything below is the library's intended shape, in order: read the answers off
the actions, write the forward model, record the conditions, compute sigma before
any fusion, certify with a split sample, charge for drift, apply the rules, and
emit a record someone else can check.
"""

from __future__ import annotations

import numpy as np

from standing.answers import AnswerSet
from standing.certify import certify, split
from standing.checks import null_calibration, quotient_check
from standing.model import Conditions
from standing.record import Record, verify
from standing.rules import Anchor, decide
from standing.sigma import sigma

FLOOR = 0.25


# --- Step 1: read the answers off the actions, do not invent a list ---------
#
# The consumer has three actions. Group the states the model can describe by the
# action they demand, and the groups ARE the answers. They come out as regions
# because a group is a set of states, which is what keeps sigma from degenerating.

STATES = [round(x, 2) for x in np.arange(0.0, 1.01, 0.01)]


def action_for(occupancy_level: float) -> str:
    if occupancy_level < 0.2:
        return "leave_idle"
    if occupancy_level < 0.6:
        return "keep_warm"
    return "run_full"


ANSWERS = AnswerSet.from_actions(STATES, action_for)

# The quotient is the one condition nothing can check against the world, but it
# CAN be checked against the policy it serves: run the policy over the states and
# see whether it acts differently anywhere inside an answer. Here it cannot, since
# the same function built the set. In a deployment the two are different objects,
# written by different people, and that is where they come apart.
POLICY_CHECK = quotient_check(ANSWERS, action_for, STATES)


# --- Step 2: the forward model, of the DECISION STATISTIC -------------------


class WindowMeanModel:
    """What the pipeline decides on: a mean over k windows, not a raw reading.

    Getting this wrong is the most expensive mistake available. Simulate what the
    pipeline computes, including its window, not the sensor's own distribution.
    """

    def __init__(self, sensor_noise: float = 0.35):
        self.sensor_noise = sensor_noise

    def law(self, answer, conditions: Conditions) -> "WindowMeanLaw":
        windows = int(conditions.get("windows_reporting", 0))
        centre = float(np.mean(sorted(answer.state))) if answer.state else 0.0
        return WindowMeanLaw(centre, self.sensor_noise, windows)


class WindowMeanLaw:
    def __init__(self, centre: float, noise: float, windows: int):
        self.centre, self.noise, self.windows = centre, noise, windows

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if self.windows <= 0:
            # Nothing was in a position to report: no statistic exists. Both
            # answers return nothing, and sigma is zero structurally.
            return np.empty(0)
        draws = rng.normal(self.centre, self.noise, (int(size), self.windows))
        return draws.mean(axis=1)


# The anchor is a SECOND view with a question of its own: a door sensor that can
# tell a shut room from a busy one. It gets a model and a score like anything
# else, which is what makes it usable as an anchor at all.
DOOR_ANSWERS = AnswerSet.from_actions(
    ["shut", "ajar", "busy"], lambda s: "run_full" if s == "busy" else "not_run_full"
)


class DoorModel:
    def law(self, answer, conditions: Conditions) -> "WindowMeanLaw":
        centre = 3.0 if answer.name == "run_full" else 0.0
        return WindowMeanLaw(centre, 0.6, max(1, int(conditions.get("windows_reporting", 0))))


def run(windows: int, label: str) -> None:
    model = WindowMeanModel()
    conditions = Conditions({"windows_reporting": windows, "sensor": "room-1"})

    # --- Step 3: sigma, per view, before any fusion -------------------------
    result = sigma(ANSWERS, model, conditions, draws=6000, rng=0, statistic="window_mean")

    # --- Step 4: certify, choosing the threshold on one half ----------------
    #
    # NOTE, and it matters: the draws below come from the MODEL, so what this
    # certifies is the arithmetic, not the world. A certificate earns its name
    # only on real recorded data, where the rule chosen on one half is scored on
    # the other. Applied to real data a model-fitted rule can overstate badly;
    # treat this step as a demonstration of the split-sample discipline.
    rng = np.random.default_rng(4)
    selection, scoring, certified, certificates = {}, {}, 0.0, {}
    if windows > 0:
        for answer in ANSWERS.answers:
            draws = model.law(answer, conditions).sample(8000, rng)
            selection[answer.name], scoring[answer.name] = split(draws, rng)
        certified, certificates = certify(ANSWERS, selection, scoring, source="model")

    # --- Step 5: charge for drift, which needs TWO sets of conditions -------
    #
    # Scoring an answer against itself under the SAME conditions prices the
    # estimator, not the world: that number shrinks as draws grow. Drift is the
    # same answer under conditions that moved, so it is measured against a second
    # Conditions and does not shrink.
    drifted = Conditions({"windows_reporting": windows, "sensor": "room-2"})
    calibration = (
        null_calibration(model, ANSWERS, conditions, against=drifted, draws=8000, rng=1)
        if windows > 0
        else null_calibration(model, ANSWERS, conditions, draws=8000, rng=1)
    )
    band = calibration.budget_for(max(result.value, calibration.value))

    # --- Step 6: apply the rules --------------------------------------------
    #
    # The anchor's own standing is SCORED, never asserted. A bare number here is
    # refused, because RECRUIT admits a claim that failed its own floor on the
    # strength of this one, and a claim without standing cannot lend standing.
    door_standing = sigma(
        DOOR_ANSWERS, DoorModel(), conditions, draws=6000, rng=2, statistic="door_state"
    )
    # Ruled on the BRACKET, not on the raw sigma, because the record carries the
    # bracket and a third party re-derives the disposition from its low end.
    ruling = decide(
        band,
        floor=FLOOR,
        answers=ANSWERS,
        anchors=[Anchor("door_sensor", "run_full", standing=door_standing, floor=FLOOR)],
        recompute=lambda narrowed: sigma(narrowed, model, conditions, draws=6000, rng=0),
        # Admission needs certificates scored on readings. These were scored on
        # draws from the model's own laws (source="model"), so the ruling that
        # follows is ADMISSIBLE at best: the arithmetic checks, the world was
        # not consulted, as the step-4 note says.
        certificates=certificates if certificates else None,
    )

    print(f"\n=== {label}: {windows} windows reporting ===")
    print(f"answers        {ANSWERS.names}")
    print(f"sigma          {result.value:.4f}   binding {result.binding_pair}")
    if certificates:
        print(f"simulated      {certified:.4f}   (a lower bound on the MODEL's sigma; scored on model samples, admits nothing)")
    print(f"drift null     {calibration.value:.4f}   bracket [{band.low:.4f}, {band.high:.4f}]")
    print(f"rule           {ruling.rule.value.upper()}: {ruling.reason}")
    print(f"quotient       {POLICY_CHECK.reason()}")

    # --- Step 7: hand it across the boundary --------------------------------
    record = Record.build(
        claim="room occupancy",
        interval=label,
        ruling=ruling,
        result=result,
        band=band,
        certificates=certificates,
        conditions=conditions.fields,
        provenance={"example": "toy_pipeline"},
        answer_set=ANSWERS,
        check=POLICY_CHECK,
    )
    problems = verify(Record.from_json(record.to_json()))
    print(f"record         {'verifies with no model and no data' if not problems else problems}")


if __name__ == "__main__":
    for windows, label in ((24, "well observed"), (4, "sparse"), (1, "one window"), (0, "silent")):
        run(windows, label)
    print(
        "\nNote the shape of the answer: standing falls with the evidence base and "
        "goes to exactly zero when nothing reported, where refusing costs nothing "
        "to argue."
    )
