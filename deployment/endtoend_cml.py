"""End to end on the deployed CML pipeline: sigma, certificate, record, verify.

Run inside the CML container, from /app:

    docker exec -w /app -e PYTHONPATH=/app:/tmp/standing-lib mt_poc_api \
        python /tmp/standing-lib/endtoend_cml.py

The point is the last step. The record is written from the deployment's own day
statistics, and then checked by `standing.record.verify`, which imports nothing
from the deployment and never sees the model or the data. If that passes, a
consumer on the other side of the boundary can audit the claim.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.certify import certify, split
from standing.drift import bracket
from standing.model import Conditions
from standing.record import Record, verify
from standing.rules import decide
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.sigma import load_standing_config

try:
    import standing_properties as props  # type: ignore
except ModuleNotFoundError:
    from experiments import standing_properties as props  # type: ignore


class CmlLaw:
    def __init__(self, day, answer, base_seed):
        self.day, self.answer, self.base_seed = day, answer, base_seed

    def sample(self, size, rng):
        return cml_channel.simulate_day_statistic(self.day, self.answer, int(size), self.base_seed)


class CmlModel:
    def __init__(self, day, loci, base_seed=0):
        self.day, self.loci, self.base_seed = day, loci, base_seed

    def _cml(self, answer):
        return CmlAnswer(name=answer.name, locus=self.loci[answer.name])

    def law(self, answer, conditions):
        return CmlLaw(self.day, self._cml(answer), self.base_seed)

    def indistinguishable(self, left, right, conditions) -> bool:
        return cml_channel.structurally_identical(self.day, self._cml(left), self._cml(right))


def main() -> int:
    cfg = load_standing_config()
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    answers = AnswerSet(tuple(LibAnswer(n, tuple(sorted(loci[n].items()))) for n in loci))
    day = props.day(96, label="1_full")
    model = CmlModel(day, loci)
    conditions = Conditions({"windows": day.observed_windows, "peak_k": day.peak_k})

    result = lib_sigma(answers, model, conditions, draws=4000, rng=0)
    print(f"sigma          {result.value:.4f}  binding {result.binding_pair}  floor {cfg.sigma_min}")

    # Certificates, on the deployment's own simulated statistics, split-sample.
    # These are draws from the MODEL, so they are labelled source="model" and
    # cannot admit: a certificate on the model's own samples is as wrong as
    # the model. The CML answer set has no physical referent for one of its
    # states, so a certificate on readings is not reachable here at all.
    rng = np.random.default_rng(11)
    selection, scoring = {}, {}
    for answer in answers.answers:
        draws = model.law(answer, conditions).sample(8000, rng)
        selection[answer.name], scoring[answer.name] = split(draws, rng)
    floor_bound, certificates = certify(answers, selection, scoring, source="model")
    print(f"simulated      {floor_bound:.4f}  (a lower bound on the MODEL's sigma; scored on model samples, admits nothing)")
    for pair, certificate in certificates.items():
        print(
            f"  {pair[0]:>18} vs {pair[1]:<18} statistic >= {certificate.threshold:.4f} "
            f"separates at {certificate.value:.4f}"
        )

    # The drift bracket, using one answer against itself as the null.
    null_draws = model.law(answers.answers[0], conditions).sample(8000, rng)
    null_left, null_right = split(null_draws, rng)
    null_pair = certify(
        AnswerSet((LibAnswer("null_a", ("n", 0)), LibAnswer("null_b", ("n", 1)))),
        {"null_a": null_left, "null_b": null_right},
        {"null_a": null_right, "null_b": null_left},
        source="model",
    )[0]
    band = bracket(max(floor_bound, null_pair), null_pair)
    print(f"same-answer null {null_pair:.4f}  ->  bracket [{band.low:.4f}, {band.high:.4f}]")

    # Ruled on the BRACKET, not on the raw sigma. Under drift the claim rests on
    # the low end, and `Record.build` refuses the mismatch: this script ruled on
    # 0.4411 while carrying a bracket whose low end was 0.2736, which is a record
    # contradicting itself. A bracket carries no answer set, so decide() needs
    # `answers` here.
    # And ruled WITH the certificates, which were scored on the deployment's own
    # held-out day statistics: that is what lets the record say ADMITTED rather
    # than admissible, and what a third party re-applies.
    ruling = decide(band, floor=cfg.sigma_min, answers=answers, certificates=certificates)
    record = Record.build(
        claim=f"{day.indicator_name} present on {day.day}",
        interval=str(day.day),
        ruling=ruling,
        result=result,
        band=band,
        certificates=certificates,
        conditions=conditions.fields,
        provenance={"source": "cml-depression-poc", "windows": day.observed_windows},
    )
    print(f"rule           {record.rule}")

    problems = verify(Record.from_json(record.to_json()))
    print(f"\nverify (no model, no data): {'OK' if not problems else problems}")
    print(json.dumps(json.loads(record.to_json())["sigma"], indent=2, sort_keys=True))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
