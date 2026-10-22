"""Differential test: this library against the deployed CML implementation.

Run ON the elitedesk, from the CML repo root, so that CML's own imports resolve:

    PYTHONPATH=.:/path/to/standing/src python3 differential_cml.py

Two implementations computing overlapping quantities is the cheapest bug-finding
instrument available: any disagreement is a defect in one of them, and which one
is a question with an answer rather than an opinion. Nothing here is a unit test
in the library's own terms; it is a comparison against a system that has been
running.
"""

from __future__ import annotations

import sys

import numpy as np

from standing.answers import Answer as LibAnswer
from standing.answers import AnswerSet
from standing.divergence import hellinger2_gaussian, hellinger2_samples
from standing.model import Conditions
from standing.sigma import sigma as lib_sigma

from src.core.standing import channel as cml_channel
from src.core.standing import divergence as cml_divergence
from src.core.standing.answers import Answer as CmlAnswer
from src.core.standing.answers import AnswerSet as CmlAnswerSet
from src.core.standing.sigma import compute_sigma, load_standing_config

TOLERANCE = 0.02  # Monte Carlo noise at 2000 draws is about +/-0.02


class CmlLaw:
    """Wrap the deployed day-statistic simulator as a library Law."""

    def __init__(self, day, answer: CmlAnswer, base_seed: int):
        self.day, self.answer, self.base_seed = day, answer, base_seed

    def sample(self, size: int, rng) -> np.ndarray:
        return cml_channel.simulate_day_statistic(
            self.day, self.answer, int(size), self.base_seed
        )


class CmlModel:
    """A library ForwardModel that delegates to the deployment's own model."""

    def __init__(self, day, loci: dict[str, dict[str, float]], base_seed: int):
        self.day, self.loci, self.base_seed = day, loci, base_seed

    def _cml(self, answer: LibAnswer) -> CmlAnswer:
        return CmlAnswer(name=answer.name, locus=self.loci[answer.name])

    def law(self, answer: LibAnswer, conditions: Conditions) -> CmlLaw:
        return CmlLaw(self.day, self._cml(answer), self.base_seed)

    def indistinguishable(self, left: LibAnswer, right: LibAnswer, conditions) -> bool:
        """Delegate the deployment's own structural check.

        This is the shortcut whose absence the first differential run exposed:
        when no reporting aspect places the two answers at different loci, the
        pair is a structural zero and must read exactly 0, not the estimator's
        upward bias near 0.006.
        """
        return cml_channel.structurally_identical(self.day, self._cml(left), self._cml(right))


def compare_estimators(report: list[str]) -> int:
    """The divergence estimator, both against the Gaussian closed form."""
    rng = np.random.default_rng(2026)
    failures = 0
    report.append(f"{'delta':>6} {'closed form':>12} {'library':>10} {'CML':>10} {'|diff|':>8}")
    for delta in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 12.0):
        truth = hellinger2_gaussian(0.0, 1.0, delta, 1.0)
        lib, old = [], []
        for _ in range(8):
            p = rng.normal(0.0, 1.0, 2000)
            q = rng.normal(delta, 1.0, 2000)
            lib.append(hellinger2_samples(p, q))
            old.append(cml_divergence.hellinger2_from_samples(p, q, 256))
        left, right = float(np.mean(lib)), float(np.mean(old))
        gap = abs(left - right)
        failures += gap > 1e-6
        report.append(f"{delta:6.2f} {truth:12.4f} {left:10.4f} {right:10.4f} {gap:8.4f}")
    return failures


def compare_degenerate(report: list[str]) -> int:
    """The case the library's own suite caught: two disjoint point masses."""
    p, q = np.full(50, 2.0), np.full(50, 9.0)
    left = hellinger2_samples(p, q)
    right = cml_divergence.hellinger2_from_samples(p, q, 256)
    report.append(
        f"two disjoint point masses: library {left:.4f}, CML {right:.4f}, truth 1.0000"
    )
    return int(abs(left - right) > 1e-6)


def compare_sigma(report: list[str], days) -> int:
    """Sigma over the deployment's real answer set and real day conditions."""
    cfg = load_standing_config()
    loci = {a.name: dict(a.locus) for a in cfg.answers.answers}
    lib_answers = AnswerSet(tuple(LibAnswer(n, tuple(sorted(loci[n].items()))) for n in loci))
    report.append(
        f"\nanswers: {lib_answers.names}   floor: {cfg.sigma_min}   days: {len(days)}"
    )
    report.append(f"{'day':>12} {'indicator':>24} {'CML':>9} {'library':>9} {'|diff|':>8}")

    failures = 0
    for day in days:
        cml_value = compute_sigma(day, cfg, cfg.answers).value
        model = CmlModel(day, loci, base_seed=getattr(cfg, "seed", 0))
        try:
            lib_value = lib_sigma(
                lib_answers, model, Conditions({"windows": day.observed_windows}),
                draws=2000, rng=0,
            ).value
        except Exception as exc:  # a refusal is a result, not a crash
            report.append(f"{str(day.day):>12} {day.indicator_name:>24} {cml_value:9.4f}  RAISED  {exc}")
            failures += 1
            continue
        gap = abs(cml_value - lib_value)
        failures += gap > TOLERANCE
        flag = "  <-- differs" if gap > TOLERANCE else ""
        report.append(
            f"{str(day.day):>12} {day.indicator_name:>24} {cml_value:9.4f} "
            f"{lib_value:9.4f} {gap:8.4f}{flag}"
        )
    return failures


def main() -> int:
    report: list[str] = ["== estimator, against the closed form =="]
    failures = compare_estimators(report)
    report.append("")
    failures += compare_degenerate(report)

    try:
        try:
            import standing_properties as props  # type: ignore
        except ModuleNotFoundError:
            from experiments import standing_properties as props  # type: ignore

        days = [
            props.day(96, label="1_full"),
            props.day(48, label="2_half"),
            props.day(14, label="3_gate_edge"),
            props.day(1, label="4_single_window"),
            props.day(96, speech=False, label="5_network_only"),
            props.day(96, network=False, label="6_speech_only"),
            props.day(96, speech=False, network=False, label="7_nothing_live"),
            props.day(0, label="8_silent"),
            props.day(96, live=1, label="9_one_biomarker"),
            props.day(96, live=4, label="10_four_biomarkers"),
        ]
    except Exception as exc:
        days = []
        report.append(f"\nno day builder found ({exc}); sigma comparison skipped")
    if days:
        report.append("\n== sigma, on the deployment's own conditions ==")
        failures += compare_sigma(report, days)

    print("\n".join(report))
    print(f"\ndisagreements: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
