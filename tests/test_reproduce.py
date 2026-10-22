"""The reproduce scripts compute the paper's numbers from data of the documented shape.

Synthetic files are written with exactly the paper's counts, each script runs on them
in strict mode, and a non-zero exit means a script's arithmetic no longer matches the
statement it claims to reproduce. This tests the scripts, not the deployments.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduce"


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("repro")
    # --- the two WiFi episodes ---
    rows, e = [], 0
    def occ(claim, motion, validity):
        nonlocal e
        e += 1
        rows.append(["occupied", e, "occupied", claim, motion, validity])
    for _ in range(48):
        occ("occupied", 1, "")
    for _ in range(22):
        occ("none", 0, 0.95)
    wrong_validity = [0.997] * 285 + [0.95] * (452 - 285 - 22)
    for i in range(430):
        occ("empty", 1 if i < 63 else 0, wrong_validity[i])
    e = 0
    for i in range(1001):
        e += 1
        if i < 171:
            rows.append(["empty", e, "empty", "occupied", 1, ""])
        elif i < 171 + 291:
            rows.append(["empty", e, "empty", "empty", 1, ""])
        else:
            rows.append(["empty", e, "empty", "empty", 0, ""])
    _write(d / "wifi_episodes.csv", ["episode", "epoch", "label", "claim", "registered_motion", "baseline_validity"], rows)
    # --- zones: one deciding zone per epoch ---
    rows = []
    for ep in range(67821):
        before_inel = ep < 28146
        after_inel = ep < 12140
        contested = ep < 2112
        rows.append([ep, "z1", 1, int(contested), int(not before_inel), int(not after_inel)])
        rows.append([ep, "z2", 0, int(contested), int(before_inel), int(after_inel)])
    _write(d / "wifi_zones.csv", ["epoch", "zone", "eligible", "contested", "decided_before", "decided_after"], rows)
    # --- BFI delivery ---
    rows = [[f"2026-08-{day:02d}", 0.0 if day in (8, 9, 10, 11) else 60.0, 1000, {8: 940, 9: 960, 10: 970, 11: 980}.get(day, 5)] for day in range(1, 27)]
    _write(d / "bfi_daily.csv", ["day", "bfi_delivery_pct", "presence_decisions", "presence_decisions_bfi"], rows)
    # --- in-home days: 9 indicators x 15 days ---
    rows = []
    for day in range(1, 16):
        for ind in range(1, 10):
            slots = 1 if day == 10 else (95 if day == 3 else 96)
            lik = 0.548 if day == 10 else 0.8
            rows.append([day, f"i{ind}", slots, lik])
    _write(d / "inhome_days.csv", ["day", "indicator", "slots_observed", "likelihood"], rows)
    # --- in-home pairs: 14 patterns x 3 pairs ---
    rows, k = [], 0
    for pat in range(1, 15):
        for pair in ("present-absent", "present-context", "absent-context"):
            k += 1
            if k <= 13:
                rows.append([pat, pair, 0.0, 0.0, 0.25])
            elif k <= 22:                      # 9 flips: above the floor under MC, below under Gaussian
                rows.append([pat, pair, 0.30, 0.20, 0.25])
            elif k == 23:                      # the largest fall
                rows.append([pat, pair, 0.87, 0.49, 0.25])
            else:
                rows.append([pat, pair, 0.60, 0.55, 0.25])
    _write(d / "inhome_pairs.csv", ["pattern", "pair", "sigma_mc", "sigma_gauss", "floor"], rows)
    return d


@pytest.mark.parametrize("script", ["wifi_episodes", "wifi_zones", "bfi_delivery", "inhome_days", "inhome_models"])
def test_script_reproduces_the_paper_on_data_of_the_documented_shape(data, script):
    env = dict(os.environ, STANDING_REPRODUCE_DATA=str(data), STANDING_REPRODUCE_STRICT="1")
    out = subprocess.run([sys.executable, str(REPRO / f"{script}.py")], capture_output=True, text=True, env=env, cwd=str(REPRO))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "<- differs" not in out.stdout, out.stdout


def test_scripts_refuse_cleanly_without_data(tmp_path):
    env = dict(os.environ, STANDING_REPRODUCE_DATA=str(tmp_path))
    out = subprocess.run([sys.executable, str(REPRO / "wifi_episodes.py")], capture_output=True, text=True, env=env, cwd=str(REPRO))
    assert out.returncode == 2 and "data not released yet" in out.stderr
