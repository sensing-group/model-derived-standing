"""Two models of the in-home daily statistic: which verdicts move and which zeros stay.

Expected output (paper v112): 42 pairs over 14 patterns; 29 with non-zero standing, of which 9 change
their admit-or-withhold verdict between the models; 13 pairs with exactly zero standing unchanged;
some magnitudes nearly halved (0.87 against 0.49).
"""
from _common import load, show

rows = load("inhome_pairs.csv", ["pattern", "pair", "sigma_mc", "sigma_gauss", "floor"])
nonzero = [r for r in rows if float(r["sigma_mc"]) > 0.0]
zeros = [r for r in rows if float(r["sigma_mc"]) == 0.0]
flips = [r for r in nonzero if (float(r["sigma_mc"]) >= float(r["floor"])) != (float(r["sigma_gauss"]) >= float(r["floor"]))]
zeros_moved = [r for r in zeros if float(r["sigma_gauss"]) != 0.0]
ratios = [(float(r["sigma_mc"]), float(r["sigma_gauss"])) for r in nonzero if float(r["sigma_mc"]) > 0]
worst = min(ratios, key=lambda p: p[1] / p[0]) if ratios else None
show("two models", [
    ("patterns", 14, len({r["pattern"] for r in rows})),
    ("pairs", 42, len(rows)),
    ("pairs with non-zero standing", 29, len(nonzero)),
    ("verdict flips between the models", 9, len(flips)),
    ("exact zeros", 13, len(zeros)),
    ("exact zeros that moved", 0, len(zeros_moved)),
    ("largest fall in magnitude", "0.87 -> 0.49", f"{worst[0]:.2f} -> {worst[1]:.2f}" if worst else "n/a"),
])
