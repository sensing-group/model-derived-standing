"""The in-home daily summaries: how many slots each observed, and the one-slot day's likelihood.

Expected output (paper v112): 135 summaries; 117 with all 96 slots, 9 with 95, 9 with one slot
(one calendar day across the nine indicators), scored at a likelihood of 0.548.
The gate comparison (eight indicators and the episode kept with the layer, three and none without)
needs the pipeline's daily statistic and the 62-line patch, which this repository does not hold yet.
"""
from collections import Counter

from _common import load, show

rows = load("inhome_days.csv", ["day", "indicator", "slots_observed", "likelihood"])
slots = Counter(int(r["slots_observed"]) for r in rows)
one = [r for r in rows if int(r["slots_observed"]) == 1]
days_one = {r["day"] for r in one}
lik = sorted({round(float(r["likelihood"]), 3) for r in one})
show("daily summaries", [
    ("summaries", 135, len(rows)),
    ("with all 96 slots", 117, slots.get(96, 0)),
    ("with 95 slots", 9, slots.get(95, 0)),
    ("with one slot", 9, len(one)),
    ("calendar days with one slot", 1, len(days_one)),
    ("likelihood scored on the one-slot day", 0.548, ", ".join(map(str, lik)) if lik else "n/a"),
])
print("  gate comparison: requires the in-home daily statistic and the patch; not reproducible from this repository yet")
