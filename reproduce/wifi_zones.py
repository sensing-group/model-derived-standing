"""Zone votes: an ineligible zone decided every contested epoch; enforcing eligibility cut its share.

Expected output (paper v112): 2112 of 2112 contested epochs decided by an ineligible zone;
share of zone decisions by ineligible zones over 67,821 epochs 41.5% before, 17.9% after.
"""
from _common import load, show

rows = load("wifi_zones.csv", ["epoch", "zone", "eligible", "contested", "decided_before", "decided_after"])
epochs = {r["epoch"] for r in rows}
contested = {r["epoch"] for r in rows if r["contested"] == "1"}
contested_by_ineligible = {r["epoch"] for r in rows if r["contested"] == "1" and r["decided_before"] == "1" and r["eligible"] == "0"}
before = sum(1 for r in rows if r["decided_before"] == "1" and r["eligible"] == "0")
after = sum(1 for r in rows if r["decided_after"] == "1" and r["eligible"] == "0")
n_before = sum(1 for r in rows if r["decided_before"] == "1")
n_after = sum(1 for r in rows if r["decided_after"] == "1")
show("zone eligibility", [
    ("epochs", 67821, len(epochs)),
    ("contested epochs decided by an ineligible zone", "2112 of 2112", f"{len(contested_by_ineligible)} of {len(contested)}"),
    ("ineligible share of zone decisions, as deployed", "41.5%", f"{100.0 * before / n_before:.1f}%" if n_before else "n/a"),
    ("ineligible share, eligibility enforced", "17.9%", f"{100.0 * after / n_after:.1f}%" if n_after else "n/a"),
])
