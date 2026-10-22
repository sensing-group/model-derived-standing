"""BFI delivery per day and the share of presence decisions the BFI view made.

Expected output (paper v112): four days with no BFI delivery on which the BFI view made 94 to 98% of presence decisions.
"""
from _common import load, show

rows = load("bfi_daily.csv", ["day", "bfi_delivery_pct", "presence_decisions", "presence_decisions_bfi"])
dead = [r for r in rows if float(r["bfi_delivery_pct"]) == 0.0]
shares = [100.0 * int(r["presence_decisions_bfi"]) / int(r["presence_decisions"]) for r in dead if int(r["presence_decisions"])]
show("BFI outage", [
    ("days without BFI delivery", 4, len(dead)),
    ("BFI share of presence decisions on those days", "94-98%", f"{min(shares):.0f}-{max(shares):.0f}%" if shares else "n/a"),
])
for r in dead:
    print(f"  {r['day']}: {r['presence_decisions_bfi']} of {r['presence_decisions']} presence decisions by BFI")
