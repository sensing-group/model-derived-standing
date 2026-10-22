"""The two labelled WiFi episodes: what the admission layer retracted and at what price.

Expected output (paper v112):
  occupied episode, 500 epochs: 48 correct "occupied"; 452 false "empty" of which 389 retracted and 63 survived;
    false "empty" 90% -> 13%; 78% cannot assess; 22 restart epochs; 77% retracted over the 478 epochs with a claim;
    baseline validity median 0.997, above 0.99 on 63% of the wrong epochs; a 0.99 threshold retracts 37%
  empty episode, 1,001 epochs: 17% false "occupied" untouched; correct "empty" 83% -> 29%; 54% cannot assess; 291 surviving
  across both: 86% of the wrong "empty" claims retracted (389 of 452), 65% of the right ones (539 of 830)
"""
from statistics import median

from _common import load, show

rows = load("wifi_episodes.csv", ["episode", "epoch", "label", "claim", "registered_motion", "baseline_validity"])
occ = [r for r in rows if r["episode"] == "occupied"]
emp = [r for r in rows if r["episode"] == "empty"]
pct = lambda a, b: f"{100.0 * a / b:.0f}%"

false_empty = [r for r in occ if r["claim"] != "occupied"]           # includes restart epochs with no claim
retracted = [r for r in false_empty if r["registered_motion"] == "0"]
survived = [r for r in false_empty if r["registered_motion"] == "1"]
correct_occ = [r for r in occ if r["claim"] == "occupied"]
restarts = [r for r in occ if r["claim"] == "none"]
with_claim = [r for r in occ if r["claim"] != "none"]
retracted_with_claim = [r for r in retracted if r["claim"] != "none"]
validity = [float(r["baseline_validity"]) for r in false_empty if r["baseline_validity"]]
above = [v for v in validity if v > 0.99]

show("occupied episode", [
    ("epochs", 500, len(occ)),
    ("correct 'occupied' claims", 48, len(correct_occ)),
    ("false 'empty' claims", 452, len(false_empty)),
    ("retracted (quiet epochs)", 389, len(retracted)),
    ("survived (registered motion)", 63, len(survived)),
    ("false 'empty' share as reported", "90%", pct(len(false_empty), len(occ))),
    ("false 'empty' share with the layer", "13%", pct(len(survived), len(occ))),
    ("cannot assess share", "78%", pct(len(retracted), len(occ))),
    ("restart epochs with no claim", 22, len(restarts)),
    ("retracted share over epochs with a claim", "77%", pct(len(retracted_with_claim), len(with_claim)) if with_claim else "n/a"),
    ("baseline validity, median on wrong epochs", 0.997, f"{median(validity):.3f}" if validity else "n/a"),
    ("validity above 0.99 on wrong epochs", "63%", pct(len(above), len(validity)) if validity else "n/a"),
    ("a 0.99 threshold would retract", "37%", pct(len(validity) - len(above), len(validity)) if validity else "n/a"),
])

false_occ = [r for r in emp if r["claim"] == "occupied"]
correct_empty = [r for r in emp if r["claim"] == "empty"]
kept = [r for r in correct_empty if r["registered_motion"] == "1"]
lost = [r for r in correct_empty if r["registered_motion"] == "0"]
show("empty episode", [
    ("epochs", 1001, len(emp)),
    ("false 'occupied' share, untouched", "17%", pct(len(false_occ), len(emp))),
    ("correct 'empty' share as reported", "83%", pct(len(correct_empty), len(emp))),
    ("correct 'empty' share with the layer", "29%", pct(len(kept), len(emp))),
    ("cannot assess share", "54%", pct(len(lost), len(emp))),
    ("surviving correct 'empty' claims", 291, len(kept)),
])
show("across both episodes", [
    ("wrong 'empty' claims retracted", "86% (389 of 452)", f"{pct(len(retracted), len(false_empty))} ({len(retracted)} of {len(false_empty)})"),
    ("right 'empty' claims retracted", "65% (539 of 830)", f"{pct(len(lost), len(correct_empty))} ({len(lost)} of {len(correct_empty)})"),
])
