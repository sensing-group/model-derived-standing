# Data card

All files are CSV with a header row. Epochs are 10 s; days are calendar days of the synthetic workload. No raw channel state information, audio or location is included; every column is a label, a flag, a count or a derived statistic.

`wifi_episodes.csv`: `episode` (occupied | empty), `epoch` (int), `label` (occupied | empty, ground truth), `claim` (empty | occupied | none; none marks a restart epoch with no reported claim), `registered_motion` (0 | 1, whether some live link registered motion in the epoch), `baseline_validity` (float in [0, 1], the system's own term; empty where not logged).

`wifi_zones.csv`: `epoch` (int), `zone` (id), `eligible` (0 | 1, enough live links), `contested` (0 | 1, the zones disagreed in this epoch), `decided_before` (0 | 1, this zone cast the deciding vote as deployed), `decided_after` (0 | 1, with eligibility enforced).

`bfi_daily.csv`: `day` (YYYY-MM-DD), `bfi_delivery_pct` (float), `presence_decisions` (int), `presence_decisions_bfi` (int, those made by the BFI view).

`inhome_days.csv`: `day` (int), `indicator` (id, nine of them), `slots_observed` (int, of 96), `likelihood` (float, the pipeline's daily likelihood).

`inhome_pairs.csv`: `pattern` (id of the observed-days pattern), `pair` (answer pair), `sigma_mc` (standing under the Monte Carlo model), `sigma_gauss` (under the closed-form model), `floor` (float).

Source: the WiFi files are exported from the occupancy system's epoch logs for the two labelled episodes and the zone and BFI series the paper reports; the in-home files come from the synthetic 15-day workload of one evaluation user. Nothing here carries a clinical claim.
