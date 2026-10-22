# Reproducing the paper's numbers

One script per row of the paper's outcomes table. Each reads a plain CSV from `reproduce/data/`, prints the paper's number next to the one it computes, and exits with status 2 and a message naming the missing file and its columns when the data is not there yet. The data files follow with the paper's acceptance; until then the scripts document exactly what will be released and how each number is obtained.

| Paper statement | Script | Data | Expected output |
|---|---|---|---|
| 86% of the false "empty" claims retracted (389 of 452); 65% of the right ones (539 of 830); false "empty" from 90% to 13%, 78% "cannot assess", 77% over the 478 epochs with a claim; baseline validity median 0.997, above 0.99 on 63% of the wrong epochs, a 0.99 threshold retracts 37% | `wifi_episodes.py` | `wifi_episodes.csv` | the table in the script's docstring |
| an ineligible zone cast the deciding vote in all 2,112 contested epochs; its share over 67,821 epochs falls from 41.5% to 17.9% | `wifi_zones.py` | `wifi_zones.csv` | 2112 / 2112, 41.5%, 17.9% |
| a view without data made 94 to 98% of presence decisions for four days | `bfi_delivery.py` | `bfi_daily.csv` | four days with 0% delivery and 94 to 98% share |
| 135 daily summaries, 117 with 96 slots, 9 with 95, 9 with one; the one-slot day scored at 0.548 | `inhome_days.py` | `inhome_days.csv` | those counts and that likelihood |
| 42 pairs, 29 with non-zero standing, 9 verdict flips between the two models, 13 zeros unchanged, magnitudes 0.87 against 0.49 | `inhome_models.py` | `inhome_pairs.csv` | those counts |

The gate comparison (eight indicators and the episode kept, against three and none) needs the in-home pipeline's daily statistic and the 62-line patch, which are not in this repository yet; `inhome_days.py` says so when run.

Run from the repository root with `python3 reproduce/<script>.py`. Only the standard library and numpy are needed.
