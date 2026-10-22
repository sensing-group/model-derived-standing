"""Shared loader for the reproduce scripts: a clear message when data is absent."""
import csv
import os
import sys
from pathlib import Path

#: STANDING_REPRODUCE_DATA points the scripts at another data folder (the self-test uses it);
#: STANDING_REPRODUCE_STRICT=1 makes `show` exit 1 when any computed value differs from the paper's.
DATA = Path(os.environ.get("STANDING_REPRODUCE_DATA", Path(__file__).resolve().parent / "data"))


def load(name: str, columns: list[str]) -> list[dict]:
    path = DATA / name
    if not path.exists():
        sys.stderr.write(
            f"data not released yet: expected {path}\n"
            f"columns: {', '.join(columns)}\n"
            f"see reproduce/data/README.md\n"
        )
        sys.exit(2)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in columns if rows and c not in rows[0]]
    if missing:
        sys.stderr.write(f"{path} lacks columns {missing}\n")
        sys.exit(2)
    return rows


def show(title: str, items: list[tuple[str, object, object]]) -> None:
    """Print paper value against computed value, one line per statement."""
    print(title)
    width = max(len(s) for s, _, _ in items)
    print(f"  {'statement'.ljust(width)}  paper      computed")
    bad = 0
    for s, paper, got in items:
        flag = "" if str(paper) == str(got) else "   <- differs"
        bad += bool(flag)
        print(f"  {s.ljust(width)}  {str(paper):<10} {got}{flag}")
    if bad and os.environ.get("STANDING_REPRODUCE_STRICT") == "1":
        sys.exit(1)
