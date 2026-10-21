#!/usr/bin/env python3
"""Operator mutation over a module, outside string literals, against the suite.

    python tests/mutate_operators.py src/standing/audit.py [src/standing/rules.py ...]

For every comparison, boolean, arithmetic and min/max/all/any/True/False token in
the files given, write the mutant, run the fast tests (and the property tests only
if the fast ones pass), restore the file, and report. A mutant that survives is a
line no test constrains: either pin it with a test or write down why it is
equivalent. AGENTS.md keeps the list of survivors that are equivalent by
construction so the next sweep can skip reading them again.

Restores every file it touches, even on interruption.
"""
from __future__ import annotations

import io
import pathlib
import subprocess
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable
SWAP = {">=": ">", ">": ">=", "<=": "<", "<": "<=", "==": "!=", "!=": "==",
        "and": "or", "or": "and", "min": "max", "max": "min", "not": "", "+": "-", "-": "+",
        "True": "False", "False": "True", "is": "is not", "all": "any", "any": "all"}
FAST = ["tests/test_audit.py", "tests/test_rules.py", "tests/test_record.py"]
SLOW = FAST + ["tests/test_properties.py"]


def failed(tests: list[str]) -> bool:
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode != 0


def sites(src: str):
    lines = src.splitlines(keepends=True)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, t in enumerate(toks):
        if t.type not in (tokenize.OP, tokenize.NAME) or t.string not in SWAP:
            continue
        row = t.start[0] - 1
        line = lines[row]
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith('"""') or "raise " in line:
            continue
        if t.string == "not" and i > 0 and toks[i - 1].string == "is":
            continue
        if t.string == "is" and i + 1 < len(toks) and toks[i + 1].string == "not":
            continue
        if t.string in ("+", "-") and toks[i - 1].string in ("(", ",", "="):
            continue
        yield row, t.start[1], t.end[1], t.string


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    total = killed = 0
    survivors = []
    for rel in argv:
        path = ROOT / rel
        src = path.read_text()
        try:
            for row, c0, c1, tok in sites(src):
                lines = src.splitlines(keepends=True)
                line = lines[row]
                mutant = line[:c0] + SWAP[tok] + line[c1:]
                if mutant == line:
                    continue
                lines[row] = mutant
                path.write_text("".join(lines))
                total += 1
                dead = failed(FAST) or failed(SLOW)
                path.write_text(src)
                killed += dead
                if not dead:
                    survivors.append(f"{rel}:{row + 1} {tok!r}->{SWAP[tok]!r}  {line.strip()[:100]}")
                print(f"{'killed  ' if dead else 'SURVIVED'} {rel}:{row + 1} {tok!r}->{SWAP[tok]!r}", flush=True)
        finally:
            path.write_text(src)
    print(f"\n{killed} killed of {total}, {len(survivors)} survived")
    for s in survivors:
        print("  ", s)
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
