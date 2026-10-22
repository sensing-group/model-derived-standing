"""`standing verify` and `standing report`: audit claims without their model.

`verify` checks that one record replays into its own claim. `report` reads many
records and produces the deployment-level rates, which are the useful output:
not a number per claim but a property of the system. What fraction of issued
verdicts were entitled, what fraction of refusals needed no answer set to
justify, and how much of the budget went to claims with no standing. Those sit
beside yield and coverage, and a system that reports them can be compared with
one that does not.

The record carries certificates and a bracket, and both are arithmetic. So a
consumer who trusts none of the machinery that produced a claim can still check
that the claim replays into itself, in one command, in a process that imports
nothing from the system under audit.

    standing verify record.json
    standing verify record.json --strict     exit non-zero on any discrepancy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

from standing.errors import StandingError
from standing.record import Record, verify

__all__ = ["main"]


def _num(block: object, key: str, spec: str = ".4f") -> str:
    """Render one number out of an UNTRUSTED record, or say it is missing.

    This is the third-party audit entry point: the records it is handed were
    written by someone else, including malformed ones, and its job is to report
    on them rather than to die on them. `verify` was hardened for exactly this
    and the printing was not, so a record with an empty sigma block took the
    whole command down with a KeyError while the discrepancy list that would
    have explained it was already computed and never got printed.
    """
    try:
        return format(float(block[key]), spec)  # type: ignore[index]
    except (KeyError, IndexError, TypeError, ValueError):
        return "?"


def _verify(path: Path, strict: bool) -> int:
    try:
        record = Record.from_json(path.read_text())
    except (OSError, json.JSONDecodeError, StandingError) as exc:
        print(f"{path}: unreadable as a standing record: {exc}", file=sys.stderr)
        return 2

    problems = verify(record)
    print(f"claim      {record.claim}")
    print(f"interval   {record.interval}")
    print(f"answers    {', '.join(str(a) for a in record.answers)}")
    print(
        f"rule       {record.rule}   sigma {_num(record.sigma, 'value')}   "
        f"floor {_num({'f': record.floor}, 'f')}"
    )
    if record.bracket:
        band = record.bracket
        print(
            f"bracket    [{_num(band, 'low')}, {_num(band, 'high')}] from measured "
            f"{_num(band, 'measured')} against null {_num(band, 'null')}"
        )
    # The new blocks are the point of the schema change: a consumer auditing
    # from outside must be able to see that a claim was admitted over a collapse,
    # and to tell "not audited" from "audited and clean".
    if record.quotient is None:
        print("quotient   not audited")
    elif isinstance(record.quotient, Mapping):
        found = record.quotient.get("collapses")
        found = found if isinstance(found, (list, tuple)) else []
        print(
            f"quotient   floor {_num(record.quotient, 'floor')} "
            f"delta {_num(record.quotient, 'delta')}, "
            f"{record.quotient.get('pairs_scored', '?')} pairs, "
            f"{len(found)} collapse(s)"
        )
        for entry in found:
            if not isinstance(entry, Mapping):
                continue
            kind = "structural" if entry.get("structural") else "below floor"
            print(
                f"           {entry.get('left')!r} in {entry.get('left_answer')!r} "
                f"vs {entry.get('right')!r} in {entry.get('right_answer')!r}: "
                f"{_num(entry, 'value')} ({kind}, mass {_num(entry, 'mass', '.2f')})"
            )
    # The partition, and who vouches for it. This is the condition that used to
    # cross the boundary invisibly, so it prints even when there is nothing to
    # report: "declared, not checked" is the fact a reader most needs.
    if record.answer_map is None:
        print("answers     partition not declared")
    elif isinstance(record.answer_map, Mapping):
        kind = record.answer_map.get("kind", "?")
        states = record.answer_map.get("states")
        action = record.answer_map.get("action")
        head = f"answers     {kind}"
        if states is not None:
            head += f" from {states} state(s)"
        if action:
            head += f", map {action}"
        print(head)
        checked = record.answer_map.get("checked")
        if checked is None:
            print("            not checked against a policy")
        elif isinstance(checked, Mapping):
            splits = checked.get("splits")
            splits = splits if isinstance(splits, (list, tuple)) else []
            verdict = "agrees" if checked.get("agrees") else "DISAGREES"
            print(
                f"            policy {verdict} on "
                f"{checked.get('states_checked', '?')} state(s), "
                f"{len(splits)} split(s)"
            )
            for entry in splits[:3]:
                if not isinstance(entry, Mapping):
                    continue
                print(
                    f"            {entry.get('left')!r} and {entry.get('right')!r} "
                    f"are both {entry.get('answer')!r}, but the policy does "
                    f"{entry.get('left_action')!r} and {entry.get('right_action')!r}"
                )

    if record.statistic is None:
        print("statistic  which statistic this is: not declared")
    elif isinstance(record.statistic, Mapping):
        kind = record.statistic.get("kind", "?")
        name = record.statistic.get("name")
        line = f"statistic  {kind}" + (f", {name!r}" if name else "")
        if record.statistic.get("ceiling") is not None:
            line += (f"   ceiling {_num(record.statistic, 'ceiling')}"
                     f"   gap {_num(record.statistic, 'gap')}")
        print(line)

    if isinstance(record.chain, Mapping):
        print(
            f"chain      rank {record.chain.get('rank', '?')}, recruited "
            f"{record.chain.get('recruited')!r} ruling out "
            f"{record.chain.get('ruled_out')!r}"
        )
    if isinstance(record.span, Mapping):
        bits = ", ".join(f"{k}={v}" for k, v in list(record.span.items())[:4])
        print(f"span       {bits}")

    entries = record.certificates if isinstance(record.certificates, (list, tuple)) else ()
    for entry in entries:
        if not isinstance(entry, Mapping):
            print(f"certificate: unreadable, expected an object, found {type(entry).__name__}")
            continue
        raw = entry.get("pair", ())
        pair = " vs ".join(str(x) for x in raw) if isinstance(raw, (list, tuple)) else "?"
        print(
            f"certificate {pair}: rule 'statistic >= {_num(entry, 'threshold')}' "
            f"separates at {_num(entry, 'value')} "
            f"(chosen on {entry.get('n_select', '?')}, scored on "
            f"{entry.get('n_score', '?')})"
        )

    if not problems:
        print("\nOK: every number in this record re-derives from the record itself.")
        return 0
    print(f"\n{len(problems)} discrepancy(ies):", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1 if strict else 0


def _report(paths: list[Path], strict: bool) -> int:
    """Deployment-level rates over many records, model-free."""
    records, unreadable = [], 0
    for path in paths:
        try:
            records.append(Record.from_json(path.read_text()))
        except (OSError, json.JSONDecodeError, StandingError):
            unreadable += 1
    if not records:
        print("no readable records", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for record in records:
        counts[record.rule] = counts.get(record.rule, 0) + 1
    total = len(records)
    # A recruit counts as entitled only when its ruling admitted it; records
    # older than that gate carry no flag and are read the way they were written.
    entitled = sum(
        1 for r in records
        if r.rule == "admitted" or (r.rule == "recruit" and r.admitted is not False)
    )
    admissible = counts.get("admissible", 0)
    certified = sum(1 for r in records if r.certificates)
    bracketed = sum(1 for r in records if r.bracket)
    inconsistent = sum(1 for r in records if verify(r))

    print(f"records                     {total}" + (f"  ({unreadable} unreadable)" if unreadable else ""))
    for rule in ("admissible", "admitted", "recruit", "carry", "abstain"):
        count = counts.get(rule, 0)
        print(f"  {rule:<24}{count:5d}  {100.0 * count / total:5.1f}%")
    print(f"entitled                    {100.0 * entitled / total:5.1f}%")
    print(f"admissible, uncertified     {100.0 * admissible / total:5.1f}%"
          f"   (clears the floor on the model alone, may not be relied on)")
    print(f"refusals needing no floor   {100.0 * counts.get('abstain', 0) / total:5.1f}%"
          f"   (sigma exactly zero, a theorem rather than a policy)")
    print(f"claims carrying a witness   {100.0 * certified / total:5.1f}%")
    print(f"claims charged for drift    {100.0 * bracketed / total:5.1f}%")
    if inconsistent:
        print(f"\nrecords that do not replay into themselves: {inconsistent}", file=sys.stderr)
        return 1 if strict else 0
    print("\nevery record replays into its own claim.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="standing", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("verify", help="re-derive a claim record")
    check.add_argument("record", type=Path)
    check.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when the record does not replay into itself",
    )
    rates = sub.add_parser("report", help="deployment-level rates over many records")
    rates.add_argument("records", type=Path, nargs="+")
    rates.add_argument(
        "--strict", action="store_true", help="exit non-zero if any record fails to replay"
    )
    args = parser.parse_args(argv)
    if args.command == "verify":
        return _verify(args.record, args.strict)
    if args.command == "report":
        return _report(args.records, args.strict)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
