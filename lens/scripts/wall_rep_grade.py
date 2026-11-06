#!/usr/bin/env python3
"""Grade WALL-REP (the replication block) against R1-R4, from the log only.

Predictions were written before the runs: docs/history/WALL-REP-PREDICTIONS.md
  R1 DIRECTION   off above on on both new seeds (off-on > 0 at seeds 2 and 3)
  R2 MAGNITUDE   both new deltas inside +1.0 .. +4.0 pp
  R3 SPREAD      each arm's FOUR draws (the two from WALL-AB.txt plus the two new) span <= 1.5 pp
  R4 NOT POOLED  this block's arms must not appear in the six-arm table (and vice versa)

Rules of this file, same as wall_ab_grade.py:
  * every number is READ from a log line - nothing is estimated, and a delta always prints both inputs;
  * a missing arm yields UNDETERMINED, never a guessed verdict;
  * a KILLED attempt is not a result: an attempt counts only if it has BOTH an `ep 3/3` line and a
    `final VAL` line. The last completed attempt for a tag wins (relaunches happen when a turn kills one);
  * no threshold here was chosen after seeing a number.

Usage: wall_rep_grade.py [--log PATH] [--old-log PATH] [--six-arm PATH]
Exit: 0 always (reporter, not a gate).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")

# Arm markers are written by say(), which prefixes a timestamp, so "-----" is NOT at column 0 in the live
# log: "2026-11-06 18:27:49 ----- off seed 2 attempt 1/2 : ...". The timestamp is optional in the pattern,
# and the failure lines must not be mistaken for markers: the runner writes
# "-----off seed 2 GAVE UP after 2 attempts" and "----- SKIP on seed 2: ...". Measured bug: the first
# version required ^----- and parsed 0 of 4 arms from a log that already had one finished arm.
MARK = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?-----\s*(off|on)\s+seed\s+(\d+)\b")
SKIPWORDS = ("GAVE UP", "SKIP", "INCOMPLETE", "DONE")
EP3 = re.compile(r"ep 3/3\s+loss ([\d.]+)\s+concept ([\d.]+)\s+acc ([\d.]+)%\s+"
                 r"VAL ppl ([\d.]+)\s+acc ([\d.]+)%.*?"
                 r"ASST\(by-position\) ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+/\d+)")

SEEDS = (2, 3)
OLDDRAWS = {  # the six-arm block's values, for R3 only, from lens/runs/WALL-AB.txt
    "off": {0: (12.18, 13.00), 1: (12.79, 13.21)},
    "on":  {0: (10.79, 11.42), 1: (9.68, 10.30)},
}


def parse(path: Path) -> dict[tuple[str, int], dict]:
    """(arm, seed) -> last COMPLETED attempt {val, asst, ppl, ep3_line, final_line}. Killed attempts skipped."""
    out: dict[tuple[str, int], dict] = {}
    cur = None          # (arm, seed) of the current attempt
    ep3 = None          # latest ep3 tuple in this attempt
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = MARK.match(line)
        if m and not any(w in line for w in SKIPWORDS):
            cur, ep3 = (m.group(1), int(m.group(2))), None
            continue
        if cur is None:
            continue
        e = EP3.search(line)
        if e:
            ep3 = {"ppl": float(e.group(4)), "val": float(e.group(5)),
                   "asst": float(e.group(7)), "ep3_line": n}
            continue
        f = FINAL.search(line)
        if f and ep3 is not None:
            rec = dict(ep3)
            rec["final_val"] = float(f.group(2))
            rec["concepts"] = f.group(3)
            rec["final_line"] = n
            out[cur] = rec      # last completed attempt wins
            ep3 = None
    return out


def fmt(rec: dict | None) -> str:
    if not rec:
        return "MISSING"
    return (f"VAL {rec['val']:.2f}%  ASST {rec['asst']:.2f}%  ppl {rec['ppl']:.2f}  "
            f"concepts {rec['concepts']}  [log :{rec['ep3_line']}]")


def six_arm_tags(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {ln.split("\t")[0] for ln in path.read_text().splitlines()[1:] if ln.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "lens/runs/WALL-REP.txt"))
    ap.add_argument("--six-arm", default=str(ROOT / "lens/runs/WALL-AB-RESULTS.tsv"))
    a = ap.parse_args()
    log = Path(a.log)
    if not log.exists():
        print(f"no log at {log}")
        return 0
    d = parse(log)

    print("WALL-REP GRADE - source", log)
    print("arms as parsed (an arm appears only if ONE attempt completed: ep 3/3 AND final VAL):")
    for seed in SEEDS:
        for arm in ("off", "on"):
            print(f"  {arm:3s} s{seed}: {fmt(d.get((arm, seed)))}")
    have = [k for k in ((arm, s) for s in SEEDS for arm in ("off", "on")) if k in d]
    print(f"  arms with a result: {len(have)} of 4")

    def need():
        return all((arm, s) in d for s in SEEDS for arm in ("off", "on"))

    print("\nR1 DIRECTION - off above on on BOTH new seeds (off-on > 0)")
    if need():
        ok = True
        for s in SEEDS:
            off, on = d[("off", s)]["val"], d[("on", s)]["val"]
            print(f"  s{s}: off {off:.2f}% - on {on:.2f}% = {off - on:+.2f}")
            ok = ok and (off - on) > 0
        print(f"  -> {'PASS' if ok else 'FAIL'}: direction "
              f"{'replicates' if ok else 'does NOT replicate on every new seed'}")
    else:
        print("  -> UNDETERMINED (needs all four arms; a killed attempt is not a result)")

    print("\nR2 MAGNITUDE - both new deltas inside +1.0 .. +4.0 pp")
    if need():
        inside = True
        for s in SEEDS:
            dv = d[("off", s)]["val"] - d[("on", s)]["val"]
            within = 1.0 <= dv <= 4.0
            inside = inside and within
            print(f"  s{s}: {dv:+.2f} pp  {'inside' if within else 'OUTSIDE'} 1.0..4.0")
        print(f"  -> {'PASS' if inside else 'FAIL'}")
    else:
        print("  -> UNDETERMINED")

    print("\nR3 SPREAD - each arm's four draws (2 from WALL-AB.txt + 2 new) span <= 1.5 pp")
    if need():
        allok = True
        for arm in ("off", "on"):
            vals = [OLDDRAWS[arm][0][0], OLDDRAWS[arm][1][0], d[(arm, 2)]["val"], d[(arm, 3)]["val"]]
            span = max(vals) - min(vals)
            ok = span <= 1.5
            allok = allok and ok
            print(f"  {arm:3s}: {', '.join(f'{v:.2f}' for v in vals)}  span {span:.2f} pp "
                  f"{'<=' if ok else '>'} 1.5")
        print(f"  -> {'PASS' if allok else 'FAIL'}")
        print("  NOTE: R3 is the one place the two blocks are put side by side, and only as a spread;"
              " no delta is pooled across blocks anywhere in this output.")
    else:
        print("  -> UNDETERMINED")

    print("\nR4 NOT POOLED - this block's tags must not appear in the six-arm table")
    tags = six_arm_tags(Path(a.six_arm))
    leaked = sorted(t for t in tags if "wall_rep" in t)
    print(f"  six-arm table rows: {len(tags)}; rows containing 'wall_rep': {len(leaked)}")
    print(f"  -> {'PASS' if not leaked else 'FAIL: ' + ', '.join(leaked)}")

    print("\nNo talk number, no ship, no 'smart'. Next-token accuracy at this rung is all this measures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
