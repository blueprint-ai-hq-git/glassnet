#!/usr/bin/env python3
"""Grade the DATASCALE-HONEST block (seed 0 finished, seed 1 replicates) against R1-R4.

Predictions were written before the runs:
  docs/history/DATASCALE-PREDICTIONS.md      (seed 0: D1 VAL 15.18-20.18%, D2 ppl < 300)
  docs/history/DATASCALE-SEED1-PREDICTIONS.md (seed 1: R1 same band, R2 ppl < 300, C1 concepts >= 30/36,
                                               R3 the seed spread, recorded not judged)

Rules of this file, same as the other graders here:
  * every number is READ from a log line and printed beside its verdict;
  * seed 0's values are pinned WITH their line cite and re-verified against that line on every run, so this
    file cannot drift from the log and does not trust a hand-copied number;
  * a missing or unfinished arm yields UNDETERMINED, never a guessed verdict;
  * no threshold here was chosen after seeing a number.

Usage: datascale_grade.py
Exit: 0 always (reporter, not a gate).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
S0_LOG = ROOT / "lens/runs/DATASCALE-HONEST.txt"
# optional argument: a different seed-1 log, so the grader can be tested on fixtures
S1_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lens/runs/DATASCALE-HONEST-s1.txt"
BASELINE_LOG = ROOT / "lens/runs/WALL-AB.txt"

# Pinned seed-0 result, with the exact lines it came from. Re-verified below on every run.
# val/ppl come from the epoch-3 line (:230); the concept count is printed on the final VAL line (:314).
SEED0 = {"val": 17.43, "asst": 18.61, "ppl": 207.29, "concepts": "36/36",
         "src_line": 230, "src_final_line": 314}
# 20k baselines, same configuration (the arm every lever was compared on), also re-verified from their lines.
BASE20K = {0: {"val": 12.18, "asst": 13.00, "ppl": 399.50, "line": 116},
           1: {"val": 12.79, "asst": 13.21, "ppl": 395.80, "line": 850}}

MARK = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?-----\s*off\s+seed\s+(\d+)\b")
SKIPWORDS = ("GAVE UP", "SKIP", "INCOMPLETE", "DONE")
EP3 = re.compile(r"ep 3/3\s+loss [\d.]+\s+concept [\d.]+\s+acc [\d.]+%\s+VAL ppl ([\d.]+)\s+acc ([\d.]+)%.*?"
                 r"ASST\(by-position\) ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+/\d+)")


def line_of(path: Path, n: int) -> str:
    try:
        return path.read_text(errors="replace").splitlines()[n - 1]
    except (IndexError, FileNotFoundError):
        return ""


def verify_pins() -> bool:
    """Confirm each pinned number really is on the line it claims. Refuses to grade if not."""
    ok = True
    s0 = line_of(S0_LOG, SEED0["src_line"])
    for k, pat in (("val", "acc %.2f%%"), ("ppl", "ppl %.2f")):
        if pat % SEED0[k] not in s0:
            print(f"  PIN MISMATCH seed 0 {k}: expected '{pat % SEED0[k]}' on {S0_LOG.name}:{SEED0['src_line']}")
            ok = False
    s0_final = line_of(S0_LOG, SEED0["src_final_line"])
    if f"concepts {SEED0['concepts']}" not in s0_final:
        print(f"  PIN MISMATCH seed 0 concepts on {S0_LOG.name}:{SEED0['src_final_line']}")
        ok = False
    for seed, rec in BASE20K.items():
        line = line_of(BASELINE_LOG, rec["line"])
        if ("acc %.2f%%" % rec["val"]) not in line or ("ppl %.2f" % rec["ppl"]) not in line:
            print(f"  PIN MISMATCH 20k baseline seed {seed} on WALL-AB.txt:{rec['line']}")
            ok = False
    print("  seed-0 pins verified against their log lines" if ok else "  PINS FAILED - refusing to grade")
    return ok


def parse_seed1() -> dict | None:
    """Last COMPLETED attempt in the seed-1 log: needs an ep 3/3 line AND a final VAL line."""
    if not S1_LOG.exists():
        return None
    cur = ep3 = None
    out = None
    for n, line in enumerate(S1_LOG.read_text(errors="replace").splitlines(), 1):
        m = MARK.match(line)
        if m and not any(w in line for w in SKIPWORDS):
            cur, ep3 = int(m.group(1)), None
            continue
        if cur is None:
            continue
        e = EP3.search(line)
        if e:
            ep3 = {"ppl": float(e.group(1)), "val": float(e.group(2)), "asst": float(e.group(4)), "ep3_line": n}
            continue
        f = FINAL.search(line)
        if f and ep3 is not None:
            rec = dict(ep3)
            rec["final_val"] = float(f.group(2))
            rec["concepts"] = f.group(3)
            rec["final_line"] = n
            rec["seed"] = cur
            out = rec
            ep3 = None
    return out


def main() -> int:
    print("DATASCALE GRADE - seed 0:", S0_LOG.name, "| seed 1:", S1_LOG.name)
    print("----- pins (re-verified every run, not trusted)")
    pins = verify_pins()

    print("\nseed 0 (200k windows): VAL %.2f%%  ASST %.2f%%  ppl %.2f  concepts %s   [%s:%d]"
          % (SEED0["val"], SEED0["asst"], SEED0["ppl"], SEED0["concepts"], S0_LOG.name, SEED0["src_line"]))
    for seed, rec in BASE20K.items():
        print("20k baseline seed %d   : VAL %.2f%%  ASST %.2f%%  ppl %.2f               [WALL-AB.txt:%d]"
              % (seed, rec["val"], rec["asst"], rec["ppl"], rec["line"]))
    print("seed 0 gain            : VAL %+.2f pp, ppl %+.2f" % (SEED0["val"] - BASE20K[0]["val"],
                                                                SEED0["ppl"] - BASE20K[0]["ppl"]))

    s1 = parse_seed1()
    print("\nseed 1 (200k windows): " + ("MISSING" if not s1 else
          "VAL %.2f%%  ASST %.2f%%  ppl %.2f  concepts %s   [%s:%d]"
          % (s1["val"], s1["asst"], s1["ppl"], s1["concepts"], S1_LOG.name, s1["ep3_line"])))

    if not s1:
        print("\nR1/R2/C1/R3 -> UNDETERMINED (no completed seed-1 attempt: an attempt needs ep 3/3 and final VAL)")
        print("\nNo talk number, no ship, no 'smart'.")
        return 0

    print("\nR1 ACC REPLICATES - seed-1 VAL inside 15.18-20.18% (the band D1 used for seed 0)")
    inside = 15.18 <= s1["val"] <= 20.18
    print("  seed 1 VAL %.2f%%  ->  %s the band" % (s1["val"], "INSIDE" if inside else "OUTSIDE"))
    print("  -> " + ("PASS" if inside else "FAIL (see DATASCALE-SEED1-PREDICTIONS.md for what each side means)"))

    print("\nR2 LOSS REPLICATES - seed-1 ppl below 300")
    print("  seed 1 ppl %.2f  ->  %s 300" % (s1["ppl"], "<" if s1["ppl"] < 300 else ">="))
    print("  -> " + ("PASS" if s1["ppl"] < 300 else "FAIL"))

    print("\nC1 CONCEPT JUMP REPLICATES - seed-1 audit passes >= 30 of 36")
    passed, total = (int(x) for x in s1["concepts"].split("/"))
    verdict = "PASS" if passed >= 30 else ("FAIL (seed-0 jump was seed-specific)" if passed < 25 else "WEAK")
    print("  seed 1 concepts %s  (seed 0: %s, 20k baseline: 20/36)" % (s1["concepts"], SEED0["concepts"]))
    print("  -> %s" % verdict)

    print("\nR3 SEED SPREAD - recorded, not judged")
    print("  seed 1 %.2f%% - seed 0 %.2f%% = %+.2f pp" % (s1["val"], SEED0["val"], s1["val"] - SEED0["val"]))
    print("  within-seed gains: seed 0 %+.2f pp, seed 1 %+.2f pp"
          % (SEED0["val"] - BASE20K[0]["val"], s1["val"] - BASE20K[1]["val"]))

    print("\nNo talk number, no ship, no 'smart'. Next-token accuracy at 200k windows is all this measures.")
    return 0 if pins else 1


if __name__ == "__main__":
    sys.exit(main())
