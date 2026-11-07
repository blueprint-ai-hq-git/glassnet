#!/usr/bin/env python3
"""Grade the 500k-window arm against C1-C4 (predictions: docs/history/DATASCALE-500K-PREDICTIONS.md).

C1  VAL >= 18.93%  (seed 0's 200k value 17.43% + 1.5 pp)   <=17.43 = more data hurt; 17.43-18.93 = flattened
C2  ppl < 207.29   (seed 0's 200k ppl)
C3  concept audit >= 36 of the kept set (this run keeps 36)
C4  ep3 VAL >= ep2 VAL (monotone path, as in every datascale arm so far)

Every threshold was fixed before the run. The script prints the compared value beside each verdict, because a
right number given a wrong conclusion is the defect this project keeps finding in its own tools.

Usage: datascale_500k_grade.py [log]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lens/runs/DATASCALE-500K.txt"

# Pinned predecessors, re-verified on every run.
S0_20K = {"val": 12.18, "file": ROOT / "lens/runs/WALL-AB.txt", "line": 116}
S0_200K = {"val": 17.43, "ppl": 207.29, "file": ROOT / "lens/runs/DATASCALE-HONEST.txt", "line": 230}

C1_MIN = 18.93      # S0_200K + 1.5 pp
C2_MAX = 207.29     # strictly less than
C3_MIN = 36

EP = re.compile(r"ep (\d)/3\s+loss ([\d.]+)\s+concept ([\d.]+)\s+acc ([\d.]+)%\s+VAL ppl ([\d.]+)\s+acc ([\d.]+)%.*?"
                r"ASST\(by-position\) ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+)/(\d+)")
KEPT = re.compile(r"concepts: (\d+) kept, (\d+) dropped")


def line_of(path: Path, n: int) -> str:
    try:
        return path.read_text(errors="replace").splitlines()[n - 1]
    except (IndexError, FileNotFoundError):
        return ""


def main() -> int:
    print("DATASCALE-500K GRADE - source", LOG.name)
    keep_ok = True
    for name, rec in (("seed 0 @ 20k", S0_20K), ("seed 0 @ 200k", S0_200K)):
        line = line_of(rec["file"], rec["line"])
        good = "acc %.2f%%" % rec["val"] in line
        print("  pin %-12s VAL %.2f%% on %s:%d -> %s" % (name, rec["val"], rec["file"].name, rec["line"],
                                                         "ok" if good else "MISMATCH"))
        keep_ok = keep_ok and good
    kept = dropped = None
    try:
        text = LOG.read_text(errors="replace")
    except FileNotFoundError:
        print("\nlog missing -> UNDETERMINED")
        return 1
    m = KEPT.search(text)
    if m:
        kept, dropped = int(m.group(1)), int(m.group(2))
        print("  this run keeps %d concepts (%d dropped)" % (kept, dropped))

    eps = {}
    for match in EP.finditer(text):
        n = int(match.group(1))
        eps[n] = {"ppl": float(match.group(5)), "val": float(match.group(6)), "asst": float(match.group(8))}
    fin = FINAL.search(text)

    if 3 not in eps or not fin:
        done = sorted(eps)
        print("\nprogress: epochs seen %s - no epoch 3 yet -> C1-C4 UNDETERMINED" % (done or "none"))
        for n in done:
            print("  ep %d  VAL %.2f%%  ppl %.2f" % (n, eps[n]["val"], eps[n]["ppl"]))
        return 0 if keep_ok else 1

    val3, ppl3 = eps[3]["val"], eps[3]["ppl"]
    fin_val, fin_ppl = float(fin.group(2)), float(fin.group(1))
    concepts_passed, concepts_total = int(fin.group(3)), int(fin.group(4))
    print("\nresult: ep3 VAL %.2f%%  ASST %.2f%%  ppl %.2f  final VAL %.2f%%  ppl %.2f  concepts %d/%d"
          % (val3, eps[3]["asst"], ppl3, fin_val, fin_ppl, concepts_passed, concepts_total))
    print("curve at seed 0: 20k %.2f%% -> 200k %.2f%% -> 500k %.2f%%"
          % (S0_20K["val"], S0_200K["val"], val3))

    print("\nC1 VAL >= %.2f%%" % C1_MIN)
    if val3 >= C1_MIN:
        print("  %.2f%% -> +%.2f pp over the 200k arm -> PASS (the lever is still paying)" % (val3, val3 - S0_200K["val"]))
    elif val3 <= S0_200K["val"]:
        print("  %.2f%% -> %+.2f pp vs 200k -> FAIL (more data hurt)" % (val3, val3 - S0_200K["val"]))
    else:
        print("  %.2f%% -> +%.2f pp vs 200k, inside the flattened band -> FAIL as predicted (curve flattened;"
              " 2M is not obviously worth hours)" % (val3, val3 - S0_200K["val"]))

    print("\nC2 ppl < %.2f" % C2_MAX)
    print("  %.2f -> %s" % (ppl3, "PASS" if ppl3 < C2_MAX else "FAIL"))

    print("\nC3 concepts >= %d of the kept set" % C3_MIN)
    print("  %d/%d (kept %s) -> %s" % (concepts_passed, concepts_total, kept, "PASS" if concepts_passed >= C3_MIN else "FAIL"))
    if kept is not None and concepts_total != kept:
        print("  NOTE: the audit denominator (%d) is not this run's kept count (%d) - report both, do not reconcile"
              % (concepts_total, kept))

    print("\nC4 ep3 >= ep2")
    if 2 in eps:
        print("  ep2 %.2f%% -> ep3 %.2f%% -> %s" % (eps[2]["val"], val3, "PASS" if val3 >= eps[2]["val"] else "FAIL"))
    else:
        print("  ep2 missing -> UNDETERMINED")

    print("\nNo significance test, no headline change (37.37%% at d=512, full corpus, 6 epochs stands).")
    return 0 if keep_ok else 1


if __name__ == "__main__":
    sys.exit(main())
