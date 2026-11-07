#!/usr/bin/env python3
"""Grade DATASCALE-EP6 (the 6-epoch schedule test at 200k windows, seed 0) against E1-E4.

Predictions were written before the run: docs/history/DATASCALE-EP6-PREDICTIONS.md
  E1  final VAL > 18.43%  (+1.0 pp over the 3-epoch 17.43)   FALSIFIER: <= 17.43
  E2  final ppl < 207.29                                     FALSIFIER: >= 207.29
  E3  17.43 - 18.43 is a PARTIAL result ("small benefit, not worth 2x compute"), not a pass
  E4  a 6-epoch arm is never pooled with the 3-epoch arms

Rules of this file, as with the other graders here: every number is read from a log line and printed beside
its verdict; the 3-epoch baseline is pinned WITH its line cite and re-verified every run; an unfinished arm
yields UNDETERMINED; no threshold here was chosen after seeing a number.

Usage: datascale_ep6_grade.py [log]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lens/runs/DATASCALE-EP6.txt"
BASE_LOG = ROOT / "lens/runs/DATASCALE-HONEST.txt"

# 3-epoch 200k baseline, seed 0: val/ppl on :230, concept count on :314. Re-verified on every run.
BASE = {"val": 17.43, "ppl": 207.29, "line": 230, "final_line": 314}

MARK = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?-----\s*off\s+seed\s+0\b")
SKIPWORDS = ("GAVE UP", "SKIP", "INCOMPLETE", "DONE")
# epochs are now "ep N/6"; capture every epoch so the whole schedule can be printed
EP = re.compile(r"^\s+ep (\d+)/(\d+)\s+loss [\d.]+\s+concept [\d.]+\s+acc [\d.]+%\s+VAL ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+/\d+)")


def line_of(path: Path, n: int) -> str:
    try:
        return path.read_text(errors="replace").splitlines()[n - 1]
    except (IndexError, FileNotFoundError):
        return ""


def main() -> int:
    print("DATASCALE-EP6 GRADE - source", LOG.name)
    print("----- 3-epoch baseline pin (re-verified, not trusted)")
    b = line_of(BASE_LOG, BASE["line"])
    bf = line_of(BASE_LOG, BASE["final_line"])
    ok_pin = ("acc %.2f%%" % BASE["val"] in b and "ppl %.2f" % BASE["ppl"] in b)
    print("  %s:%d -> %s" % (BASE_LOG.name, BASE["line"], b.strip()[:80] if ok_pin else "PIN MISMATCH"))
    print("  %s:%d -> %s" % (BASE_LOG.name, BASE["final_line"], bf.strip()[:60] if bf else "missing"))

    eps: list[tuple[int, float, float]] = []
    final = None
    seen_mark = False
    if LOG.exists():
        for line in LOG.read_text(errors="replace").splitlines():
            if MARK.match(line) and not any(w in line for w in SKIPWORDS):
                seen_mark, eps, final = True, [], None
                continue
            if not seen_mark:
                continue
            m = EP.match(line)
            if m:
                eps.append((int(m.group(1)), float(m.group(4)), float(m.group(3))))   # (epoch, val, ppl)
                continue
            f = FINAL.search(line)
            if f and eps:
                final = {"val": float(f.group(2)), "ppl": float(f.group(1)), "concepts": f.group(3)}

    if eps:
        print("\nepoch path (from the log, this run):")
        for e, val, ppl in eps:
            print("  ep %d  VAL %.2f%%  ppl %.2f" % (e, val, ppl))

    if not final or not eps or eps[-1][0] != 6:
        print("\nE1/E2/E3 -> UNDETERMINED (no completed 6-epoch arm: needs all six epoch lines and a final VAL)")
        print("\nNo talk number, no ship, no 'smart'.")
        return 0

    print("\nfinal VAL %.2f%%  ppl %.2f  concepts %s" % (final["val"], final["ppl"], final["concepts"]))
    print("3-epoch baseline: VAL %.2f%%  ppl %.2f   (delta %+.2f pp, ppl %+.2f)"
          % (BASE["val"], BASE["ppl"], final["val"] - BASE["val"], final["ppl"] - BASE["ppl"]))

    print("\nE1 SCHEDULE GAIN - final VAL above 18.43% (+1.0 pp over the 3-epoch 17.43)")
    v = final["val"]
    if v > 18.43:
        print("  %.2f%% > 18.43%%  -> PASS: the 3-epoch schedule was truncating the data-scale gain" % v)
    elif v <= 17.43:
        print("  %.2f%% <= 17.43%%  -> FAIL: the last three epochs bought nothing; the schedule was not the limit" % v)
    else:
        print("  %.2f%% is inside 17.43-18.43  -> PARTIAL (E3): small benefit, not worth 2x the compute at this"
              " rung; NOT rounded up to E1" % v)

    print("\nE2 LOSS - final ppl below 207.29")
    print("  %.2f %s 207.29  -> %s" % (final["ppl"], "<" if final["ppl"] < 207.29 else ">=",
                                        "PASS" if final["ppl"] < 207.29 else "FAIL"))

    print("\nE4 NOT POOLED - this arm is graded on its own; the 3-epoch result (+5.25 pp seed 0, +6.92 pp seed 1,"
          " mean +6.09) stands as measured and is NOT averaged with a 6-epoch number here.")
    print("\nNo talk number, no ship, no 'smart'. Next-token accuracy at 200k windows is all this measures.")
    return 0 if ok_pin else 1


if __name__ == "__main__":
    sys.exit(main())
