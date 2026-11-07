#!/usr/bin/env python3
"""Grade seed 2 of the 200k data-scale arm against S1-S3, and report the three-seed mean.

Predictions were written before the run: docs/history/DATASCALE-SEED2-PREDICTIONS.md
  S1  seed-2 VAL inside 15.18-20.18 (the band both existing seeds fell in)
  S2  three-seed mean VAL inside 16.5-20.5, i.e. mean gain +4.3 to +8.3 pp over the two 20k baselines
  S3  seed-2 concept audit >= 30 of 36 (both existing seeds scored 36/36)

Seed 0 and seed 1 are graded by lens/scripts/datascale_grade.py (R1-R3); this file does not re-grade them, it
reads their numbers through the same pinned-and-re-verified method and adds the third draw. Every number is
read from a log line and printed beside its verdict; no threshold was chosen after seeing a number.

Usage: datascale_seed2_grade.py [seed2-log]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
S2_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lens/runs/DATASCALE-HONEST-s2.txt"
S0_LOG = ROOT / "lens/runs/DATASCALE-HONEST.txt"
BASE_LOG = ROOT / "lens/runs/WALL-AB.txt"

# Pinned, with the exact lines; re-verified on every run.
S0 = {"val": 17.43, "line": 230}
S1 = {"val": 19.71, "file": ROOT / "lens/runs/DATASCALE-HONEST-s1.txt", "line": 262}
B20K = {"s0": {"val": 12.18, "line": 116}, "s1": {"val": 12.79, "line": 850}}

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


def parse_last_completed(path: Path) -> dict | None:
    if not path.exists():
        return None
    cur = ep3 = None
    out = None
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = MARK.match(line)
        if m and not any(w in line for w in SKIPWORDS):
            cur, ep3 = int(m.group(1)), None
            continue
        if cur is None:
            continue
        e = EP3.search(line)
        if e:
            ep3 = {"ppl": float(e.group(1)), "val": float(e.group(2)), "asst": float(e.group(4)), "line": n}
            continue
        f = FINAL.search(line)
        if f and ep3 is not None:
            rec = dict(ep3)
            rec["concepts"] = f.group(3)
            rec["seed"] = cur
            out = rec
            ep3 = None
    return out


def main() -> int:
    print("DATASCALE SEED-2 GRADE - source", S2_LOG.name)
    print("----- pins (re-verified every run, not trusted)")
    ok = True
    for name, path, rec in (("seed 0", S0_LOG, S0), ("seed 1", S1["file"], S1),):
        line = line_of(path, rec["line"])
        good = "acc %.2f%%" % rec["val"] in line
        print("  %s VAL %.2f%% on %s:%d -> %s" % (name, rec["val"], path.name, rec["line"],
                                                   "ok" if good else "MISMATCH"))
        ok = ok and good
    for k, rec in B20K.items():
        line = line_of(BASE_LOG, rec["line"])
        good = "acc %.2f%%" % rec["val"] in line
        print("  20k baseline %s VAL %.2f%% on WALL-AB.txt:%d -> %s" % (k, rec["val"], rec["line"],
                                                                       "ok" if good else "MISMATCH"))
        ok = ok and good

    s2 = parse_last_completed(S2_LOG)
    print("\nseed 2: " + ("MISSING / not finished" if not s2 else
          "VAL %.2f%%  ASST %.2f%%  ppl %.2f  concepts %s   [%s:%d]"
          % (s2["val"], s2["asst"], s2["ppl"], s2["concepts"], S2_LOG.name, s2["line"])))
    if not s2:
        print("\nS1/S2/S3 -> UNDETERMINED (no completed seed-2 attempt: needs ep 3/3 and final VAL)")
        return 0 if ok else 1

    print("\nS1 BAND - seed-2 VAL inside 15.18-20.18")
    inside = 15.18 <= s2["val"] <= 20.18
    print("  %.2f%% -> %s the band  -> %s" % (s2["val"], "INSIDE" if inside else "OUTSIDE",
                                              "PASS" if inside else "FAIL"))

    vals = [S0["val"], S1["val"], s2["val"]]
    mean = sum(vals) / len(vals)
    base_mean = sum(b["val"] for b in B20K.values()) / len(B20K)
    print("\nS2 THREE-SEED MEAN - inside 16.5-20.5, i.e. mean gain +4.3 to +8.3 pp")
    print("  values %s -> mean %.3f%%" % (", ".join("%.2f" % v for v in vals), mean))
    print("  20k baselines %s -> mean %.3f%%  -> mean gain %+.2f pp"
          % (", ".join("%.2f" % b["val"] for b in B20K.values()), base_mean, mean - base_mean))
    ok_mean = 16.5 <= mean <= 20.5
    print("  -> %s" % ("PASS" if ok_mean else "FAIL"))

    print("\nS3 CONCEPTS - seed-2 audit >= 30 of 36")
    passed, total = (int(x) for x in s2["concepts"].split("/"))
    print("  %s  (seed 0 and seed 1 both 36/36)  -> %s"
          % (s2["concepts"], "PASS" if passed >= 30 else ("FAIL (concept jump not stable)" if passed < 25 else "WEAK")))

    print("\nNo significance test from three seeds; the values and their mean are what is reported.")
    print("NOTE (2026-11-07 10:1x): S2 above is the PRE-REGISTERED cross-seed form (mean of the two baselines that\n"
          "  existed when it was written). Seed 2 turned out to have a matched 20k baseline after all -\n"
          "  wall_rep_off_s2, VAL 13.40%, WALL-REP.txt:329 - so the number to quote is the within-seed pairing:\n"
          "  %+.2f pp (%.2f - 13.40), and the three paired gains are +5.25 / +6.92 / %+.2f, mean %+.2f pp\n"
          "  (docs/history/DATASCALE-20K-S2-RESULT.md)."
          % (s2["val"] - 13.40, s2["val"], s2["val"] - 13.40, (5.25 + 6.92 + (s2["val"] - 13.40)) / 3))
    print("No talk number, no ship, no 'smart'.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
