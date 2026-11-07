#!/usr/bin/env python3
"""Grade a data-ladder arm against thresholds that were fixed before that arm ran.

The ladder at seed 0 (d=256, batch 256, 3 epochs, one flag changing): 12.18% (20k) -> 17.43% (200k) ->
20.81% (500k) -> 2M in flight. Each rung is a separate pre-registration, so each has its own thresholds; they
live in SPECS below and are never edited after a result is known. If a threshold ever needs changing, the change
is a new entry, not a rewrite.

Usage: datascale_ladder_grade.py [arm]        arm in {500k, 2m}; default 2m
Every verdict is printed next to the value it compares, because a right number inside a wrong sentence is the
defect this project keeps finding in its own tooling.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")

SPECS = {
    "500k": {
        "log": ROOT / "lens/runs/DATASCALE-500K.txt",
        "pred": "docs/history/DATASCALE-500K-PREDICTIONS.md",
        # C1 VAL >= 200k + 1.5 pp; C2 ppl < 200k ppl; C3 concepts >= 36; C4 ep3 >= ep2
        "c1_min": 18.93, "c2_max": 207.29, "c3_min": 36, "c4": True,
        "prev": {"name": "seed 0 @ 200k", "val": 17.43, "ppl": 207.29,
                 "file": ROOT / "lens/runs/DATASCALE-HONEST.txt", "line": 230},
    },
    "nogate": {
        "log": ROOT / "lens/runs/DATASCALE-500K-NOGATE.txt",
        "pred": "docs/history/NOGATE-500K-PREDICTIONS.md",
        # G1 VAL >= 21.81 (gated twin + 1.0 pp); G2 gap pair vs the gated twin; G3 mechanical; G4 concepts >= 36
        "c1_min": 21.81, "c2_max": 171.47, "c3_min": 36, "c4": True, "gap_check": True,
        "prev": {"name": "gated 500k @ seed 0", "val": 20.81, "ppl": 171.47, "train_acc": 22.25,
                 "file": ROOT / "lens/runs/DATASCALE-500K.txt", "line": 392},
    },
    "2m": {
        "log": ROOT / "lens/runs/DATASCALE-2M.txt",
        "pred": "docs/history/DATASCALE-2M-PREDICTIONS.md",
        # T1 VAL > 20.81 (strict); T2 lands 21.5-24.0; T3 ppl < 171.47; T4 concepts >= 36
        "c1_min": 20.81, "c1_strict": True, "band": (21.5, 24.0), "c2_max": 171.47, "c3_min": 36, "c4": True,
        "prev": {"name": "seed 0 @ 500k", "val": 20.81, "ppl": 171.47,
                 "file": ROOT / "lens/runs/DATASCALE-500K.txt", "line": 392},
    },
}

EP = re.compile(r"ep (\d)/3\s+loss ([\d.]+)\s+concept ([\d.]+)\s+acc ([\d.]+)%\s+VAL ppl ([\d.]+)\s+acc ([\d.]+)%.*?"
                r"ASST\(by-position\) ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+)/(\d+)")
KEPT = re.compile(r"concepts: (\d+) kept, (\d+) dropped")


def main() -> int:
    key = sys.argv[1] if len(sys.argv) > 1 else "2m"
    if key not in SPECS:
        print("unknown arm %r; known: %s" % (key, ", ".join(SPECS)))
        return 2
    spec = SPECS[key]
    log = Path(sys.argv[2]) if len(sys.argv) > 2 else spec["log"]
    print("LADDER GRADE - arm %s - source %s" % (key, log.name))
    print("  thresholds fixed in %s before the run" % spec["pred"])

    prev = spec["prev"]
    try:
        pline = prev["file"].read_text(errors="replace").splitlines()[prev["line"] - 1]
    except (IndexError, FileNotFoundError):
        pline = ""
    pin_ok = ("acc %.2f%%" % prev["val"]) in pline
    print("  pin %-14s VAL %.2f%% / ppl %.2f on %s:%d -> %s"
          % (prev["name"], prev["val"], prev["ppl"], prev["file"].name, prev["line"],
             "ok" if pin_ok else "MISMATCH"))

    try:
        text = log.read_text(errors="replace")
    except FileNotFoundError:
        print("\nlog missing -> UNDETERMINED")
        return 1
    m = KEPT.search(text)
    kept = int(m.group(1)) if m else None

    eps = {}
    for match in EP.finditer(text):
        eps[int(match.group(1))] = {"ppl": float(match.group(5)), "val": float(match.group(6)),
                                    "asst": float(match.group(8))}
    fin = FINAL.search(text)
    if 3 not in eps or not fin:
        print("\nprogress: epochs seen %s - no epoch 3 yet -> UNDETERMINED" % (sorted(eps) or "none"))
        for n in sorted(eps):
            print("  ep %d  VAL %.2f%%  ppl %.2f" % (n, eps[n]["val"], eps[n]["ppl"]))
        return 0 if pin_ok else 1

    val, ppl, asst = eps[3]["val"], eps[3]["ppl"], eps[3]["asst"]
    cpass, ctot = int(fin.group(3)), int(fin.group(4))
    print("\nresult: VAL %.2f%%  ASST %.2f%%  ppl %.2f  concepts %d/%d  (kept %s)"
          % (val, asst, ppl, cpass, ctot, kept))
    print("paired against %s: %.2f%% -> %.2f%%  (%+.2f pp)"
          % (prev["name"], prev["val"], val, val - prev["val"]))

    c1_ok = (val > spec["c1_min"]) if spec.get("c1_strict") else (val >= spec["c1_min"])
    print("\nC1 VAL %s %.2f%%" % (">" if spec.get("c1_strict") else ">=", spec["c1_min"]))
    print("  %.2f%% -> %s (%+.2f pp vs the predecessor)"
          % (val, "PASS" if c1_ok else "FAIL", val - prev["val"]))
    if "band" in spec:
        lo, hi = spec["band"]
        print("  band %s: %s" % (("%.1f-%.1f" % (lo, hi)), "INSIDE" if lo <= val <= hi else "OUTSIDE"))

    print("\nC2 ppl < %.2f" % spec["c2_max"])
    print("  %.2f -> %s" % (ppl, "PASS" if ppl < spec["c2_max"] else "FAIL"))

    print("\nC3 concepts >= %d" % spec["c3_min"])
    print("  %d/%d -> %s" % (cpass, ctot, "PASS" if cpass >= spec["c3_min"] else "FAIL"))
    if kept is not None and ctot != kept:
        print("  NOTE: denominator %d != this run's kept count %d - report both, do not reconcile" % (ctot, kept))

    if spec.get("c4"):
        print("\nC4 ep3 >= ep2")
        if 2 in eps:
            print("  ep2 %.2f%% -> ep3 %.2f%% -> %s"
                  % (eps[2]["val"], val, "PASS" if val >= eps[2]["val"] else "FAIL"))
        else:
            print("  ep2 missing -> UNDETERMINED")

    if spec.get("gap_check"):
        print("\nG2 OVERFITTING SIGNATURE (train acc minus VAL acc, same epoch)")
        t3 = None
        for match in re.finditer(r"ep 3/3\s+loss [\d.]+\s+concept [\d.]+\s+acc ([\d.]+)%\s+VAL ppl [\d.]+\s+acc ([\d.]+)%", text):
            t3 = (float(match.group(1)), float(match.group(2)))
        if t3 and "train_acc" in prev:
            print("  this arm : train %.2f%% - VAL %.2f%% = gap %.2f pp" % (t3[0], t3[1], t3[0] - t3[1]))
            print("  gated twin: train %.2f%% - VAL %.2f%% = gap %.2f pp"
                  % (prev["train_acc"], prev["val"], prev["train_acc"] - prev["val"]))
            print("  -> this arm overfits %+.2f pp more, while its VAL is %+.2f pp %s"
                  % ((t3[0] - t3[1]) - (prev["train_acc"] - prev["val"]), val - prev["val"],
                     "BETTER" if val > prev["val"] else "worse"))

    ag = len(re.findall(r"agreement step", text))
    gs = len(re.findall(r"^\s+gate step", text, re.M))
    method = ""
    mm = re.search(r"METHOD: (.+)", text)
    if mm:
        method = mm.group(1).strip()
    print("\nG3 MECHANICAL - arm must prove it tested what it claims")
    print("  agreement-step lines=%d  gate-step lines=%d  METHOD=%s" % (ag, gs, method[:70]))
    print("  -> %s" % ("PASS (gates demonstrably absent)" if (ag == 0 and gs == 0) else "FAIL - arm is void"))

    print("\nNo significance test. Headline (37.37%, d=512, full corpus, 6 epochs) unchanged.")
    return 0 if pin_ok else 1


if __name__ == "__main__":
    sys.exit(main())
