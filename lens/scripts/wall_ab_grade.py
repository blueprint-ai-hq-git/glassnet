#!/usr/bin/env python3
"""Grade WALL-AB predictions P1-P5 from the log. Prints the numbers it used beside every verdict.

Rules of this file:
  * numbers are READ from lens/runs/WALL-AB.txt - nothing is computed except the deltas the
    predictions themselves define, and every delta is printed with both inputs;
  * a missing arm yields UNDETERMINED, never a guessed verdict;
  * the predictions are the ones written BEFORE the run (docs/history/WALL-AB-PREDICTIONS.md);
    no threshold here is chosen after seeing a number.

Usage: wall_ab_grade.py [--log PATH]
Exit: 0 always (it is a reporter, not a gate); the verdicts are in the output.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
MARK = re.compile(r"^----- (\S+) seed (\d+) :")
EP3 = re.compile(r"ep 3/3\s+loss ([\d.]+)\s+concept ([\d.]+)\s+acc ([\d.]+)%\s+"
                 r"VAL ppl ([\d.]+)\s+acc ([\d.]+)%")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+/\d+)")

ARMS = ("off", "on", "r64")
SEEDS = (0, 1)


def parse(log: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    arm = seed = None
    for n, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
        m = MARK.match(line)
        if m:
            arm, seed = m.group(1), int(m.group(2))
            continue
        if arm is None:
            continue
        e = EP3.search(line)
        if e:
            out[(arm, seed)] = {
                "train_acc": float(e.group(3)),
                "ppl": float(e.group(4)),
                "acc": float(e.group(5)),
                "ep3_line": n,
            }
            continue
        f = FINAL.search(line)
        if f and (arm, seed) in out:
            out[(arm, seed)].update({"final_acc": float(f.group(2)),
                                     "final_ppl": float(f.group(1)),
                                     "concepts": f.group(3),
                                     "final_line": n})
    return out


def fmt(rec: dict | None) -> str:
    if not rec:
        return "MISSING"
    gap = rec["train_acc"] - rec["acc"]
    return (f"val {rec['acc']:.2f}% ppl {rec['ppl']:.2f} train {rec['train_acc']:.2f}% "
            f"gap {gap:+.2f} concepts {rec.get('concepts', '?')} "
            f"[log :{rec['ep3_line']}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "lens/runs/WALL-AB.txt"))
    a = ap.parse_args()
    log = Path(a.log)
    if not log.exists():
        print(f"no log at {log}")
        return 0
    d = parse(log)
    print("WALL-AB GRADE - source", log)
    print("arm table as parsed (an arm appears only if its epoch-3 line exists):")
    for seed in SEEDS:
        for arm in ARMS:
            print(f"  {arm:4s} s{seed}: {fmt(d.get((arm, seed)))}")

    def need(*keys):
        return all(k in d for k in keys)

    print("\nP1 CONTROL SANITY - wall_on s0/s1 inside 10.0-12.5% final val acc")
    if need(("on", 0), ("on", 1)):
        vals = [d[("on", s)]["acc"] for s in SEEDS]
        ok = all(10.0 <= v <= 12.5 for v in vals)
        print(f"  inputs: on s0 {vals[0]:.2f}%  on s1 {vals[1]:.2f}%  band 10.0-12.5")
        print(f"  -> {'PASS' if ok else 'FAIL'}"
              + ("" if ok else " (block not comparable to the stored rung; no delta may be quoted)"))
    else:
        print("  -> UNDETERMINED (wall_on seed 1 and/or seed 0 has no epoch-3 line yet)")

    print("\nP2 MAIN - wall_off < wall_on on BOTH seeds; predicted delta -1.0 to -6.0 points")
    if need(("off", 0), ("on", 0), ("off", 1), ("on", 1)):
        deltas = []
        for s in SEEDS:
            off, on = d[("off", s)]["acc"], d[("on", s)]["acc"]
            deltas.append(on - off)
            print(f"  s{s}: on {on:.2f}% - off {off:.2f}% = {on - off:+.2f}")
        if all(x < 0.5 for x in deltas):
            print("  -> FALSIFIED: bypass earns under 0.5 pp on both seeds; the lossy-wall premise is "
                  "NOT established and the wider-residual build is cancelled")
        elif all(x > 0 for x in deltas):
            band = all(-6.0 <= x <= -1.0 for x in deltas)
            print(f"  -> PASS direction; magnitude {'inside' if band else 'OUTSIDE'} the -1.0..-6.0 band")
        else:
            print("  -> MIXED: direction not consistent across seeds; report as no effect")
    else:
        print("  -> UNDETERMINED (needs all four of off/on x s0/s1)")

    print("\nP3 CAPACITY DOSE - wall_r64 within 1.0 pp of wall_on on both seeds")
    if need(("r64", 0), ("on", 0), ("r64", 1), ("on", 1)):
        diffs = []
        for s in SEEDS:
            r, on = d[("r64", s)]["acc"], d[("on", s)]["acc"]
            diffs.append(r - on)
            print(f"  s{s}: r64 {r:.2f}% - on {on:.2f}% = {r - on:+.2f}")
        if all(abs(x) <= 1.0 for x in diffs):
            print("  -> PASS: rank 64 matches the full-rank mix; buy rank, not width")
        elif all(x < -1.0 for x in diffs):
            print("  -> FALSIFIED: rank 64 is under-provisioned; bypass capacity itself is the lever")
        else:
            print("  -> MIXED: outside 1.0 pp but not below on both seeds")
    else:
        print("  -> UNDETERMINED (needs r64 and on on both seeds)")

    print("\nP4 OVERFIT DIRECTION - wall_off has the SMALLEST train-minus-val gap at epoch 3")
    if need(*[(arm, s) for arm in ARMS for s in SEEDS]):
        gaps = {arm: [d[(arm, s)]["train_acc"] - d[(arm, s)]["acc"] for s in SEEDS] for arm in ARMS}
        for arm in ARMS:
            print(f"  {arm:4s} gaps s0 {gaps[arm][0]:+.2f} s1 {gaps[arm][1]:+.2f} "
                  f"mean {sum(gaps[arm]) / 2:+.2f}")
        means = {arm: sum(v) / 2 for arm, v in gaps.items()}
        smallest = min(means.items(), key=lambda kv: kv[1])[0]
        print(f"  -> {'PASS' if smallest == 'off' else 'FAIL'}: smallest mean gap is {smallest}")
    else:
        print("  -> UNDETERMINED (needs all six arms)")

    print("\nP5 FALSIFIER AGAINST MY OWN EXPECTATION - wall_off ABOVE wall_on by >1.0 pp on both seeds")
    if need(("off", 0), ("on", 0), ("off", 1), ("on", 1)):
        diffs = [d[("off", s)]["acc"] - d[("on", s)]["acc"] for s in SEEDS]
        print(f"  off - on: s0 {diffs[0]:+.2f}  s1 {diffs[1]:+.2f}")
        if all(x > 1.0 for x in diffs):
            print("  -> CONTRADICTION CONFIRMED: the bypass is harming the concept path; report it, "
                  "do not spin it as noise")
        else:
            print("  -> not triggered")
    else:
        print("  -> UNDETERMINED")

    print("\nNo talk number, no ship, no 'smart'. Only next-token accuracy at this rung is claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
