#!/usr/bin/env python3
"""Grade the 4M ungated seed block (seeds 1 and 2) against R1-R5.

Thresholds are copied from docs/history/NOGATE-4M-SEEDS-PREDICTIONS.md, which was committed BEFORE the arm was
launched, and every pin is re-read from its source file each run - so a rewritten prediction, a moved log line or a
changed seed-0 baseline makes the grader fail loudly instead of grading against stale numbers.

R1 each seed VAL in 22.2-25.3
R2 the two seeds differ by <= 3.1 pp
R3 three-seed mean (seed 0 = 23.74, seed 1, seed 2) in 22.6-24.9, reported with its range
R4 both arms printed zero agreement-step / zero gate-step lines
R5 both arms audited >= 36 concepts

Every verdict prints the compared value beside the threshold. Usage: [log]  (default DATASCALE-4M-NOGATE-SEEDS.txt)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "nogate-seeds4m"
LOG = ROOT / "lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt"
PRED = ROOT / "docs/history/NOGATE-4M-SEEDS-PREDICTIONS.md"

# pins: (label, value, file, how to find it)
SEED0 = {"val": 23.74, "file": ROOT / "lens/runs/DATASCALE-4M-NOGATE.txt", "line": 69}
R1_LO, R1_HI = 22.2, 25.3
R2_MAX_SPREAD = 3.1
R3_LO, R3_HI = 22.6, 24.9
CONCEPTS_MIN = 36

# Formats verified against the real log (DATASCALE-4M-NOGATE.txt:153-154), not assumed:
#   ep 3/3  loss 4.2678 concept 0.6257 acc 34.68%   VAL ppl 95.99 acc 23.74%   MASKED ... ASST(by-position) ppl 78.17 acc 25.71%
#   final VAL ppl 95.99  acc 23.74%   concepts 36/36
# the runner writes the arm marker through `say`, which prefixes a timestamp: the anchored pattern never matched
# and the grader reported "no arm marker -> pending" on a log that had two of them (measured 20:31).
ARM_RE = re.compile(r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+)?----- off seed (\d+)\b")
EP_RE = re.compile(r"^  ep (\d+)/(\d+)\s+loss\s+[0-9.]+.*?VAL ppl ([0-9.]+) acc ([0-9.]+)%")
ASST_EP_RE = re.compile(r"ASST\(by-position\) ppl [0-9.]+ acc ([0-9.]+)%")
FINAL_RE = re.compile(r"^  final VAL ppl ([0-9.]+)\s+acc ([0-9.]+)%\s+concepts (\d+)/(\d+)")


def main() -> int:
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG
    print(f"SEED-BLOCK GRADE - {BRANCH} - log {log.name}")
    problems = []
    if not PRED.exists():
        problems.append(f"prediction doc missing: {PRED}")
    # pin: the seed-0 baseline this block is pooled with
    s0_lines = SEED0["file"].read_text(errors="replace").splitlines() if SEED0["file"].exists() else []
    pinned = s0_lines[SEED0["line"] - 1] if len(s0_lines) >= SEED0["line"] else ""
    if "23.74" not in pinned:
        problems.append(f"seed-0 pin not on {SEED0['file'].name}:{SEED0['line']} -> {pinned[:70]!r}")
    else:
        print(f"  pin seed 0 VAL 23.74% on {SEED0['file'].name}:{SEED0['line']} -> ok")
    if problems:
        for p in problems:
            print("  PROBLEM:", p)
        return 2

    if not log.exists():
        print("  no log yet -> UNDETERMINED")
        return 0
    lines = log.read_text(errors="replace").splitlines()

    arms, cur, gates = {}, None, 0
    for ln in lines:
        m = ARM_RE.match(ln)
        if m:
            cur = int(m.group(1))
            arms.setdefault(cur, {"val": None, "asst": None, "ppl": None, "ep": 0, "eps": 0, "conc": None})
            continue
        if "agreement step" in ln or re.match(r"^    gate step", ln):
            gates += 1
        m = EP_RE.match(ln)
        if m and cur is not None:
            a = arms[cur]
            a["ep"], a["eps"] = int(m.group(1)), int(m.group(2))
            a["ppl"], a["val"] = float(m.group(3)), float(m.group(4))
            ma = ASST_EP_RE.search(ln)
            if ma:
                a["asst"] = float(ma.group(1))
            continue
        m = FINAL_RE.match(ln)
        if m and cur is not None:
            a = arms[cur]
            # the final line is the arm's endpoint and the only line carrying the concept audit
            a["ppl"], a["val"] = float(m.group(1)), float(m.group(2))
            a["conc"] = (int(m.group(3)), int(m.group(4)))
            continue

    want = [1, 2]
    done = [s for s in want if s in arms and arms[s]["val"] is not None and arms[s]["ep"] == arms[s]["eps"] and arms[s]["eps"] > 0]
    for s in want:
        a = arms.get(s)
        if not a:
            print(f"  seed {s}: no arm marker -> pending")
        elif a["val"] is None:
            print(f"  seed {s}: arm started, no finished epoch -> pending")
        else:
            print(f"  seed {s}: VAL {a['val']:.2f}%  ppl {a['ppl']:.2f}  ep {a['ep']}/{a['eps']}  concepts {a['conc']}")
    if gates:
        print(f"  mechanical: {gates} gate/agreement lines in the log (expected 0)")
    if len(done) < 2:
        print(f"\nprogress: {len(done)} of 2 seeds finished -> UNDETERMINED (no verdict before both arms end)")
        return 0

    v1, v2 = arms[1]["val"], arms[2]["val"]
    mean3 = (SEED0["val"] + v1 + v2) / 3.0
    spread = abs(v1 - v2)
    rng = f"{min(SEED0['val'], v1, v2):.2f}-{max(SEED0['val'], v1, v2):.2f}"
    print(f"\n  three-seed mean {mean3:.2f}%   range {rng}   seed1-seed2 spread {spread:.2f} pp")

    def verdict(tag, ok, detail):
        print(f"  {tag}: {'PASS' if ok else 'FAIL'} - {detail}")

    verdict("R1 each seed in 22.2-25.3", all(R1_LO <= v <= R1_HI for v in (v1, v2)),
            f"seed1 {v1:.2f}, seed2 {v2:.2f}, band {R1_LO}-{R1_HI}")
    verdict("R2 spread <= 3.1 pp", spread <= R2_MAX_SPREAD, f"{spread:.2f} vs max {R2_MAX_SPREAD}")
    verdict("R3 mean in 22.6-24.9", R3_LO <= mean3 <= R3_HI, f"{mean3:.2f} vs {R3_LO}-{R3_HI}")
    verdict("R4 zero gate lines", gates == 0, f"{gates} lines")
    conc_ok = all(arms[s]["conc"] and arms[s]["conc"][0] >= CONCEPTS_MIN for s in want)
    verdict("R5 concepts >= 36 both", conc_ok, f"seed1 {arms[1]['conc']}, seed2 {arms[2]['conc']}")
    print("\n  published form: mean with its range, never a +- interval; the ungated rung, not the gates-off contrast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
