#!/usr/bin/env python3
"""_probe_gate_counts.py - THROWAWAY probe (2026-10-24 self-wake) used ONLY to understand the shape of the
per-step gate/agreement logs before the real grader (grade_sweep_armA.py) is written, so that grader's
reproduction criterion is not guessed. Reads two jsonl files, prints counts. Writes nothing."""
import json
import sys

for path in sys.argv[1:]:
    n = 0
    refused = accepted = other = 0
    steps = []
    epochs = {}
    maxstep = 0
    keys = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            d = json.loads(line)
            for k in d:
                keys[k] = keys.get(k, 0) + 1
            steps.append(d.get("step"))
            epochs[d.get("epoch")] = epochs.get(d.get("epoch"), 0) + 1
            r = d.get("refused")
            if r is True:
                refused += 1
            elif r is False:
                accepted += 1
            else:
                other += 1
            maxstep = max(maxstep, d.get("step") or 0)
    print(f"--- {path}")
    print(f"    lines {n}   refused {refused}   accepted {accepted}   neither {other}")
    print(f"    epochs {epochs}   max step {maxstep}   step range {steps[0]}..{steps[-1]}")
    # duplicates?
    dup = n - len(set(steps))
    print(f"    duplicate step values: {dup}   keys seen: {sorted(keys)}")
