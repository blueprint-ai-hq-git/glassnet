#!/usr/bin/env python3
"""opt_smoke_compare.py -- P42a: is the default step rule UNCHANGED?

Reads lens/runs/OPT-SMOKE.txt, takes the epoch lines ("  ep N/M ...") printed by
  A2  the PRE-CHANGE train_chat.py (lens/runs/_tmp/train_chat_prefix.py, md5 87de1332446b7a654947fdc87ae54ae8)
  B   the CHANGED train_chat.py with the DEFAULT step rule (--opt sgd)
and prints them side by side. The trailing "(NNs)" timing field is stripped before comparing, because
wall-clock is not a result; every other character must match.

It also lists the epoch lines of arm C2 (--opt relgrad on the GPU after the lens/opt.py device fix) and
says whether every number in them is finite (P42b).
"""
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SMOKE = os.path.join(PROJ, "lens", "runs", "OPT-SMOKE.txt")
EP = re.compile(r"^\s*ep \d+/\d+\s")


def blocks(path):
    """Split the log into {marker: [lines]} by the '--- X:' markers."""
    out, cur = {}, None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^--- (\S+):", line)
            if m:
                cur = m.group(1)
                out.setdefault(cur, [])
                continue
            if cur:
                out[cur].append(line.rstrip("\n"))
    return out


def epochs(lines):
    return [l.strip() for l in lines if EP.match(l)]


def strip_time(s):
    return re.sub(r"\s*\(\d+s\)\s*$", "", s).strip()


def finite(line):
    for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line):
        try:
            v = float(tok)
        except ValueError:
            return False
        if v != v or abs(v) == float("inf"):
            return False
    return True


def main():
    if not os.path.exists(SMOKE):
        print(f"no {SMOKE}")
        return
    b = blocks(SMOKE)
    a = epochs(b.get("A2", []))
    bb = epochs(b.get("B", []))
    c = epochs(b.get("C2", []))
    print(f"source: {SMOKE}")
    print(f"  A2 epoch lines: {len(a)}   B epoch lines: {len(bb)}   C2 epoch lines: {len(c)}")
    print()
    print("P42a -- PRE-CHANGE vs CHANGED, default step rule, same seed, same data:")
    print(f"  {'PRE-CHANGE (A2)':<62} {'CHANGED (B)':<62} same")
    n = max(len(a), len(bb))
    diff = 0
    for i in range(n):
        la = strip_time(a[i]) if i < len(a) else "(missing)"
        lb = strip_time(bb[i]) if i < len(bb) else "(missing)"
        same = la == lb
        diff += 0 if same else 1
        print(f"  {la:<62} {lb:<62} {str(same)}")
    if n:
        print(f"  epoch lines compared {n}, DIFFERING {diff}  -> "
              f"{'PASS (the default path is unchanged)' if diff == 0 else 'FAIL (REGRESSION)'}")
        print("  note: the '(NNs)' wall-clock field is stripped before comparing; wall-clock is not a result.")
    print()
    print("P42b -- the relgrad arm on the GPU (C2, after the lens/opt.py device fix):")
    if not c:
        print("  no C2 epoch lines found")
    for line in c:
        print(f"  finite={finite(line)}  {line}")
    print(f"  -> {'PASS (runs, every number finite)' if c and all(finite(x) for x in c) else 'FAIL/UNKNOWN'}")


if __name__ == "__main__":
    main()
