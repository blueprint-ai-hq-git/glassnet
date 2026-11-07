#!/usr/bin/env python3
"""Compare the d512 one-epoch probe (relgrad, gates off) with the headline's epoch 1 (plain SGD, gates off).

Both logs live in one file (lens/runs/PROBE-D512.txt) because two probe attempts were appended to it, so this
script SEGMENTS the log at the "----- probe d512" markers and reports the LAST segment, plus a note on what the
earlier segment was.

What it reports, side by side: VAL, ASST, ppl, epoch seconds, and the cost per step, so the extrapolation to a
6-epoch full-corpus attempt is arithmetic on a measured number rather than a guess. It also prints the four
configuration differences between the d=256 ladder and the headline, because a one-flag story is wrong.

Usage: compare_d512_probe.py [probe-log] [headline-log]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/run/media/yahya/HARD DISK 4/GlassNet")
PROBE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "lens/runs/PROBE-D512.txt"
HEAD = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "lens/runs/TIED-D512-EXT6.txt"

MARK = re.compile(r"----- probe d512 ?:")
EP = re.compile(r"ep (\d)/1\s+loss ([\d.]+)\s+concept ([\d.]+)\s+acc ([\d.]+)%\s+VAL ppl ([\d.]+)\s+acc ([\d.]+)%.*?"
                r"ASST\(by-position\) ppl ([\d.]+)\s+acc ([\d.]+)%")
SECS = re.compile(r"\((\d+)s\)")
WIN = re.compile(r"windows ([\d,]+) train / ([\d,]+) val")
TPP = re.compile(r"tokens per parameter: ([\d.]+)")
WALL = re.compile(r"epoch_wall=(\d+)s")
FINAL = re.compile(r"final VAL ppl ([\d.]+)\s+acc ([\d.]+)%\s+concepts (\d+)/(\d+)")
STEPR = re.compile(r"STEP RULE: (.+)")
METH = re.compile(r"METHOD: (.+)")


def segments(text: str):
    out, cur = [], None
    for line in text.splitlines():
        if MARK.search(line):
            if cur is not None:
                out.append(cur)
            cur = [line]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        out.append(cur)
    return out


def main() -> int:
    try:
        segs = segments(PROBE.read_text(errors="replace"))
    except FileNotFoundError:
        print("probe log missing:", PROBE)
        return 1
    if not segs:
        print("no probe segments found in", PROBE.name)
        return 1
    print("probe log %s -> %d segment(s); reporting the LAST" % (PROBE.name, len(segs)))
    for i, seg in enumerate(segs[:-1], 1):
        body = "\n".join(seg)
        w = WIN.search(body)
        print("  earlier segment %d: windows=%s  %s" % (i, w.group(1) if w else "?",
              "(kept as a cost calibration, not the headline config)" if w and w.group(1) != "16,900,000" else ""))

    body = "\n".join(segs[-1])
    tw = WIN.search(body)
    tp = TPP.search(body)
    st = STEPR.search(body)
    mt = METH.search(body)
    print("\nthis probe")
    print("  windows      : %s train / %s val" % (tw.group(1), tw.group(2)) if tw else "  windows      : not printed yet")
    print("  tokens/param : %s" % (tp.group(1) if tp else "not printed yet"))
    print("  METHOD       : %s" % (mt.group(1)[:80] if mt else "?"))
    print("  STEP RULE    : %s" % (st.group(1)[:80] if st else "?"))
    if st and "opt_lr" in st.group(1):
        print("  NOTE: --lr is NOT the step size under this rule (train_chat.py:983 says so); the rule uses its")
        print("        own calibrated opt_lr. So this is not 'the headline at lr 0.3 with another optimiser' -")
        print("        the step size itself differs, which is part of what the comparison measures.")

    m = EP.search(body)
    if not m:
        print("\nno epoch line yet -> COMPARISON UNDETERMINED (the probe is still training)")
        return 0
    val, ppl, asst = float(m.group(6)), float(m.group(5)), float(m.group(8))
    secs = SECS.search(body[m.end():m.end() + 200])
    secs = int(secs.group(1)) if secs else None
    print("\nprobe epoch 1: VAL %.2f%%  ASST %.2f%%  ppl %.2f%s"
          % (val, asst, ppl, ("  (%d s)" % secs) if secs else ""))

    # headline epoch 1, pinned from its own log line
    htxt = HEAD.read_text(errors="replace") if HEAD.exists() else ""
    hm = re.search(r"ep 1/6.*?VAL ppl ([\d.]+) acc ([\d.]+)%.*?ASST\(by-position\) ppl ([\d.]+) acc ([\d.]+)%", htxt)
    if not hm:
        print("\nheadline epoch-1 line not found in %s" % HEAD.name)
        return 1
    hppl, hval, _, hasst = (float(hm.group(i)) for i in (1, 2, 3, 4))
    print("headline ep 1: VAL %.2f%%  ASST %.2f%%  ppl %.2f   (plain SGD, gates off, 16,504 steps)"
          % (hval, hasst, hppl))
    print("\n  VAL  %+.2f pp   ASST %+.2f pp   ppl %+.2f   (probe minus headline)"
          % (val - hval, asst - hasst, ppl - hppl))

    if secs:
        steps = int(tw.group(1).replace(",", "")) // 1024 if tw else None
        if steps:
            per = secs / steps
            print("\n  cost: %d steps in %d s = %.4f s/step" % (steps, secs, per))
            print("  six epochs at the headline's 16,504 steps/epoch = %.0f s = %.1f h (scaled, not measured)"
                  % (6 * 16504 * per, 6 * 16504 * per / 3600.0))
    fin = FINAL.search(body)
    if fin:
        print("\n  final: VAL %.2f%%  concepts %s/%s" % (float(fin.group(2)), fin.group(3), fin.group(4)))

    print("\nfour differences from the d256 ladder (a one-flag story would be wrong):")
    print("  d=512 vs d=256 | --direct --direct-mode tied vs no direct head | 16.9M vs 20k-2M windows | lr 0.3 vs 0.4")
    print("\nOne epoch is NOT a result and must not be quoted against 37.37%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
