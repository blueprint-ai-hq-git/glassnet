#!/usr/bin/env python3
"""THE SELF-CONTEXT ARM'S DECIDER GRADER (P100a-d) — text only. No checkpoint is opened, no model is built, no
GPU is used, no trainer is started. Predictions: lens/runs/SELFCTX-GRADER-PREDICTIONS-P100.txt, written BEFORE
this file existed.

WHAT IT GRADES
  P98e  the exposure-bias lever's own falsifier: step-8 free-running ABOVE the twin's 3.9%.
  P98f  the ship decider: 2+ prompt-relevant answers of six (BY EYE — the grader prints the samples and the
        by-eye field, it does not invent a text metric).

CONTROL FIRST (P100a): the same parsers are run over STORED instrument outputs whose values are already on the
record, and the read-back is printed BESIDE the stored value. If the control does not read back, the arm's
numbers are NOT graded at all.

Sources (read-only):
  arm   : lens/runs/TALK-FAIR-SELFCTX.txt        PADFREE-TALK-SELFCTX.txt
  twin  : lens/runs/TALK-FAIR-SELFCTX-TWIN.txt   (same instrument, same job = the control for P98e)
  stored: lens/runs/TALK-FAIR-finalbest-d256-repro.txt (the 3.9% bar), lens/runs/PADFREE-TALK-ATTN.txt (2.1%)
  control inputs for this grader itself: lens/runs/TALK-FAIR-W64-CONTROL.txt, lens/runs/PADFREE-TALK-ATTN.txt
"""
import os
import re
import sys

P = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # .../GlassNet
RUNS = os.path.join(P, "lens/runs")
BAR_39 = 3.9            # P98e's pre-committed bar: the twin's stored step-8 free-running [TALK-FAIR-finalbest-d256-repro.txt line 101]
TWIN_UNK = 2.1          # the twin's stored pad-free mean <unk> [PADFREE-TALK-ATTN.txt twin block]


def read(name):
    p = os.path.join(RUNS, name)
    if not os.path.exists(p):
        return None
    with open(p, errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------- parsers
def parse_free_run(text):
    """The talk_fair_test.py block '4. FREE-RUNNING vs TEACHER-FORCED by step (P65c)'.
    Returns (free_by_step, teacher_by_step, step8_free, step8_teacher) or None when the block is absent."""
    if not text:
        return None
    i = text.find("FREE-RUNNING vs TEACHER-FORCED")
    if i < 0:
        return None
    seg = text[i:i + 4000]
    m = re.search(r"teacher:\s*((?:[0-9.]+%\s*)+)", seg)
    mt = m.group(1).split() if m else []
    m = re.search(r"free-run:\s*((?:[0-9.]+%\s*)+)", seg)
    mf = m.group(1).split() if m else []
    if not mf:
        return None
    s8 = float(mf[-1].rstrip("%"))
    t8 = float(mt[-1].rstrip("%")) if len(mt) == len(mf) else None
    return ([float(x.rstrip("%")) for x in mf], [float(x.rstrip("%")) for x in mt], s8, t8)


def parse_padfree(text, only_label=None):
    """The talk_fair_v3_ship.py 'SHIP SUMMARY' blocks: one per checkpoint, 2-space indent label then a
    'mean <unk> share over 6 greedy 24-token samples:' line then six 'conv N:' sample lines."""
    if not text:
        return []
    i = text.find("SHIP SUMMARY")
    if i < 0:
        return []
    out, cur = [], None
    for line in text[i:].splitlines():
        m = re.search(r"mean <unk> share over 6 greedy 24-token samples:\s*([0-9.]+)%\s*"
                      r"samples with a >=4-word run and not all-<unk>:\s*(\d+)/(\d+)\s*"
                      r"all-<unk> samples:\s*(\d+)", line)
        if m and cur is not None:
            cur.update(mean_unk=float(m.group(1)), runs=int(m.group(2)), n=int(m.group(3)),
                       allunk=int(m.group(4)))
            out.append(cur)
            cur = None
            continue
        m2 = re.match(r"^  (\S.*?)\s*$", line)
        if m2 and not line.startswith("      ") and "====" not in line and "SHIP SUMMARY" not in line:
            cur = {"label": m2.group(1)}
    if only_label:
        out = [b for b in out if only_label in b["label"]]
    return out


def parse_samples(text, label):
    """The six printed greedy samples of one checkpoint block, for the by-eye count."""
    if not text:
        return []
    i = text.find("SHIP SUMMARY")
    if i < 0:
        return []
    lines = text[i:].splitlines()
    start = None
    for k, line in enumerate(lines):
        if re.match(r"^  (\S.*?)\s*$", line) and label in line and not line.startswith("      "):
            start = k
            break
    if start is None:
        return []
    out = []
    for line in lines[start + 1:]:
        if re.match(r"^  (\S.*?)\s*$", line) and not line.startswith("      ") and "conv" not in line:
            break
        m = re.match(r"^\s+conv\s+(\d+):\s*<unk>\s*([0-9.]+)%\s*longest word run\s*(\d+)\s*(.*)$", line)
        if m:
            out.append(dict(conv=m.group(1), unk=float(m.group(2)), run=int(m.group(3)), text=m.group(4)))
    return out


# ---------------------------------------------------------------- control (P100a) — runs NOW, on stored files
def control():
    print("=" * 100)
    print("PART 1 — P100a CONTROL ON STORED FILES (this runs even before the arm's deciders have written anything)")
    print("=" * 100)
    ok = True

    # (a) the free-running parser, on the stored twin control run of the SAME instrument
    print("\n(a) free-running parser vs lens/runs/TALK-FAIR-W64-CONTROL.txt (the TWIN's stored control run):")
    got = parse_free_run(read("TALK-FAIR-W64-CONTROL.txt"))
    if got is None:
        ok = False
        print("    PARSER FOUND NO BLOCK -> FAIL (a not-found branch is a FAIL, never a printout)")
    else:
        free, teach, s8, t8 = got
        print("    read: free-run by step %s" % ["%.1f%%" % x for x in free])
        print("    read: teacher   by step %s" % ["%.1f%%" % x for x in teach])
        print("    read: step-8 free-running %.1f%%  against the STORED 3.9%%  -> %s"
              % (s8, "MATCH" if abs(s8 - 3.9) < 0.05 else "MISMATCH -> FAIL"))
        print("    read: step-8 teacher-forced %.1f%%  against the STORED 23.0%% -> %s"
              % (t8 if t8 is not None else -1.0,
                 "MATCH" if t8 is not None and abs(t8 - 23.0) < 0.05 else "MISMATCH -> FAIL"))
        ok = ok and abs(s8 - 3.9) < 0.05 and t8 is not None and abs(t8 - 23.0) < 0.05

    # (b) the pad-free parser, on the stored attention/twin instrument output
    print("\n(b) pad-free parser vs lens/runs/PADFREE-TALK-ATTN.txt (both blocks, stored values known):")
    blocks = parse_padfree(read("PADFREE-TALK-ATTN.txt"))
    if not blocks:
        ok = False
        print("    PARSER FOUND NO BLOCK -> FAIL")
    for b in blocks:
        stored = 2.1 if "TWIN" in b["label"].upper() else (6.9 if "ATTENTION BEST" in b["label"].upper() else None)
        print("    read: %-60s mean <unk> %.1f%%  runs %d/%d  all-<unk> %d"
              % (b["label"][:60], b["mean_unk"], b["runs"], b["n"], b["allunk"]))
        if stored is not None:
            print("          against the STORED %.1f%% -> %s" % (stored,
                  "MATCH" if abs(b["mean_unk"] - stored) < 0.05 else "MISMATCH -> FAIL"))
            ok = ok and abs(b["mean_unk"] - stored) < 0.05
    # (c) the sample parser
    print("\n(c) sample parser vs the TWIN block of the same stored file (expect six samples):")
    smp = parse_samples(read("PADFREE-TALK-ATTN.txt"), "TWIN")
    print("    read %d sample(s)" % len(smp))
    for s in smp:
        print("      conv %-6s <unk> %5.1f%%  longest word run %2d  %s" % (s["conv"], s["unk"], s["run"], s["text"][:60]))
    if len(smp) != 6:
        ok = False
        print("    -> FAIL: expected 6 samples, read %d" % len(smp))
    print("\nCONTROL VERDICT: %s" % ("PASS — the parsers reproduce every stored value they were pointed at"
                                     if ok else "FAIL — DO NOT GRADE THE ARM WITH THIS GRADER"))
    return ok


# ---------------------------------------------------------------- the arm's verdicts (P98e/P98f)
def grade():
    print("\n" + "=" * 100)
    print("PART 2 — THE ARM'S VERDICTS (P98e, P98f)")
    print("=" * 100)

    arm_t, tw_t = read("TALK-FAIR-SELFCTX.txt"), read("TALK-FAIR-SELFCTX-TWIN.txt")
    arm_p, twin_p = read("PADFREE-TALK-SELFCTX.txt"), None

    # ---- P98e
    print("\nP98e — THE LEVER'S OWN FALSIFIER (step-8 free-running), pre-committed rule: ABOVE the twin's %.1f%%."
          % BAR_39)
    a = parse_free_run(arm_t)
    t = parse_free_run(tw_t)
    if not arm_t:
        print("  UNMEASURED: lens/runs/TALK-FAIR-SELFCTX.txt does not exist yet (the chained decider 2/3 writes it")
        print("  at the END of the live job). No verdict is printed. The bar it will be graded against: %.1f%%" % BAR_39)
    elif a is None:
        print("  FILE PRESENT BUT NO FREE-RUNNING BLOCK FOUND -> FAIL (not a silent skip). File: %d bytes."
              % len(arm_t))
    else:
        free, teach, s8, t8 = a
        print("  arm  lens/runs/TALK-FAIR-SELFCTX.txt      : step-8 free-running %5.1f%%   (teacher-forced %s)"
              % (s8, ("%.1f%%" % t8) if t8 is not None else "unread"))
        print("  arm  by step: %s" % ["%.1f" % x for x in free])
        ctrl_ok = False
        if t is None:
            print("  twin lens/runs/TALK-FAIR-SELFCTX-TWIN.txt : UNMEASURED -> the arm's column has NO same-run")
            print("       control, so P98e is NOT graded. The stored bar is quoted for context only: %.1f%%." % BAR_39)
        else:
            tf, tt, ts8, tt8 = t
            ctrl_ok = abs(ts8 - BAR_39) < 0.05
            print("  twin lens/runs/TALK-FAIR-SELFCTX-TWIN.txt : step-8 free-running %5.1f%%   (teacher-forced %s)"
                  % (ts8, ("%.1f%%" % tt8) if tt8 is not None else "unread"))
            print("  CONTROL: the twin must reproduce the stored %.1f%% in this run -> %s"
                  % (BAR_39, "PASS" if ctrl_ok else "FAIL (the run's instrument differs from the stored one)"))
            if ctrl_ok:
                print("  P98e VERDICT: %5.1f%% (arm) against %.1f%% (twin) and the stored bar %.1f%% -> %s"
                      % (s8, ts8, BAR_39, "THE EXPOSURE-BIAS LEVER IS CLOSED WITH ITS NUMBER (at or below the bar)"
                         if s8 <= BAR_39 else "ABOVE THE BAR — the lever moved the drift; P98f decides if it matters"))
        print("  P100b's own prediction was that the twin's control reproduces to the digit; the line above is the")
        print("  measurement of that, not the expectation.")

    # ---- P98f
    print("\nP98f — THE SHIP DECIDER (pad-free talk). My pre-stated expectation: 0 or 1 of 6 prompt-relevant.")
    if not arm_p:
        print("  UNMEASURED: lens/runs/PADFREE-TALK-SELFCTX.txt does not exist yet (chained decider 1/3, writes it")
        print("  when the arm's LAST epoch is done). The TWIN's stored control is 2.1% mean <unk> / 6/6 runs.")
    else:
        blocks = parse_padfree(arm_p)
        for b in blocks:
            print("  %-78s mean <unk> %5.1f%%  runs %d/%d  all-<unk> %d"
                  % (b["label"][:78], b["mean_unk"], b["runs"], b["n"], b["allunk"]))
        for b in blocks:
            if "TWIN" in b["label"].upper():
                print("  CONTROL (twin in the same run): must read %.1f%% -> read %.1f%% -> %s"
                      % (TWIN_UNK, b["mean_unk"], "PASS" if abs(b["mean_unk"] - TWIN_UNK) < 0.05 else "FAIL"))
        arm_blocks = [b for b in blocks if "TWIN" not in b["label"].upper()]
        if arm_blocks:
            ab = arm_blocks[0]
            print("  the arm's mean <unk> %.1f%% is printed ABOVE the sample list so the by-eye count below is read"
                  % ab["mean_unk"])
            print("  WITH the <unk> share in view (the project's own rule for this instrument).")
        smp = parse_samples(arm_p, "SELF-CONTEXT") or parse_samples(arm_p, [b["label"] for b in arm_blocks][0]
                                                                   if arm_blocks else "")
        for s in smp:
            print("    conv %-6s <unk> %5.1f%%  run %2d  %s" % (s["conv"], s["unk"], s["run"], s["text"][:90]))
        print("  BY-EYE ANSWER COUNT (a human field, NOT computed here): ___ of 6")
        print("  P98f VERDICT: FAIL unless that field is 2 or more. NOTHING SHIPS until it is.")
    print("\nNO NUMBER ABOVE IS QUOTED WITHOUT ITS FILE. NOTHING IS SHIPPED BY THIS FILE.")


if __name__ == "__main__":
    ok = control()
    grade()
    sys.exit(0 if ok else 1)
