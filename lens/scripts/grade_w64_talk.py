#!/usr/bin/env python3
"""grade_w64_talk.py — P95d/e/f READ FROM THE TWO TALK OUTPUTS, stored text only (no model, no GPU).

P95d  free-running step-8 above the twin's 3.9%   [lens/runs/TALK-FAIR-finalbest-d256-repro.txt]
P95e  pad-free mean <unk> share at or below the twin's 2.1%
P95f  hedges at or below the twin's 4/6 (counted with the same detector grade_unk_drop_talk.py uses)
P95g  the topical-answer count is BY EYE and is NOT produced here; this file prints the six samples so the
      reader can count them, and it says so instead of pretending the number is mechanical.
Writes lens/runs/GRADE-P95-TALK.txt (APPEND).
"""
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(PROJ)
OUT = "lens/runs/GRADE-P95-TALK.txt"
PADFREE = "lens/runs/PADFREE-TALK-W64.txt"
FAIR = "lens/runs/TALK-FAIR-W64.txt"
HEDGES = ("sorry", "not sure", "don't know", "don't have", "no idea", "can't", "cannot",
          "i'm not", "unable", "afraid", "apolog")
TWIN_UNK = 2.1
TWIN_STEP8 = 3.9
TWIN_HEDGES = 4

lines = []


def w(s=""):
    print(s)
    lines.append(s)


def hedge_of(text):
    first6 = text.split()[:6]
    return [h.replace("'", "’") for h in HEDGES if any(h.replace("'", "’") in t for t in first6)]


w("=" * 94)
w("GRADE P95d/e/f — THE WINDOW-64 TALK NUMBERS (stored text only)")
w("predictions: lens/runs/W64-PREDICTIONS-P95.txt")
w("=" * 94)
if os.path.exists(OUT):
    w("RE-RUN: this file already existed, so this block is APPENDED after an earlier one. The earlier block's")
    w("SAMPLE LISTS are biased (they dropped every sample whose longest word run was a single digit —")
    w("BUG-GRADER-DROPS-SINGLE-DIGIT-RUNS, fixed before this run); its mean <unk> numbers were not affected.")
    w("THE LAST BLOCK IN THIS FILE IS THE AUTHORITATIVE ONE.")
    w("")
w("STORED COMPARATORS, quoted with their files before anything is measured here:")
w(f"  twin (d256 w32) step-8 free-running {TWIN_STEP8}%   [lens/runs/TALK-FAIR-finalbest-d256-repro.txt]")
w(f"  twin (d256 w32) pad-free mean <unk>  {TWIN_UNK}%   [lens/runs/PADFREE-TALK-ATTN.txt]")
w(f"  twin (d256 w32) hedges               {TWIN_HEDGES}/6  [lens/runs/GRADE-P76-P77.txt]")
w("")

# ---------------------------------------------------------------- P95e + P95f: the pad-free blocks
if not os.path.exists(PADFREE):
    w(f"MISSING {PADFREE} -> P95e/P95f not gradable yet.")
else:
    ptext = open(PADFREE, encoding="utf-8", errors="replace").read().splitlines()
    # 2026-10-25 self-wake: THE PARSE IS ANCHORED TO THE SUMMARY SECTION. The old whole-file label pattern
    # (`^\s{2}\S.*held-out`) also matched any two-space indented line containing the word "held-out" — and the
    # per-checkpoint header the repaired instrument prints contains "held-out windows 40,262", so the old
    # pattern would have attributed samples to a header line instead of to the checkpoint's label. If the
    # marker is absent the file is parsed whole, as before.
    _si = [i for i, _ln in enumerate(ptext) if "SHIP SUMMARY" in _ln]
    ptext = ptext[_si[-1]:] if _si else ptext
    cur = None
    blocks = {}
    for ln in ptext:
        if re.match(r"^\s{2}\S.*\(.*by-position|^\s{2}\S.*held-out", ln) and "mean <unk>" not in ln \
                and "CROSS-CHECKPOINT" not in ln and "NOTE (" not in ln:
            cur = ln.strip()
            blocks.setdefault(cur, {"mean": None, "samples": []})
        if "mean <unk> share over 6 greedy" in ln and cur:
            m = re.search(r"mean <unk> share over 6 greedy 24-token samples:\s*([\d.]+)%", ln)
            if m:
                blocks[cur]["mean"] = float(m.group(1))
        # 2026-10-25 self-wake, SECOND GRADER BUG, FOUND AND FIXED THE SAME TURN: this pattern used to read
        # "longest word run (\d+)" with ONE literal space, but the instrument prints the run right-aligned
        # (`{run:>2}`, talk_fair_v3_ship.py), so "run  6" did not match while "run 24" did. It therefore
        # DROPPED EXACTLY THE WORST SAMPLES — the first run of this file reported 4 samples for ARM1 and 3 for
        # ARM2 (both arms have 6) and a hedge denominator of 4 and 3 instead of 6. The mean <unk> line was
        # never affected (it is a different pattern), so no verdict number was wrong, but the sample set the
        # P95g by-eye count is made from WAS BIASED. [BUG-GRADER-DROPS-SINGLE-DIGIT-RUNS]
        m = re.match(r"^\s{6}conv\s+(\d+):\s*<unk>\s*([\d.]+)%\s*longest word run\s+(\d+)\s+(.*)$", ln)
        if m and cur:
            blocks[cur]["samples"].append((m.group(1), float(m.group(2)), int(m.group(3)), m.group(4).strip()))
    w("P95e — THE PAD-FREE <unk> SHARE (bar: at or below the twin's 2.1%)")
    for name, b in blocks.items():
        if b["mean"] is None:
            w(f"  {name}: mean <unk> NOT PARSED (no summary line)")
            continue
        w(f"  {name}: mean <unk> {b['mean']}%   ({len(b['samples'])} samples parsed)")
        for _c, _u, _r, _t in b["samples"]:
            # the TEXT is printed on purpose: P95g is a BY-EYE count and it has to be countable from this file
            w(f"      conv {_c:>6}: <unk> {_u:5.1f}%  longest word run {_r:>2}   {_t[:110]}")
    arms = [(k, v) for k, v in blocks.items() if not k.upper().startswith("TWIN") and v["mean"] is not None]
    for name, b in arms:
        w(f"  verdict for '{name}': {b['mean']}% vs twin {TWIN_UNK}% -> "
          f"{'PASS' if b['mean'] <= TWIN_UNK else 'FAIL'}")
    w("")
    w("P95f — HEDGES in the first six tokens (bar: at or below the twin's 4/6)")
    for name, b in blocks.items():
        if not b["samples"]:
            continue
        hl = [hedge_of(s[3]) for s in b["samples"]]
        n = sum(1 for h in hl if h)
        tag = "twin reference" if name.upper().startswith("TWIN") else "arm"
        w(f"  {name} [{tag}]: hedges {n}/{len(b['samples'])}   ({[h for h in hl if h]})")
        if tag == "arm":
            w(f"    -> {'PASS' if n <= TWIN_HEDGES else 'FAIL'} against the twin's {TWIN_HEDGES}/6")
    w("")
    w("P95g (THE DECIDER) — COHERENCE, WRITTEN SO IT CANNOT BE FAKED: the topical-answer count is BY EYE.")
    w("  The six samples per checkpoint are printed in the block above and in " + PADFREE + ".")
    w("  Nothing in this file counts them; the turn that reads it writes the count into STATE.md, and a")
    w("  word-salad or non-answering sample is a FAIL whatever the loss says (project rule 4).")

# ---------------------------------------------------------------- P95d: step-8 + context
w("")
if not os.path.exists(FAIR):
    w(f"MISSING {FAIR} -> P95d not gradable yet.")
else:
    f = open(FAIR, encoding="utf-8", errors="replace").read()
    m = re.search(r"step 8 free-running ([\d.]+)% against teacher-forced ([\d.]+)%", f)
    wr = re.search(r"real context\s+([\d.]+)%", f)
    mm = re.search(r"mismatched context\s*([\d.]+)%\s*\(drop ([\d.]+) points\)", f)
# 2026-10-25 self-wake: THIS FILE HAD NEVER BEEN EXECUTED. .venv-gpu is Python 3.11.15 and the P95d verdict
# line below put a backslash inside an f-string expression, which is only legal from Python 3.12 — so the
# chained grader died with "SyntaxError: f-string expression part cannot include a backslash" and P95d/e/f/g
# were left unmeasured (lens/runs/TIED-W64.txt 11:46, BUGS.md [BUG-TALK-GRADER-SYNTAX]). The verdict text is
# now built OUTSIDE the f-string. Predictions for this repair: lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt.
    w("P95d — THE THREAD (the point of the experiment): free-running step-8 ABOVE the twin's 3.9%")
    if m:
        fr, tf = float(m.group(1)), float(m.group(2))
        if fr > TWIN_STEP8:
            verdict_d = "PASS"
        else:
            verdict_d = ("FAIL (AT OR BELOW the twin's 3.9% -> THE CONTEXT LEVER IS CLOSED WITH ITS NUMBER)")
        w(f"  measured: step-8 free-running {fr}% against teacher-forced {tf}%   [{FAIR}]")
        w(f"  twin's own step-8, quoted from its file: {TWIN_STEP8}%   [lens/runs/TALK-FAIR-finalbest-d256-repro.txt]")
        w(f"  -> P95d {verdict_d}")
    else:
        w("  NOT PARSED: no 'step 8 free-running ... against teacher-forced ...' line found.")
    if wr and mm:
        w(f"  context ablation: real {wr.group(1)}%  mismatched {mm.group(1)}% (drop {mm.group(2)} points)"
          f"   [twin drop 33.07 points, lens/runs/TALK-FAIR-finalbest-d256-repro.txt]")
w("")
w(f"written to {OUT} (appended)")
open(OUT, "a").write("\n".join(lines) + "\n" + "=" * 94 + "\n")
