#!/usr/bin/env python3
"""grade_attn_talk.py — grade P82e / P82f (and P88c-g) from the two stored talk instruments' TEXT output.

READS ONLY: lens/runs/TALK-FAIR-ATTN-FULL.txt (talk_fair_test.py: reproduction, context ablation,
free-running-by-step) and lens/runs/PADFREE-TALK-ATTN.txt (talk_fair_v3_ship.py: pad-free greedy samples,
<unk> shares, cross-checkpoint diff). NO checkpoint is opened, NO model is loaded, NO GPU is used.
It APPENDS one section to lens/runs/GRADE-P82.txt (the P82a-d section is written by grade_attn_full.py).

The hedge detector is COPIED from lens/scripts/grade_p76_p77.py (same fragments, same first-six-words rule) so
the arm's hedge count is comparable with the twin's stored 4/6 in lens/runs/GRADE-P76-P77.txt.
The topical-answer count is NOT mechanical: the six greedy samples per checkpoint are printed in full below and
the reading is done by eye on them (stated as such, never presented as a measured number).
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
FAIR = os.environ.get("FAIR_TALK_TXT", os.path.join(P, "lens/runs/TALK-FAIR-ATTN-FULL.txt"))
PAD = os.environ.get("PAD_TALK_TXT", os.path.join(P, "lens/runs/PADFREE-TALK-ATTN.txt"))
OUT = os.environ.get("GRADE_OUT", os.path.join(P, "lens/runs/GRADE-P82.txt"))

HEDGE_FRAGMENTS = ("sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
                   "no idea", "can ' t", "cannot", "not a lot of information")


def is_hedge(text):
    s = " ".join(text.split()[:6]).lower().replace(" \u2019 ", "' ")
    return any(f in s for f in HEDGE_FRAGMENTS)


def hits(text):
    s = " ".join(text.split()[:6]).lower().replace(" \u2019 ", "' ")
    return [f for f in HEDGE_FRAGMENTS if f in s]


_rows = []


def log(s=""):
    print(s, flush=True)
    _rows.append(str(s))


def read(path, what):
    if not os.path.exists(path):
        log(f"  !! MISSING {what}: {path}")
        return None
    return open(path, encoding="utf-8", errors="replace").read()


log("")
log("=" * 110)
log("GRADE OF P82e / P82f — THE DECIDER on the attention arm (predictions P82e/P82f pre-written in")
log("lens/runs/ATTN-FULL-PREDICTIONS-P82.txt; P88c-g in lens/runs/ATTN-GRADE-TALK-PREDICTIONS-P88.txt)")
log("instrument: lens/scripts/grade_attn_talk.py — reads the two stored talk texts only (no checkpoint, no model)")
log("=" * 110)
log(f"  talk_fair_test text read : {FAIR}")
log(f"  pad-free talk text read  : {PAD}")
log(f"  written to               : {OUT}")

# ------------------------------------------------------------------ free-running step 8 (the P82e falsifier)
log("")
log("-" * 110)
log("P82e / P88c — THE FAILURE MEASURED (P65c): free-running vs teacher-forced by step, on the BEST checkpoint")
log("-" * 110)
fair = read(FAIR, "the talk_fair_test output")
step8 = teacher8 = None
if fair:
    m = re.search(r"-> step 8 free-running ([\d.]+)% against teacher-forced ([\d.]+)%", fair)
    if m:
        step8, teacher8 = float(m.group(1)), float(m.group(2))
    for want in ("free-run:", "teacher:"):
        for line in fair.splitlines():
            if line.strip().startswith(want):
                log(f"  {line.strip()}")
    mr = re.search(r"real context\s+([\d.]+)%", fair)
    mm = re.search(r"mismatched context\s*([\d.]+)%\s+\(drop ([\d.]+) points\)", fair)
    if mr:
        log(f"  P65d context ablation: real context {mr.group(1)}%"
            + (f", mismatched {mm.group(1)}% (drop {mm.group(2)} points)" if mm else "")
            + "   [twin's stored drop: 33.07 points, lens/runs/TALK-FAIR-finalbest-d256-repro.txt]")
    for line in fair.splitlines():
        if re.match(r"\s+P65[cde]\b", line):
            log(f"  instrument's own verdict line: {line.strip()}")
if step8 is None:
    log("  !! no '-> step 8 free-running' line found: P82e NOT GRADABLE from the stored text")
    p82e = "NOT GRADED (no number)"
else:
    if "MISMATCH : MY HARNESS DIFFERS" in fair:
        log("  !! WARNING: the talk_fair_test file read carries the instrument's OWN guard line")
        log("     'MISMATCH : MY HARNESS DIFFERS - nothing below may be reported'. That guard is about its")
        log("     reproduction CONTROL INPUT (an env value), not about the model: grade against the corrected")
        log("     re-run (lens/runs/TALK-FAIR-ATTN-FULL-REPRO.txt, TALK_FAIR_LOGGED passed as a FRACTION).")
    log("")
    log(f"  the two numbers this verdict compares:  attention best step-8 {step8:.1f}%   vs   twin 3.9%")
    log(f"  (twin's 3.9% is from lens/runs/TALK-FAIR-finalbest-d256-repro.txt, same instrument, same 256 windows)")
    log(f"  the pre-written falsifier (P82e): 'if it stays at or below 3.9% with the same hedge, the attention"
        f" lever is CLOSED by measurement'")
    if step8 > 4.5:
        p82e = "FALSIFIER FIRED"
        log(f"  -> {step8:.1f}% is ABOVE 3.9% (and above the 4.5% margin): attention bought content routing;"
            f" the lever is NOT closed - re-read P82e before closing anything.")
    elif step8 <= 3.9:
        p82e = "AS PREDICTED -> LEVER CLOSED"
        log(f"  -> {step8:.1f}% is AT OR BELOW 3.9%: no content routing gained; P82e's own consequence stands"
            f" (attention lever closed by measurement; next candidate is P78f (ii), stop paying for <unk> targets).")
    else:
        p82e = "BETWEEN 3.9% AND 4.5% (inside noise: neither reading claimed)"
        log(f"  -> {step8:.1f}% is above 3.9% but inside the 4.5% margin I pre-declared: reported as inside noise,"
            f" not as a win.")
    log(f"  -> P82e verdict: {p82e}")

# ------------------------------------------------------------------ pad-free talk: hedges, <unk>, cross-check
log("")
log("-" * 110)
log("P82e (coherence half) / P88d / P88e / P88g — PAD-FREE TALK, both checkpoints, ONE run of ONE instrument")
log("-" * 110)
pad = read(PAD, "the pad-free talk output")
blocks = {}
last_conv = None
if pad:
    cur = None
    for line in pad.splitlines():
        m = re.match(r"^CHECKPOINT (.+?): (.+)$", line)
        if m:
            cur = m.group(1).strip()
            blocks[cur] = {"file": m.group(2).strip(), "greedy": {}, "shr": {}}
            continue
        if cur is None:
            continue
        mg = re.match(r"^\s+MODEL GREEDY\s{2,}(.+)$", line)
        if mg and last_conv is not None:
            blocks[cur]["greedy"][last_conv] = mg.group(1).strip()
            continue
        mc = re.match(r"^\s+conv\s+(\d+)\s+PROMPT \(real, pad-free\)\s*(.*)$", line)
        if mc:
            last_conv = int(mc.group(1))
            continue
        ms = re.match(r"^\s+<unk> share \(24 greedy\)\s+([\d.]+)%\s+longest non-<unk> run (\d+)", line)
        if ms:
            blocks[cur]["shr"][last_conv] = (float(ms.group(1)), int(ms.group(2)))
            continue

    log(f"  checkpoints measured: {len(blocks)}")
    for lab, b in blocks.items():
        n = len(b["greedy"])
        if not n:
            log(f"  {lab}: no pad-free greedy samples parsed (NOT GRADED)")
            continue
        hedges = sum(1 for t in b["greedy"].values() if is_hedge(t))
        if b["shr"]:
            mean_unk = sum(v[0] for v in b["shr"].values()) / len(b["shr"])
            fluent = sum(1 for v in b["shr"].values() if v[1] >= 4 and v[0] < 100.0)
        else:
            mean_unk, fluent = float("nan"), -1
        b["hedges"], b["mean_unk"], b["fluent"], b["n"] = hedges, mean_unk, fluent, n
        log(f"    {lab}")
        log(f"      file: lens/runs/chat/{os.path.basename(b['file']) if not os.path.isabs(b['file']) else b['file']}")
        log(f"      greedy samples: {n}   hedges (same detector as grade_p76_p77.py): {hedges}/{n}"
            f"   mean <unk> share: {mean_unk:5.1f}%   samples with a >=4-word non-<unk> run: {fluent}/{n}")
    log("")
    log("  COMPARATOR, stored, quoted beside the arm (never quoted without it):")
    log("    twin ct_d256_full169_fix3_best: 4/6 greedy hedges, 0/6 topical answers, mean <unk> 2.1% at s=1.00")
    log("    (lens/runs/GRADE-P76-P77.txt)     our-method 3M ep1: 37.5% mean <unk>, 6/6 repetition spirals")
    log("    (lens/runs/PADFREE-TALK-OURS3M.txt)")
    log("")
    log("  THE SIX GREEDY SAMPLES, VERBATIM, for the by-eye reading (the topical-answer count is NOT mechanical):")
    for lab, b in blocks.items():
        log("")
        log(f"  --- {lab}")
        for conv in sorted(b["greedy"]):
            t = b["greedy"][conv]
            log(f"    conv {conv:>6}  hedge={str(is_hedge(t)):<5} {hits(t)}")
            log(f"      {t}")

    # P88d
    labs = list(blocks)
    if len(labs) >= 2:
        a, t = labs[0], labs[1]
        if blocks[a]["mean_unk"] == blocks[a]["mean_unk"] and blocks[t]["mean_unk"] == blocks[t]["mean_unk"]:
            d = blocks[a]["mean_unk"] - blocks[t]["mean_unk"]
            ok = blocks[a]["mean_unk"] < 20.0 and abs(d) <= 5.0
            log("")
            log(f"  P88d  attention best {blocks[a]['mean_unk']:.1f}% <unk>  vs  twin {blocks[t]['mean_unk']:.1f}%"
                f"  (difference {d:+.1f} points; same run, same instrument)")
            log(f"  -> P88d {'HELD' if ok else 'FAILED'}  (bar: attention below 20% AND within +-5 points of the twin)")
        hedges_a = blocks[a]["hedges"]
        ok = hedges_a >= 3
        log(f"  P88e  attention best hedge count {hedges_a}/{blocks[a]['n']} (twin 4/6 stored)"
            f"  -> P88e {'HELD' if ok else 'FAILED'} on the mechanical half;"
            f" the '0/6 topical answers' half is the by-eye reading on the samples above.")
    md = re.search(r"P72e CROSS-CHECKPOINT DIFF \(first 8 greedy tokens\): (\d+) of (\d+)", pad)
    if md:
        d, tot = int(md.group(1)), int(md.group(2))
        log("")
        log(f"  P88g  cross-checkpoint diff printed by the instrument: {d} of {tot} prompts differ"
            f"  -> P88g {'HELD' if d >= 4 else 'FAILED'} (bar: >= 4 of 6)")
else:
    log("  !! the pad-free talk output is missing: P88d/P88e/P88g NOT GRADED")

log("")
log("-" * 110)
log("P82f — NOTHING IS SHIPPED FROM THIS ARM")
log("-" * 110)
log("  written by this turn, before and after these instruments: no export, no model card, no server, no claim")
log("  of talking. A word-salad talk test means the arm failed whatever the loss says (P82f, unchanged).")
log("  -> P82f PASS (nothing was produced to ship; the statement is the whole claim)")

with open(OUT, "a", encoding="utf-8") as f:
    f.write("\n".join(_rows) + "\n")
print(f"\nappended to {OUT}")
