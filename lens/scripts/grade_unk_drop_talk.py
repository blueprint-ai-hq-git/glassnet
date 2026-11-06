#!/usr/bin/env python3
"""grade_unk_drop_talk.py — P89f for the --drop-unk arm, FROM THE TWO STORED TALK TEXTS ONLY.
No checkpoint is opened, no model is built, no GPU, no training. Predictions (including the correction and the
pre-declared SPIRAL detector): lens/runs/UNK-DROP-PREDICTIONS-P89.txt. Written 2026-10-24 by the self-wake.

Every number printed here is printed BESIDE the file it came from, and every comparison prints both sides —
this project's verdict-check rule.
"""
import os
import re
import sys

P = "/run/media/yahya/HARD DISK 4/GlassNet"
# ENV OVERRIDES exist so this instrument can be DRY-RUN against a stored pair (the attention arm's pad-free
# files) to prove its parsers work BEFORE the arm it is meant for produces anything. Defaults are the real files.
PADFREE = os.environ.get("UNKDROP_PADFREE", os.path.join(P, "lens/runs/PADFREE-TALK-UNKDROP.txt"))
FAIR = os.environ.get("UNKDROP_FAIR", os.path.join(P, "lens/runs/TALK-FAIR-UNKDROP.txt"))
OUT = os.environ.get("UNKDROP_GRADE_OUT", os.path.join(P, "lens/runs/GRADE-P89.txt"))

# stored comparator numbers, each with its file (quoted, never from memory)
REF = {
    "twin pad-free mean <unk>": ("2.1%", "lens/runs/PADFREE-TALK-ATTN.txt"),
    "twin step-8 free-running": ("3.9%", "lens/runs/TALK-FAIR-finalbest-d256-repro.txt"),
    "twin context drop": ("33.07 points", "lens/runs/TALK-FAIR-finalbest-d256-repro.txt"),
    "twin talk pattern": ("4/6 greedy hedges, 0/6 topical answers", "lens/runs/GRADE-P76-P77.txt"),
    "attention arm step-8": ("2.7%", "lens/runs/TALK-FAIR-ATTN-FULL.txt"),
    "attention arm conv 10955": ("33.3% <unk>, longest non-<unk> run 3 = SPIRAL", "lens/runs/PADFREE-TALK-ATTN.txt"),
    "our-method 3M ep1 pad-free": ("37.5% mean <unk>, 6/6 repetition spirals", "lens/runs/PADFREE-TALK-OURS3M.txt"),
}
HEDGES = ("sorry", "not sure", "don't know", "don't have", "no idea", "can't", "cannot",
          "not a lot of information")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a")
    def w(self, s=""):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()
    def close(self):
        self.f.close()


def parse_padfree(txt):
    """-> list of (label, mean_unk, samples) where samples = [(conv, unk_pct, run, text)]"""
    out, cur = [], None
    lines = txt.splitlines()
    for i, ln in enumerate(lines):
        if "mean <unk> share over 6 greedy" in ln:
            lab = ""
            for j in range(i - 1, max(i - 5, -1), -1):
                if lines[j].strip() and lines[j].startswith("  ") and not lines[j].startswith("    "):
                    lab = lines[j].strip()
                    break
            m = re.search(r"mean <unk> share over 6 greedy 24-token samples:\s*([\d.]+)%", ln)
            cur = (lab, float(m.group(1)) if m else None, [])
            out.append(cur)
            continue
        m = re.match(r"\s+conv\s+(\d+):\s+<unk>\s+([\d.]+)%\s+longest word run\s+(\d+)\s+(.*)$", ln)
        if m and cur is not None:
            cur[2].append((int(m.group(1)), float(m.group(2)), int(m.group(3)), m.group(4).strip()))
    return out


def hedge_of(text):
    first6 = text.lower().replace("'", "’").split()[:6]
    return [h.replace("'", "’") for h in HEDGES if any(h.replace("'", "’") in w for w in first6)]


def main():
    T = Tee(OUT)
    T.w()
    T.w("=" * 94)
    T.w("GRADE OF P89f — THE DECIDER ON THE --drop-unk ARM (stored talk texts only; no model, no GPU)")
    T.w("predictions: lens/runs/UNK-DROP-PREDICTIONS-P89.txt (P89f(i)(ii)(iii) + the ADDENDUM's spiral rule)")
    T.w("instrument: lens/scripts/grade_unk_drop_talk.py")
    T.w("=" * 94)
    T.w("STORED COMPARATORS (quoted with their files, before anything is measured here):")
    for k, (v, f) in REF.items():
        T.w(f"  {k:<28} {v:<45} [{f}]")
    T.w()

    if not os.path.exists(PADFREE):
        T.w(f"{PADFREE} DOES NOT EXIST — the decider did not run. NO VERDICT IS GIVEN HERE; nothing")
        T.w("is inferred from a missing file. (The arm not reaching this step is recorded in lens/runs/UNK-DROP.txt.)")
        T.close()
        return
    txt = open(PADFREE).read()
    blocks = parse_padfree(txt)
    T.w(f"pad-free results read from {PADFREE} ({len(txt.splitlines())} lines, {len(blocks)} checkpoints)")
    for lab, mean, smp in blocks:
        T.w(f"  checkpoint block: {lab}")
        T.w(f"    mean <unk> share over 6 greedy 24-token samples: {mean}%")
    if len(blocks) < 2:
        T.w("FEWER THAN TWO CHECKPOINTS MEASURED — the arm cannot be compared to the twin in the same run.")
        T.w("Recorded as an incomplete decider; no P89f verdict is issued from one side.")
        T.close()
        return

    arm_lab, arm_mean, arm_smp = blocks[0]
    twin_lab, twin_mean, twin_smp = blocks[1]

    T.w()
    T.w("-" * 94)
    T.w("P89f(i) — THE <unk> FLOOD FALLS (bar: the arm's mean share is BELOW the twin's; my stated expectation")
    T.w("           was under 1%, and either way both numbers are printed)")
    T.w("-" * 94)
    T.w(f"  arm  {arm_lab}: {arm_mean}%      [{PADFREE}]")
    T.w(f"  twin {twin_lab}: {twin_mean}%      [same run, same instrument]")
    fell = (arm_mean is not None and twin_mean is not None and arm_mean < twin_mean)
    T.w(f"  -> P89f(i) {'HELD' if fell else 'FALSIFIED'}  (arm {arm_mean}% vs twin {twin_mean}%, "
        f"difference {(arm_mean - twin_mean):+.1f} points)")
    T.w(f"  my stated expectation (under 1%): {'MET' if (arm_mean is not None and arm_mean < 1.0) else 'NOT MET'}"
        f" ({arm_mean}%)")
    T.w()

    T.w("-" * 94)
    T.w("P89f(ii) — THE <unk> SPIRAL IS GONE, on the pre-declared mechanical rule (the ADDENDUM's item 2: SPIRAL iff")
    T.w("           <unk> share >= 10% AND longest non-<unk> word run <= 6, same prompt, same instrument)")
    T.w("-" * 94)
    for lab, smp in ((("arm  " + arm_lab), arm_smp), (("twin " + twin_lab), twin_smp)):
        for conv, unk, run, text in smp:
            if conv == 10955:
                spiral = (unk >= 10.0 and run <= 6)
                T.w(f"  {lab} conv {conv}: <unk> {unk}%  longest non-<unk> run {run}  -> "
                    f"{'SPIRAL' if spiral else 'not a spiral'}")
                T.w(f"      {text}")
    a10955 = next((s for s in arm_smp if s[0] == 10955), None)
    if a10955:
        spiral = (a10955[1] >= 10.0 and a10955[2] <= 6)
        T.w(f"  the arm's own number for the verdict: conv 10955 <unk> {a10955[1]}% / run {a10955[2]}"
            f"   [{PADFREE}]")
        T.w(f"  reference: attention arm 33.3% / run 3 (SPIRAL), twin 0.0% / run 24 (not a spiral)"
            f"   [lens/runs/PADFREE-TALK-ATTN.txt]")
        T.w(f"  -> P89f(ii) {'FALSIFIED (still a spiral)' if spiral else 'HELD'}")
    else:
        T.w("  conv 10955 is not in the arm's sample set — this prompt set differs from the stored one;")
        T.w("  P89f(ii) is recorded as NOT GRADED rather than transferred to a different conversation.")
    T.w()

    T.w("-" * 94)
    T.w("P89f(iii) — FLUENCY AND THE HEDGE. The topical-answer count is BY EYE and is not produced here.")
    T.w("-" * 94)
    T.w(f"  {arm_lab}, all six greedy samples verbatim (this is the text the decider is read from):")
    for conv, unk, run, text in arm_smp:
        T.w(f"    conv {conv:>6}  <unk> {unk:>5.1f}%  longest run {run:>2}  hedge={hedge_of(text)}")
        T.w(f"      {text}")
    n_hedge = sum(1 for s in arm_smp if hedge_of(s[3]))
    n_run4 = sum(1 for s in arm_smp if s[2] >= 4)
    n_allunk = sum(1 for s in arm_smp if s[1] == 100.0)
    T.w(f"  mechanical: hedges {n_hedge}/{len(arm_smp)} (twin stored: 4/6, {REF['twin talk pattern'][1]}), "
        f"samples with a >=4-word non-<unk> run {n_run4}/{len(arm_smp)}, all-<unk> samples {n_allunk}")
    T.w(f"  the twin's numbers measured in the same run for comparison: mean <unk> {twin_mean}%")
    T.w("  -> P89f(iii) verdict is read BY EYE from the six samples above against their prompts (printed in")
    T.w(f"     {PADFREE} above each sample) and written into STATE.md by the turn that reads this file.")
    T.w()

    if os.path.exists(FAIR):
        f = open(FAIR).read()
        T.w("-" * 94)
        T.w(f"THE QUANTITATIVE HALF (P65c/P65d), read from {FAIR}")
        T.w("-" * 94)
        m = re.search(r"step 8 free-running ([\d.]+)% against teacher-forced ([\d.]+)%", f)
        mc = re.search(r"real context\s+([\d.]+)%", f)
        mm = re.search(r"mismatched context\s*([\d.]+)%\s*\(drop ([\d.]+) points\)", f)
        if m:
            T.w(f"  step-8 free-running {m.group(1)}% against teacher-forced {m.group(2)}%   [{FAIR}]")
            T.w(f"    stored: twin 3.9% [lens/runs/TALK-FAIR-finalbest-d256-repro.txt], "
                f"attention 2.7% [lens/runs/TALK-FAIR-ATTN-FULL.txt]")
        if mc and mm:
            T.w(f"  context ablation: real {mc.group(1)}%, mismatched {mm.group(1)}% -> drop {mm.group(2)} points")
            T.w(f"    stored twin drop: 33.07 points [lens/runs/TALK-FAIR-finalbest-d256-repro.txt]")
    else:
        T.w(f"  {FAIR} DOES NOT EXIST — the step-8/context half is recorded as NOT MEASURED.")
    T.w()
    T.w("P89g: NOTHING IS SHIPPED from this arm by any of this. A word-salad decider means the arm failed")
    T.w("whatever the loss says, and if the decider fails the lever is CLOSED WITH ITS NUMBER, not retried.")
    T.w("=" * 94)
    T.close()


if __name__ == "__main__":
    sys.exit(main())
