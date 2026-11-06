#!/usr/bin/env python3
"""grade_w64.py — P95a/b/c FROM STORED TEXT ONLY (no checkpoint, no model, no GPU).

Reads lens/runs/TIED-W64.txt (both arms' own console) and the two arms' history records, plus the stored
twin's records for the comparison, and prints the verdicts beside the values they were read from
(BUGS.md verdict-check rule: a right number with a wrong conclusion is the signature failure here).

Writes lens/runs/GRADE-P95.txt (APPEND; nothing overwritten).
"""
import json
import os
import re
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(PROJ)
OUT = "lens/runs/GRADE-P95.txt"
LOG = "lens/runs/TIED-W64.txt"
VRAM = "lens/runs/W64-VRAM-SAMPLE.txt"
TWIN_HIST = "lens/runs/chat/ct_d256_full169_fix3_history.jsonl"
TWIN_LOG = "lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt"
TWIN_ASST_BEST = 35.26      # lens/runs/chat/ct_d256_full169_fix3_history.jsonl, ep3, by position
TWIN_SECONDS = [965, 1091, 1093]   # lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt
P95C_BAR = 33.76            # twin 35.26 - 1.5 points
P95A_PARAMS = 8300750       # promised in P95a for window 64, d=256, K=36
P95A_DELTA = 2392064        # = dmix 2097152 + concept.W 294912

lines = []


def w(s=""):
    print(s)
    lines.append(s)


def hist(tag):
    p = f"lens/runs/chat/{tag}_history.jsonl"
    if not os.path.exists(p):
        return None
    rows = [json.loads(x) for x in open(p) if x.strip()]
    return rows or None


w("=" * 94)
w("GRADE P95a/b/c — WINDOW 64 (two arms), from stored text only")
w("predictions: lens/runs/W64-PREDICTIONS-P95.txt")
w("=" * 94)

if not os.path.exists(LOG):
    w(f"MISSING {LOG} — nothing to grade.")
    open(OUT, "a").write("\n".join(lines) + "\n")
    sys.exit(0)
txt = open(LOG, encoding="utf-8", errors="replace").read()

# ---------------------------------------------------------------- P95a: is the run ACTUALLY window 64?
w("")
w("P95a — THE RUN IS WINDOW 64, and its parameter count is the promised one")
win_lines = re.findall(r"windows ([\d,]+) train / ([\d,]+) val\s+vocab ([\d,]+)\s+window (\d+)", txt)
cap_lines = re.findall(r"window cap:\s*([\d,]+) windows chosen", txt)
param_lines = re.findall(r"model:\s*([\d,]+) params\s*\(d=(\d+), K=(\d+)\)", txt)
for i, (tr, va, vo, wn) in enumerate(win_lines):
    w(f"  config line {i+1}: windows {tr} train / {va} val  vocab {vo}  window {wn}")
for i, mp in enumerate(cap_lines):
    w(f"  window-cap line {i+1}: {mp} windows chosen")
for i, (np_, dd, kk) in enumerate(param_lines):
    w(f"  model line {i+1}: {int(np_.replace(',', '')):,} params (d={dd}, K={kk})")
if not win_lines or not param_lines:
    w("  -> NOT PARSED: one of the config/model lines is absent; the arm may not have reached them.")
else:
    ok_win = all(wn == "64" for *_x, wn in win_lines)
    # THE EXPECTED PARAMETER COUNT IS COMPUTED FROM THE ARM'S OWN PRINTED (K, WINDOW), not from a constant:
    # P95a's promised 8,300,750 assumed the twin's K=36, and at window 64 the per-window support rule admits
    # 39 concepts (mentions_health/music/clothing cross the 0.005 floor), so a constant comparison would print
    # a false FAIL — see W64-PREDICTIONS-P95.txt ADDENDUM 2. The twin's own K=36 count 5,908,686 is reproduced
    # by the same formula, which is what makes the formula trustworthy.
    def expect(V, D, W, K):
        emb = V * D
        concept_w = D * W * K
        answer_w = K * V
        dmix = D * W * D if True else 0     # direct_mode=tied
        return emb + concept_w + K + answer_w + V + dmix
    for i, (np_, dd, kk) in enumerate(param_lines):
        n_meas, K, Wd = int(np_.replace(",", "")), int(kk), int(win_lines[0][3]) if win_lines else None
        exp = expect(12002, int(dd), Wd, K)
        w(f"  model line {i+1}: {n_meas:,} params (d={dd}, K={K}) ; formula for (V=12002, d={dd}, W={Wd}, K={K}) "
          f"= {exp:,} -> {'MATCH' if exp == n_meas else 'MISMATCH'}")
        w(f"    vs the twin ct_d256_full169_fix3: 5,908,686 (d=256, K=36)   delta {n_meas - 5908686:+,}"
          f"   [K {K} vs 36 is the declared confound, ADDENDUM 2]")
    w(f"  verdict: window==64 on every config line: {ok_win}   formula reproduced on every model line: "
      f"{all(expect(12002, int(dd), int(win_lines[0][3]), int(kk)) == int(np_.replace(',', '')) for np_, dd, kk in param_lines)}")
    w(f"  P95a -> {'PASS' if ok_win else 'FAIL'}; the PARAMETER NUMBER itself was FALSIFIED AS WORDED (promised "
      f"8,300,750 for K=36, measured for K=39) — the cause and the full arithmetic are in "
      f"lens/runs/W64-PREDICTIONS-P95.txt ADDENDUM 2, written BEFORE any epoch line existed.")

# ---------------------------------------------------------------- P95b: cost
w("")
w("P95b — COST: per-epoch wall 1,200-2,600 s promised; peak host RSS <= 9.5 GB; VRAM < 5 GB")
w(f"  the twin's own epochs: {TWIN_SECONDS[0]}/{TWIN_SECONDS[1]}/{TWIN_SECONDS[2]} s  [{TWIN_LOG}]")
for tag in ("ct_d256_w64_b128", "ct_d256_w64_b256"):
    rows = hist(tag)
    if not rows:
        w(f"  {tag}: NO history rows yet (arm not finished / not started)")
        continue
    secs = [round(r["seconds"]) for r in rows]
    w(f"  {tag}: per-epoch seconds {secs}  (n={len(secs)})  ratio to the twin's {TWIN_SECONDS} = "
      f"{[round(s / t, 2) for s, t in zip(secs, TWIN_SECONDS)]}")
walls = re.findall(r"ARM EXIT=(-?\d+)(?:\s+tag=(\S+)\s+batch=(\d+))?\s+wall=(\d+)s", txt)
for rc, tag, ba, wl in walls:
    w(f"  ARM EXIT={rc}  tag={tag or '-'}  batch={ba or '-'}  wall={wl} s")
if os.path.exists(VRAM):
    mus = []
    for ln in open(VRAM):
        m = re.match(r"\s*(\d+)\s*MiB", ln)
        if m:
            mus.append(int(m.group(1)))
    if mus:
        w(f"  VRAM samples {len(mus)}: max used {max(mus)} MiB, median {sorted(mus)[len(mus)//2]} MiB "
          f"[{VRAM}]  -> under the 5 GB rule: {max(mus) < 5120}")
else:
    w(f"  {VRAM} missing -> VRAM NOT MEASURED here")
w("  peak host RSS: NOT READ HERE — it lives in the arm's own systemd scope, read with")
w("    systemctl --user show -p MemoryPeak glassnet-q-<stamp>.service   (recorded as UNMEASURED if absent)")

# ---------------------------------------------------------------- P95c: the honest column
w("")
w("P95c — THE HONEST COLUMN IS NOT DAMAGED: best by-position held-out >= 33.76% (twin 35.26 minus 1.5)")
twin_rows = hist("ct_d256_full169_fix3") or []
if twin_rows:
    tb = max(twin_rows, key=lambda r: r["val_asst_acc"])
    w(f"  TWIN ct_d256_full169_fix3 best ep{tb['epoch']}: by-position {tb['val_asst_acc']*100:.2f}% "
      f"legacy {tb['val_masked_acc']*100:.2f}% train {tb['train_acc']*100:.2f}%   [{TWIN_HIST}]")
    w(f"  (the constant compared against is {TWIN_ASST_BEST}% as declared in W64-PREDICTIONS-P95.txt; "
      f"the file above reads {tb['val_asst_acc']*100:.2f}% — if they differ, THE FILE WINS and the prediction "
      f"text was wrong)")
w(f"  {'arm':<20} {'ep':>2} {'train':>7} {'by-pos':>7} {'legacy':>7} {'gap':>6}")
bests = {}
for tag in ("ct_d256_w64_b128", "ct_d256_w64_b256"):
    rows = hist(tag)
    if not rows:
        continue
    b = max(rows, key=lambda r: r["val_asst_acc"])
    bests[tag] = b
    for r in rows:
        gap = (r["train_acc"] - r["val_asst_acc"]) * 100
        mark = " <-best" if r is b else ""
        w(f"  {tag:<20} {r['epoch']:>2} {r['train_acc']*100:>6.2f}% {r['val_asst_acc']*100:>6.2f}% "
          f"{r['val_masked_acc']*100:>6.2f}% {gap:>+5.2f}{mark}")
if bests:
    for tag, b in bests.items():
        a = b["val_asst_acc"] * 100
        w(f"  {tag}: best {a:.2f}% vs bar {P95C_BAR}% and twin {TWIN_ASST_BEST}% -> "
          f"{'PASS' if a >= P95C_BAR else 'FAIL'}   (delta against the twin {a - TWIN_ASST_BEST:+.2f} points)")
else:
    w("  no arm has finished an epoch yet -> P95c NOT GRADABLE YET")

w("")
w("P95d/e/f/g come from PADFREE-TALK-W64.txt and TALK-FAIR-W64.txt via lens/scripts/grade_w64_talk.py.")
w(f"written to {OUT} (appended)")
open(OUT, "a").write("\n".join(lines) + "\n" + "=" * 94 + "\n")
