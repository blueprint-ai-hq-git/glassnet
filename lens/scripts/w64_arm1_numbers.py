#!/usr/bin/env python3
"""w64_arm1_numbers.py — READ-ONLY. ARM 1 (window 64, batch 128) of run_tied_w64.sh, printed beside the
verdicts of lens/runs/W64-PREDICTIONS-P95.txt so no number is separated from the claim it supports.

This is NOT the authoritative grade: the chained job runs lens/scripts/grade_w64.py at the end of ARM 2 and
writes lens/runs/GRADE-P95.txt (both arms, appended). This script exists because ARM 1 finished at 11:10:38
while ARM 2 was still training, so its half of P95a/b/c is already measurable and should be on the record.

Reads: lens/runs/chat/ct_d256_w64_b128_history.jsonl (+ the twin's own history for the comparison).
Opens no checkpoint, loads no model, touches no GPU. Prints to stdout only.
"""
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(PROJ)
TWIN = "ct_d256_full169_fix3"
ARM1 = "ct_d256_w64_b128"
ARM2 = "ct_d256_w64_b256"
P95C_BAR = 33.76      # twin 35.26 - 1.5 points, declared in W64-PREDICTIONS-P95.txt
TWIN_BEST = 35.26     # lens/runs/chat/ct_d256_full169_fix3_history.jsonl ep3, by position
TWIN_SECONDS = [965, 1091, 1093]


def hist(tag):
    p = f"lens/runs/chat/{tag}_history.jsonl"
    if not os.path.exists(p):
        print(f"  {tag}: NO history file ({p})")
        return None
    rows = [json.loads(x) for x in open(p) if x.strip()]
    print(f"  --- {tag}   [{p}]")
    print(f"  {'ep':>3} {'train':>7} {'by-pos':>7} {'n':>6} {'legacy':>7} {'n':>6} {'asst_ppl':>9} {'gap':>7} {'sec':>6}")
    for r in rows:
        print(f"  {r['epoch']:>3} {r['train_acc']*100:>6.2f}% {r['val_asst_acc']*100:>6.2f}% {r['val_asst_n']:>6}"
              f" {r['val_masked_acc']*100:>6.2f}% {r['val_masked_n']:>6} {r['val_asst_ppl']:>9.2f}"
              f" {(r['train_acc']-r['val_asst_acc'])*100:>+6.2f} {r['seconds']:>6.0f}")
    return rows


print("=" * 100)
print("ARM 1 (window 64, batch 128) — its own numbers, beside P95's verdicts.  READ-ONLY, no GPU, no model.")
print("  predictions: lens/runs/W64-PREDICTIONS-P95.txt")
print("  console:     lens/runs/TIED-W64.txt  (ARM EXIT=0 tag=ct_d256_w64_b128 batch=128 wall=2400s 11:10:38)")
print("=" * 100)

rows1 = hist(ARM1)
rows2 = hist(ARM2)
tw = hist(TWIN)

print("")
print("P95c — THE HONEST COLUMN IS NOT DAMAGED: best by-position >= 33.76% (twin 35.26 - 1.5)")
if tw:
    tb = max(tw, key=lambda r: r["val_asst_acc"])
    print(f"  twin {TWIN} best ep{tb['epoch']}: by-position {tb['val_asst_acc']*100:.2f}%"
          f"  [lens/runs/chat/{TWIN}_history.jsonl]   (the constant 35.26 was read from this file)")
if rows1:
    b1 = max(rows1, key=lambda r: r["val_asst_acc"])
    a1 = b1["val_asst_acc"] * 100
    print(f"  ARM 1 best ep{b1['epoch']}: by-position {a1:.4f}% (n={b1['val_asst_n']})  train {b1['train_acc']*100:.2f}%"
          f"  -> bar 33.76%: {'PASS' if a1 >= P95C_BAR else 'FAIL'}   delta vs twin {a1 - TWIN_BEST:+.2f} points")
    print(f"  VERDICT P95c (ARM 1 half): {'PASS' if a1 >= P95C_BAR else 'FAIL'} — more context COST held-out accuracy"
          f" by {TWIN_BEST - a1:.2f} points. Under the prediction's own wording that is recorded as the cost of the"
          f" lever and not re-run ({'PASS' if a1 >= P95C_BAR else 'FAIL'} stands as written).")
if rows2:
    b2 = max(rows2, key=lambda r: r["val_asst_acc"])
    print(f"  ARM 2 best ep{b2['epoch']}: by-position {b2['val_asst_acc']*100:.4f}%  -> ARM 2 is {'BETTER' if b2['val_asst_acc']>b1['val_asst_acc'] else 'NOT better'} than ARM 1")
else:
    print("  ARM 2: no epoch line yet (still training at the time of this print) -> P95c NOT gradable for ARM 2 here.")

print("")
print("P95b — COST: per-epoch wall 1,200-2,600 s promised; peak host RSS <= 9.5 GB; VRAM < 5 GB")
if rows1:
    secs = [round(r["seconds"]) for r in rows1]
    print(f"  ARM 1 per-epoch seconds {secs}  vs the twin's {TWIN_SECONDS}  -> ratios"
          f" {[round(s/t,2) for s, t in zip(secs, TWIN_SECONDS)]}")
    print(f"    band 1,200-2,600 s: {'inside' if all(1200 <= s <= 2600 for s in secs) else 'MISSED ON THE LOW SIDE'}"
          f" — the arm is FASTER than I predicted; P95b's declared falsifier was only the UPPER side"
          f" ('FALSIFIED ABOVE 3,000 s/epoch'): fired={any(s > 3000 for s in secs)}")
    print(f"    => my stated band MISSED (recorded as a miss on my wording); the falsifier did NOT fire,"
          f" so P95b HOLDS as worded. Total ARM 1 wall was 2,400 s for 3 epochs + build/val/talk.")
vram_f = "lens/runs/W64-VRAM-SAMPLE.txt"
mus = []
if os.path.exists(vram_f):
    for ln in open(vram_f):
        m = re.match(r"\s*(\d+)\s*MiB", ln)
        if m:
            mus.append(int(m.group(1)))
    print(f"  VRAM: {len(mus)} samples, max used {max(mus) if mus else '?'} MiB  [{vram_f}]"
          f"  -> under the 5 GB rule: {bool(mus) and max(mus) < 5120}")
else:
    print(f"  VRAM: {vram_f} missing -> NOT MEASURED")
peak = ""
try:
    pk = os.popen("systemctl --user show -p MemoryPeak glassnet-q-2325387518-103038.service").read().strip()
    m = re.search(r"MemoryPeak=(\d+)", pk)
    if m:
        b = int(m.group(1))
        peak = f"  host RSS: MemoryPeak={b:,} B = {b/2**30:.2f} GiB against the 9.5 GB promise and the 10 GiB MemoryMax [{pk}]  -> {'PASS' if b <= 9.5e9 else 'FAIL'}"
    else:
        peak = f"  host RSS: scope reports '{pk}' -> UNMEASURED (not estimated)"
except Exception as e:  # noqa: BLE001
    peak = f"  host RSS: scope read failed ({e}) -> UNMEASURED"
print(peak)

print("")
print("P95a — WIRING: window 64 on every config line; the parameter number was FALSIFIED AS WORDED (K=39, not 36)")
print("  log line: 'windows 8,016,416 train / 40,262 val   vocab 12,002   window 64' [lens/runs/TIED-W64.txt]")
print("  8,385,911 params (d=256, K=39) vs the promised 8,300,750 (K=36) — cause and full arithmetic in")
print("  W64-PREDICTIONS-P95.txt ADDENDUM 2, written BEFORE any epoch line existed. Not a flag error.")

print("")
print("P95d/e/f/g — the DECIDER and the drift numbers: NOT MEASURED HERE.")
print("  They come from lens/runs/PADFREE-TALK-W64.txt + TALK-FAIR-W64.txt, produced by the chained job after")
print("  ARM 2 finishes, graded by lens/scripts/grade_w64_talk.py. P95g decides whether anything ships.")
n1 = sum(r["val_asst_n"] for r in rows1) / len(rows1) if rows1 else 0
print(f"  NOTE ON COMPARABILITY (printed with every share): ARM 1's by-position column is a share over n={n1:.0f}"
      f" assistant positions cut from 40,262 window-64 val windows, the twin's over n=58,191 cut from 87,035")
print("  window-32 val windows — the SAME 2% of conversations, different windows (P95 ADDENDUM 1, item 2).")
print("=" * 100)
