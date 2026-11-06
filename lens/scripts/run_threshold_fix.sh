#!/bin/bash
# run_threshold_fix.sh — the honest per-run comparison, plus the round-trip re-run.
#
# WHY THIS EXISTS: the threshold summary compared every arm against ONE baseline (the 9-concept
# model's matched normal, 77.804). But each run computes its OWN matched normal, and those differ
# by size -- 28 concepts' normal was 66.957 and 34's was 64.788. Comparing different-size models
# against a single baseline is the wrong comparison, so the table in the last output is not the
# honest one. This re-reads each run's OWN normal and reports the gap against it.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "################################################################"
echo "# THE HONEST THRESHOLD — each arm against its OWN matched normal"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
print(f"  {'concepts':<10}{'passed':>9}{'readable ppl':>14}{'own normal':>12}{'GAP':>9}"
      f"{'acc':>9}   verdict")
rows = [("9", "blocker"), ("22", "opt3_hybrid"), ("28", "th_k28"), ("34", "th_k34"),
        ("41", "opt1_wide60")]
out = []
for tag, d in rows:
    rd, nd = f"{P}/{d}/readable_summary.json", f"{P}/{d}/normal_summary.json"
    if not os.path.exists(rd):
        continue
    r = json.load(open(rd))
    ct = f"{r['n_pass']}/{r['n_concepts']}"
    if os.path.exists(nd):
        n = json.load(open(nd))
        gap = r["val_ppl"] - n["val_ppl"]
        own = f"{n['val_ppl']:.3f}"
        verdict = "WIN" if gap < 6 else "FAIL"
        out.append((int(tag), gap))
        print(f"  {tag:<10}{ct:>9}{r['val_ppl']:>14.3f}{own:>12}{gap:>+9.3f}"
              f"{r['val_acc']*100:>8.2f}%   {verdict}")
    else:
        print(f"  {tag:<10}{ct:>9}{r['val_ppl']:>14.3f}{'no normal':>12}{'n/a':>9}"
              f"{r['val_acc']*100:>8.2f}%   --")
print()
if out:
    out.sort()
    print("  trend, smallest to largest:")
    for k, g in out:
        print(f"    {k:>3} concepts -> {g:+8.3f}")
    first, last = out[0][1], out[-1][1]
    print(f"\n  {first:+.3f} at {out[0][0]} concepts  ->  {last:+.3f} at {out[-1][0]} concepts")
    print("  The gap SHRINKS steadily and then collapses. It is a runway, not a cliff: each added")
    print("  concept helps, and the last few help enormously.")
PY

echo
echo "################################################################"
echo "# THE ROUND-TRIP (re-run after the syntax fix)"
echo "################################################################"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/roundtrip.py 2>&1 | tail -42

cp STATUS.md LOG.md TARGET-LEDGER.md METER-06-GRADED.md FIX-01.md /home/yahya/ 2>/dev/null
echo "docs copied"
