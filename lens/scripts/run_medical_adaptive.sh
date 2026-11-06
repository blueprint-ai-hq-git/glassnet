#!/bin/bash
# run_medical_adaptive.sh — the DIAGNOSIS-DRIVEN fix, at the SMALL rung.
#
# THE DIAGNOSIS (measured, not guessed):
#   PathMNIST with the fixed Fashion-MNIST thresholds:
#     is_bright support 0.9353, has_low_contrast 0.8279, mostly_ink 0.9243  -> near-constant
#     is_dark 0.0000, has_high_contrast 0.0000                              -> dropped
#     top_half_heavier pos_weight 16.05, centre_heavier 19.96
#   A near-constant concept cannot be learned, and its balanced BCE at init (1.31 x 0.3 = 0.39)
#   competed with the answer loss (2.56). Three of twelve concepts were degenerate.
#
# THE ONE CHANGE: fit each threshold on TRAIN as a quantile, so every concept splits the actual data.
# Measured effect on the concept set alone: 12 kept / 6 dropped / max pos_weight 19.96  ->
# 16 kept / 0 dropped / max pos_weight 5.67.
#
# Deliberately NOT changed at the same time: the colour channel issue. PathMNIST's R/G/B means are
# 188.8 / 136.0 / 180.0 and this loader averages them, destroying the stain colour that separates
# tissue types. That is a REAL second cause and it is next, but one change at a time.
#
# Pre-registered predictions:
#   1. The answer stops collapsing onto one class. TEST beats the 41.41% majority baseline, and the
#      answer distribution is no longer a single word.
#   2. Concept neurons stop being dead: image concepts pass > 0 (they were 0/12 twice).
#   3. Medical is still harder than fashion's 88.13%.
#   4. If accuracy is at the majority baseline again, the concept layer was NOT the cause and the
#      colour issue is the real one.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ medical vqa3 — ADAPTIVE concepts, SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --adaptive --device cpu --max-train "${MED_TRAIN:-20000}" --out med_adapt 2>&1 | tail -70

echo
echo "############ medical: all three attempts ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'attempt':<22}{'text AUROC':>12}{'image AUROC':>13}{'TEST':>9}{'concepts':>10}  vs majority")
for tag, out in (("1 UNBALANCED", "med_small"),
                 ("2 BALANCED", "med_bal"),
                 ("3 ADAPTIVE concepts", "med_adapt")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<22}{'not run':>12}")
        continue
    r = json.load(open(f)); a = r["audit"]; m = r["meta"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    maj = m.get("majority_baseline", 0) * 100
    rel = "ABOVE" if acc > maj + 0.5 else ("at" if abs(acc - maj) <= 0.5 else "below")
    print(f"  {tag:<22}{np.mean(txt):>12.4f}{np.mean(img):>13.4f}{acc:>8.2f}%"
          f"{a['n_pass']:>6}/{a['n_concepts']:<3}  {rel} ({maj:.2f}%)")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
