#!/bin/bash
# run_medical_control.sh — ISOLATE the two changes made in attempt 4.
#
# Attempt 4 changed TWO things at once:
#   (a) keep the RGB colour   (--color)
#   (b) lower the concept loss weight 0.3 -> 0.15
# and gained +11.57 points (52.87% -> 64.44%) plus concepts 8/20 -> 14/20.
#
# Neither can be credited without a control. This runs (b) alone, on grayscale, everything else
# identical to attempt 3. Whatever the control does NOT explain belongs to colour.
#
# This is the discipline the project's own rule demands: change one thing at a time so a break (or a
# win) can be attributed. I broke it in attempt 4 and this is the repair.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ CONTROL: grayscale + lam_concept 0.15 (no colour) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --adaptive --lam-concept 0.15 --device cpu --max-train "${MED_TRAIN:-20000}" \
  --out med_lam015 2>&1 | tail -22

echo
echo "############ THE ISOLATION ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"


def stat(out):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f)); a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    return {"acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "txt": float(np.mean(txt)), "img": float(np.mean(img)),
            "concepts": f"{a['n_pass']}/{a['n_concepts']}",
            "img_pass": f"{a['img_pass']}/{a['img_total']}",
            "plain": r["results"]["plain"]["history"][-1]["test_acc"] * 100}


base = stat("med_adapt")
ctl = stat("med_lam015")
col = stat("med_color")
print(f"  {'run':<34}{'TEST':>9}{'image AUROC':>13}{'concepts':>10}{'img concepts':>14}")
for tag, s in (("adaptive, gray, lam 0.3   (attempt 3)", base),
               ("adaptive, gray, lam 0.15  (CONTROL)", ctl),
               ("adaptive, COLOUR, lam 0.15 (attempt 4)", col)):
    if s:
        print(f"  {tag:<34}{s['acc']:>8.2f}%{s['img']:>13.4f}{s['concepts']:>10}{s['img_pass']:>14}")
if base and ctl and col:
    print()
    print(f"  total gain attempt3 -> attempt4 : {col['acc']-base['acc']:+.2f} points")
    print(f"  gain from lam 0.15 ALONE         : {ctl['acc']-base['acc']:+.2f} points  (the control)")
    print(f"  gain left for COLOUR to explain  : {col['acc']-ctl['acc']:+.2f} points")
    print()
    if abs(ctl['acc'] - base['acc']) < 1.0 and col['acc'] - ctl['acc'] > 5:
        print("  -> the control says lam 0.15 alone changes almost nothing, so COLOUR explains")
        print("     essentially the whole gain.")
    elif col['acc'] - ctl['acc'] < 1.0:
        print("  -> the CONTROL explains the gain. COLOUR adds nothing, and my attribution to")
        print("     colour in attempt 4 was WRONG.")
    else:
        print("  -> BOTH contribute. Neither change can be credited alone.")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
