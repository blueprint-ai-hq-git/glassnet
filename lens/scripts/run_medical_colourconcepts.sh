#!/bin/bash
# run_medical_colourconcepts.sh — CHANGE: add colour concepts. SMALL rung.
#
# Everything identical to attempt 4 (adaptive grayscale concepts + RGB input + lam 0.15) EXCEPT
# adding 15 concepts measured on the RGB channels. ONE change, so the effect is attributable.
#
# Why it should help: the model could already USE colour, but the readable middle had no way to NAME
# it. Measured on TRAIN, the colour concepts separate the tissue classes sharply:
#     red_channel_high   per-class support min 0.013  max 0.982  spread 0.970
#     green_channel_high per-class support min 0.067  max 0.997  spread 0.931
# so the layer now has neurons that can actually describe the discriminating signal.
#
# Pre-registered predictions:
#   1. Image concepts pass better than 10/16, because the concepts now cover the signal the model is
#      actually using.
#   2. Accuracy rises above the 64.44% of attempt 4, or stays within ~1 point. I do NOT expect a big
#      accuracy jump -- the model already had the colour, so this should mainly improve READABILITY.
#   3. Total concepts rises from 20 to ~35.
#   4. Mean image-concept AUROC rises above 0.6760.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ medical: adaptive + COLOUR + COLOUR CONCEPTS, SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --adaptive --color --colour-concepts --lam-concept "${LAM:-0.15}" --device cpu \
  --max-train "${MED_TRAIN:-20000}" --out med_cc 2>&1 | tail -45

echo
echo "############ the whole medical progression ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'run':<30}{'TEST':>9}{'image AUROC':>13}{'concepts':>11}{'img concepts':>14}")
rows = [("1 UNBALANCED gray", "med_small"), ("2 BALANCED gray", "med_bal"),
        ("3 ADAPTIVE gray", "med_adapt"), ("4 + COLOUR", "med_color"),
        ("5 + COLOUR CONCEPTS", "med_cc")]
for tag, out in rows:
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<30}{'not run':>9}")
        continue
    r = json.load(open(f)); a = r["audit"]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    pl = r["results"]["plain"]["history"][-1]["test_acc"] * 100
    print(f"  {tag:<30}{acc:>8.2f}%{np.mean(img):>13.4f}{a['n_pass']:>7}/{a['n_concepts']:<3}"
          f"{a['img_pass']:>9}/{a['img_total']:<4}  plain {pl:.2f}%")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
