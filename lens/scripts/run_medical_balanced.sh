#!/bin/bash
# run_medical_balanced.sh — the ONE fix for the medical collapse, at the SMALL rung.
#
# THE FAILURE (recorded, not reworded):
#   medical vqa3, unbalanced: text AUROC 0.5035, image AUROC 0.4692, TEST 41.41%, concepts 0/16.
#   TEST 41.41% == the majority baseline 41.41%, to the digit. The model answered "no" to
#   everything, including every "what is in the photo" question. It found the free lunch  -- "no"
#   is 41.41% of the answers -- and collapsed onto it, and the concept neurons died with it
#   (fires 0.000 for most, saturated at 1.000 for two).
#
# THE ONE CHANGE: weight the answer loss by inverse label frequency, so the free lunch is gone.
# The weighted cross-entropy is gradient-checked (gradcheck_wce.py: loss matches an independent
# numpy reference to 0.00e+00, gradient to 1.49e-08, and the unweighted path is bit-identical).
#
# Pre-registered predictions:
#   1. TEST accuracy rises ABOVE the 41.41% majority baseline. If it does not, weighting did not
#      help and the cause is elsewhere (probably the pixel concepts being wrong for texture).
#   2. Concept neurons stop being dead: image concepts pass > 0, where they passed 0/12.
#   3. Medical stays HARDER than fashion's 88.13%. I expect 50-70%.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ medical vqa3 — BALANCED answer loss, SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --balance --device cpu --max-train "${MED_TRAIN:-20000}" --out med_bal 2>&1 | tail -70

echo
echo "############ medical: collapsed vs balanced ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'run':<16}{'text AUROC':>12}{'image AUROC':>13}{'TEST acc':>10}{'concepts':>10}  verdict")
for tag, out in (("medical UNBAL", "med_small"), ("medical BALANCED", "med_bal")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<16}{'not run':>12}")
        continue
    r = json.load(open(f)); a = r["audit"]; m = r["meta"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    maj = m.get("majority_baseline", 0) * 100
    verdict = "AT majority" if abs(acc - maj) < 0.5 else ("ABOVE majority" if acc > maj else "below")
    print(f"  {tag:<16}{np.mean(txt):>12.4f}{np.mean(img):>13.4f}{acc:>9.2f}%"
          f"{a['n_pass']:>7}/{a['n_concepts']:<3}  {verdict} (maj {maj:.2f}%)")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
