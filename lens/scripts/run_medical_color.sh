#!/bin/bash
# run_medical_color.sh — CHANGE B: keep the RGB colour. SMALL rung.
#
# THE DIAGNOSIS THIS ADDRESSES (measured in diag_medical.py):
#   PathMNIST raw RGB channel means:  R 188.8   G 136.0   B 180.0
#   The channels differ substantially, and the loader averaged them with `.mean(axis=-1)`,
#   destroying the stain colour that separates tissue types in histology.
#
# THIS IS ONE CHANGE. The concept definitions are deliberately left IDENTICAL -- `_img()` now
# converts colour input back to grayscale internally, so the concepts see exactly what they saw
# before. That means any difference in the result comes from the MODEL seeing colour, not from the
# concepts changing underneath it. Colour-aware concepts would be a separate, later change.
#
# Pre-registered predictions:
#   1. Readable accuracy rises above the 52.87% measured on grayscale, because colour carries the
#      signal the grayscale concepts cannot see.
#   2. Image concepts pass BETTER than 5/16, for the same reason -- the model can now separate
#      tissues by stain even where the brightness concepts cannot.
#   3. Medical still stays below fashion's 88.13%.
#   4. If accuracy does NOT improve, colour was not the bottleneck and the limit is the concept
#      vocabulary itself (brightness/edge/symmetry shapes on tissue texture).
#
# The concept term is also lowered here from 0.3 to 0.15, because the diagnosis showed it competing
# with the answer loss. That is a SECOND change and it is declared rather than hidden.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ medical, COLOUR, adaptive concepts, SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --adaptive --color --lam-concept "${LAM:-0.15}" --device cpu \
  --max-train "${MED_TRAIN:-20000}" --out med_color 2>&1 | tail -60

echo
echo "############ medical: grayscale vs colour ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'run':<24}{'text AUROC':>12}{'image AUROC':>13}{'TEST':>9}{'concepts':>10}  vs majority")
for tag, out in (("1 UNBALANCED (gray)", "med_small"),
                 ("2 BALANCED (gray)", "med_bal"),
                 ("3 ADAPTIVE (gray)", "med_adapt"),
                 ("4 ADAPTIVE + COLOUR", "med_color")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<24}{'not run':>12}")
        continue
    r = json.load(open(f)); a = r["audit"]; m = r["meta"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    pl = r["results"]["plain"]["history"][-1]["test_acc"] * 100
    maj = m.get("majority_baseline", 0) * 100
    rel = "ABOVE" if acc > maj + 0.5 else ("at" if abs(acc - maj) <= 0.5 else "below")
    print(f"  {tag:<24}{np.mean(txt):>12.4f}{np.mean(img):>13.4f}{acc:>8.2f}%"
          f"{a['n_pass']:>6}/{a['n_concepts']:<3}  {rel} (plain {pl:.2f}%, maj {maj:.2f}%)")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
