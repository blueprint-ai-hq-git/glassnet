#!/bin/bash
# run_image_real.sh — the real image model on the full dataset.
#
# The smoke (2 epochs, 6,000 images) already showed the shape of the result:
#   readable 16,665 params  TEST 59.39%   vs   plain 16,705 params  TEST 59.23%
#   delta +0.16 points, 7/13 concepts passing
#
# Pre-registered predictions (written BEFORE this run):
#   1. Both models improve a lot with 60,000 images and 24 epochs. I expect readable in the
#      87-90% band, which is the published range for a small MLP on Fashion-MNIST.
#   2. Readable lands WITHIN 1 point of plain. On the text side readable paid a penalty; on images
#      the concept layer may be closer to free because the concepts are genuinely informative
#      (brightness, symmetry, edges), so I expect a smaller gap than text showed.
#   3. Concepts passing rises clearly above 7/13, but NOT to 13/13. The hard ones will be
#      `centre_heavier` (support 0.97 -- it fires on nearly everything, so AUROC is meaningless)
#      and `is_thin_structure`. A concept that fires on 97% of inputs cannot discriminate.
#   4. `left_half_heavier` was the rarest kept concept at 0.0215 support and fired at 0.004 in the
#      smoke -- I expect it to fail or be marginal again. The 2% floor is close to its support.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_chat.py|train_code.py|run_best.sh" > /dev/null; do
  sleep 20
done
echo "GPU free; starting the image model"

nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
  --d 64 --patch 7 --epochs "${IMG_EPOCHS:-24}" --batch 256 --lr 0.3 --lr-decay 0.5 \
  --lam-concept 0.3 --device gpu --out img1 2>&1 | tail -70

echo
echo "################################################################"
echo "# IMAGE — readable vs plain, and the concept audit"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/image"
for tag, out in [("smoke", "imgsmoke"), ("real", "img1")]:
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag}: -- not run --")
        continue
    r = json.load(open(f))
    rr, pp = r["results"]["readable"], r["results"]["plain"]
    ra, pa = rr["history"][-1]["test_acc"], pp["history"][-1]["test_acc"]
    print(f"  {tag}: readable {rr['params']:>7,} params  TEST {ra*100:6.2f}%   "
          f"plain {pp['params']:>7,} params  TEST {pa*100:6.2f}%   "
          f"delta {(ra-pa)*100:+.2f}")
    print(f"        concepts {r['audit']['n_pass']}/{r['audit']['n_concepts']}")
PY

cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
