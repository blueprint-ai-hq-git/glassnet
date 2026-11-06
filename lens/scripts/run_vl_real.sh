#!/bin/bash
# run_vl_real.sh — the full VL model, on Fashion-MNIST and on PathMNIST (medical).
#
# The smoke (2 epochs, 6,000 images) showed the opposite of the image-only result:
#   readable 27,241 params TEST 69.47%  vs  plain 27,620 params TEST 63.45%   delta +6.02
# and it answered correctly: "what is in the photo" -> ankle_boot (true ankle_boot).
#
# Pre-registered predictions (written BEFORE the run):
#   1. Both models improve a lot on full data + 24 epochs. Expected readable 87-91% on Fashion-MNIST.
#   2. READABLE STILL BEATS PLAIN on VL, unlike image-only classification. My reason: the question
#      tokens mean the concept layer has two independent things to be informative about (what is in
#      the picture AND which question was asked), so the middle is less of a pure bottleneck and
#      more of a useful summary. If readable LOSES here too, that explanation is wrong and the
#      image-only result generalises -- which is itself worth knowing.
#   3. IMAGE concepts pass better than the image-only run's 9/13, because the shared layer is doing
#      more work and gets more gradient. TEXT concepts: asks_name/asks_see/asks_identify should
#      pass; asks_what may not, because "what" appears in two of four templates so it fires on ~50%
#      of samples and is a coarse concept.
#   4. PathMNIST (medical, 9 colon tissue classes, CC BY 4.0) is a HARDER task than Fashion-MNIST --
#      texture differences rather than object outlines. Expect lower absolute accuracy on both.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_vl.py|train_image.py|train_chat.py|train_code.py" > /dev/null; do
  sleep 20
done
echo "GPU free; starting the VL runs"

for DATA in fashion pathmnist; do
  echo
  echo "################################################################"
  echo "# VL — $DATA"
  echo "################################################################"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
    --data "$DATA" --d 64 --patch 7 --epochs "${VL_EPOCHS:-24}" --batch 256 \
    --lr 0.3 --lr-decay 0.5 --lam-concept 0.3 --device gpu --out "vl_${DATA}" 2>&1 \
    | tail -80
done

echo
echo "################################################################"
echo "# VL — THE PICTURE"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'run':<20}{'readable':>12}{'plain':>10}{'delta':>9}{'concepts':>12}  source")
for tag, out in (("smoke fashion", "vlsmoke"), ("fashion", "vl_fashion"),
                 ("pathmnist", "vl_pathmnist")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<20}{'-- not run --':>12}")
        continue
    r = json.load(open(f))
    rr, pp = r["results"]["readable"], r["results"]["plain"]
    ra, pa = rr["history"][-1]["test_acc"] * 100, pp["history"][-1]["test_acc"] * 100
    a = r["audit"]
    print(f"  {tag:<20}{ra:>11.2f}%{pa:>9.2f}%{ra-pa:>+9.2f}{a['n_pass']:>7}/{a['n_concepts']:<4}  "
          f"{r['meta']['source']}")
PY

cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
