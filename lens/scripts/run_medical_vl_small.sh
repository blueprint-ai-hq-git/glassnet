#!/bin/bash
# run_medical_vl_small.sh — the medical VL at the SMALL rung, before any big medical run.
#
# PathMNIST (CC BY 4.0): 9 colon tissue classes, 89,996 train / 7,180 test images.
# The question "is this cancer" is medically meaningful here (yes for cancer-associated stroma and
# colorectal adenocarcinoma), which is exactly why the predicate is validated before use.
#
# Pre-registered predictions:
#   1. Medical is HARDER than Fashion-MNIST: these are texture differences between tissue types, not
#      object outlines. Expect noticeably lower accuracy than the 88.13% fashion run.
#   2. Text concepts still pass strongly (AUROC > 0.90). The questions are just as load-bearing as in
#      fashion, and reading a question does not depend on the pictures being easy.
#   3. Image concepts pass WORSE than fashion's 10/13. The pixel concepts (is_bright, edges,
#      symmetry) describe shapes; tissue texture is not shape, so fewer of them will be faithful.
#   4. Readable vs plain is much closer here, or readable loses. On image-only classification
#      readable lost by 3.06; the questions give the concept layer something to do, so I expect
#      readable to be within +/-2 of plain.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ medical VL — SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
  --device cpu --max-train "${MED_TRAIN:-20000}" --out med_small 2>&1 | tail -62

echo
echo "############ fashion vs medical ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
print(f"  {'run':<14}{'text AUROC':>12}{'image AUROC':>13}{'TEST acc':>10}{'concepts':>11}  source")
for tag, out in (("fashion vqa2", "vqa2_small"), ("medical vqa3", "med_small")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag:<14}{'not run':>12}")
        continue
    r = json.load(open(f)); a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    acc = r["results"]["readable"]["history"][-1]["test_acc"] * 100
    print(f"  {tag:<14}{np.mean(txt):>12.4f}{np.mean(img):>13.4f}{acc:>9.2f}%"
          f"{a['n_pass']:>7}/{a['n_concepts']:<4}  {r['meta']['source']}")
    m = r["meta"]
    print(f"                 majority baseline {m.get('majority_baseline', 0)*100:.2f}%"
          f"   true chance {m.get('true_chance', 0)*100:.2f}%")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
