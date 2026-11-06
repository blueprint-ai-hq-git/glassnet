#!/bin/bash
# run_vqa2_small.sh — test the load-bearing-question fix at the SMALL rung, before any big run.
#
# The vqa1 text concepts measured AUROC 0.53-0.63 (a coin flip is 0.5). The diagnosis: the questions
# all asked for the same answer, so the model never needed to read them.
#
# vqa2 makes the questions matter ("is this footwear" -> yes/no, "name the object" -> the class).
# Pre-registered prediction: text-concept mean AUROC rises clearly above 0.70.
#
# SMALL FIRST. If it does not work at 20,000 images and 6 epochs, a big run would only cost more.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ vqa2 — SMALL rung (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --task vqa2 --data fashion --d 64 --epochs 6 --batch 256 --lr 0.3 --device cpu \
  --max-train "${V2_TRAIN:-20000}" --out vqa2_small 2>&1 | tail -60

echo
echo "############ vqa1 vs vqa2 — the text-concept AUROC that this was built to move ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"
for tag, out in (("vqa1 (questions identical)", "vlsmoke"), ("vqa2 (questions matter)", "vqa2_small")):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {tag}: not run")
        continue
    r = json.load(open(f))
    a = r["audit"]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    print(f"  {tag}")
    print(f"     text  concepts mean AUROC {np.mean(txt):.4f}   (each: "
          + ", ".join(f"{x:.3f}" for x in txt) + ")")
    print(f"     image concepts mean AUROC {np.mean(img):.4f}")
    print(f"     TEST acc {r['results']['readable']['history'][-1]['test_acc']*100:.2f}%")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
