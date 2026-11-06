#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
.venv-gpu/bin/python3 -u - <<'PY'
import sys, json
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import variants as V
vr = V.build_default()
net, meta = vr._need("see_talk")
print("see_talk ship:", vr._paths.get("see_talk", "?").split("/")[-1],
      "K =", len(net.concepts), "mirror block:", bool(getattr(net, "concept_mirror", False)))
img = V._load_image_28("/home/yahya/cam_frame13.png")
vocab = meta["vocab"]                       # the SHIP's own word list (the caption task's V is
wt = [i for i, w in enumerate(vocab) if w not in ("<s>", "<cap>", "<shift>")]
ids = (wt * 2)[:6]                           # exactly the ship's window W=6
r = vr.see_and_talk(img, ids)
print("see_and_talk on a real photo:", r)
PY
