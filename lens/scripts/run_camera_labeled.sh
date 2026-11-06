#!/bin/bash
# run_camera_best.sh — the virtual camera with the BEST VL model, not the weak one.
#
# The first camera video used `cam_base_ep6.npz` -- the vqa1-era model whose text concepts sat at a
# coin flip (0.53-0.63) and which scored 69.47%. This uses `vqa2_small_ep6.npz`, the one whose
# questions actually matter (text concepts 0.9640) and which scored 88.13%.
#
# So the concepts drawn on each frame are the ones the model genuinely uses, not a weak model's guess.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ virtual camera — best VL model (vqa2, 88.13%) ############"
#  NOTE: --out needs the .mp4 extension. Without it ffmpeg cannot infer the container format, the
#  pipe dies, and the failure surfaces as `BrokenPipeError: [Errno 32] Broken pipe` at write time --
#  which looks like a camera bug and is actually a missing file extension.
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/virtual_camera.py \
  --model lens/runs/vl/vqa2_small_ep6_vocabfix.npz --source fashion --frames "${FRAMES:-120}" \
  --task vqa2 --device cpu --out cam_labeled.mp4 2>&1 | tail -22

echo
echo "############ the first video vs this one ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/video"
for tag, f in (("first (weak model)", "cam_fashion_log.json"), ("this (best model)", "cam_labeled_log.json")):
    p = os.path.join(P, f)
    if not os.path.exists(p):
        print(f"  {tag:<22} not found"); continue
    d = json.load(open(p))
    fr = d["frames"]
    conf = sum(x["conf"] for x in fr) / len(fr)
    print(f"  {tag:<22} {d['correct']}/{d['total']} correct "
          f"({d['correct']/d['total']*100:.1f}%)   mean confidence {conf*100:.1f}%   "
          f"{len(fr)} frames")
mp4 = os.path.join(P, "cam_labeled.mp4")
print()
if os.path.exists(mp4):
    print(f"  output: {mp4}  ({os.path.getsize(mp4)/1024:.0f} KB)")
PY
cp lens/runs/video/cam_labeled.mp4 /home/yahya/ 2>/dev/null && echo "copied to /home/yahya/cam_best.mp4"
