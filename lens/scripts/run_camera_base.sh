#!/bin/bash
# run_camera_base.sh — train the VL checkpoint the virtual camera will use, then TEST the camera.
#
# Runs on CPU so it does not fight whatever is on the GPU. The camera is a pipeline test: does a
# frame go in and the model's own label come out, drawn onto real video.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "############ 1. train a VL checkpoint (CPU) ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
  --data fashion --d 64 --epochs 6 --batch 256 --lr 0.3 --device cpu \
  --max-train "${CAM_TRAIN:-20000}" --out cam_base 2>&1 | tail -22

CKPT=lens/runs/vl/cam_base_ep6.npz
echo
echo "############ 2. VIRTUAL CAMERA on real data ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/virtual_camera.py \
  --model "$CKPT" --source fashion --frames 60 --fps 8 --device cpu --out cam_fashion.mp4 2>&1 | tail -20

echo
echo "############ 3. VIRTUAL CAMERA on synthetic scenes ############"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/virtual_camera.py \
  --model "$CKPT" --source synthetic --frames 40 --fps 8 --device cpu --out cam_synthetic.mp4 2>&1 | tail -14

echo
echo "############ 4. what came out ############"
ls -la lens/runs/video/*.mp4 2>/dev/null
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/video"
for f in ("cam_fashion_log.json", "cam_synthetic_log.json"):
    p = os.path.join(P, f)
    if os.path.exists(p):
        r = json.load(open(p))
        print(f"  {f}: {len(r['frames'])} frames, correct {r['correct']}/{r['total']}")
        for row in r["frames"][:3]:
            print(f"      frame {row['frame']}: says {row['answer']} ({row['conf']*100:.1f}%)  "
                  f"concepts {[c[0] for c in row['top_concepts'][:3]]}")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
