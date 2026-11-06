#!/bin/bash
# run_image_seeds.sh — 3 seeds, because the project rule says a result must beat seed noise.
#
# Run 1 (seed 0): readable 83.49% vs plain 86.42% -> delta -2.93 points.
#
# Is -2.93 a real penalty or one seed's luck? The rule is explicit:
#   "Run every key result on 3 seeds. A win smaller than the seed noise is called 'no difference'."
#
# Pre-registered prediction: the penalty is REAL and REPRODUCES. It is not a win, so seed noise
# would have to be ~3 points wide to explain it, and text-side seeds have been much tighter than
# that. Expected mean delta around -2 to -3.5 points.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_image.py|train_chat.py|train_code.py" > /dev/null; do
  sleep 20
done

for SEED in 1 2; do
  echo
  echo "############################################################"
  echo "# IMAGE seed $SEED"
  echo "############################################################"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
    --d 64 --patch 7 --epochs "${IMG_EPOCHS:-24}" --batch 256 --lr 0.3 --lr-decay 0.5 \
    --lam-concept 0.3 --device gpu --seed "$SEED" --out "img_s${SEED}" 2>&1 \
    | grep -E "READABLE:|PLAIN   :|ep 24|-> [0-9]+/13|TEST" | tail -8
done

echo
echo "################################################################"
echo "# IMAGE — 3 SEEDS, mean +/- std"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/image"
rows = []
for out in ("img1", "img_s1", "img_s2"):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        print(f"  {out}: -- not run --")
        continue
    r = json.load(open(f))
    rr, pp = r["results"]["readable"], r["results"]["plain"]
    ra, pa = rr["history"][-1]["test_acc"] * 100, pp["history"][-1]["test_acc"] * 100
    rows.append((out, r["args"]["seed"], ra, pa, ra - pa, r["audit"]["n_pass"],
                 r["audit"]["n_concepts"]))
    print(f"  {out:<10} seed {r['args']['seed']}  readable {ra:6.2f}%  plain {pa:6.2f}%  "
          f"delta {ra-pa:+5.2f}   concepts {r['audit']['n_pass']}/{r['audit']['n_concepts']}")

if len(rows) >= 2:
    d = np.array([x[4] for x in rows])
    ra = np.array([x[2] for x in rows])
    pa = np.array([x[3] for x in rows])
    print()
    print(f"  readable  {ra.mean():.2f}% +/- {ra.std(ddof=0):.2f}")
    print(f"  plain     {pa.mean():.2f}% +/- {pa.std(ddof=0):.2f}")
    print(f"  DELTA     {d.mean():+.2f} +/- {d.std(ddof=0):.2f} points")
    print()
    verdict = "NO DIFFERENCE (inside seed noise)" if abs(d.mean()) < d.std(ddof=0) \
        else ("readable WINS" if d.mean() > 0 else "readable LOSES")
    print(f"  VERDICT: {verdict}")
    print(f"  (rule: a win smaller than the seed noise is called 'no difference')")
PY

cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
