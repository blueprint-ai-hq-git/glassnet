#!/bin/bash
# run_medical_seeds.sh — 3 seeds on the colour-concepts config, because one run is not a result.
#
# The claim to verify: adding colour concepts moves image-concept fidelity from 10/16 with mean
# AUROC 0.6760 to 27/31 with mean AUROC 0.7691, for +1.59 accuracy points.
#
# The project rule: "Run every key result on 3 seeds. A win smaller than the seed noise is called
# 'no difference'." The accuracy change (+1.59) is small enough that seed noise could explain it. The
# FIDELITY change (10/16 -> 27/31, AUROC +0.0932) is much larger, but both need checking.
#
# Pre-registered predictions:
#   1. The image-concept AUROC gain REPRODUCES, mean above 0.73 across seeds. This is the change the
#      run was designed to produce, and it is large.
#   2. The accuracy gain is WITHIN SEED NOISE. +1.59 points should not survive 3 seeds, and if it
#      does not, the honest statement is "colour concepts did not measurably change accuracy" --
#      which is fine, because accuracy was never the claim.
#   3. Image concepts stay well above 10/16 on every seed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

for SEED in 1 2; do
  echo
  echo "############################################################"
  echo "# medical colour-concepts, seed $SEED"
  echo "############################################################"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_vl.py \
    --task vqa3 --data pathmnist --d 64 --epochs "${MED_EPOCHS:-8}" --batch 256 --lr 0.3 \
    --adaptive --color --colour-concepts --lam-concept 0.15 --device cpu \
    --max-train "${MED_TRAIN:-20000}" --seed "$SEED" --out "med_cc_s${SEED}" 2>&1 \
    | grep -E "image concepts|text  concepts|TEST|-> [0-9]+/|majority-answer" | tail -6
done

echo
echo "################################################################"
echo "# 3 SEEDS — is the fidelity gain real, and is the accuracy gain noise?"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/vl"


def stat(out):
    f = os.path.join(P, f"{out}_log.json")
    if not os.path.exists(f):
        return None
    r = json.load(open(f)); a = r["audit"]
    img = [c["auroc"] for c in a["concepts"] if c["side"] == "image" and c["auroc"] is not None]
    txt = [c["auroc"] for c in a["concepts"] if c["side"] == "text " and c["auroc"] is not None]
    return {"seed": r["args"]["seed"],
            "acc": r["results"]["readable"]["history"][-1]["test_acc"] * 100,
            "plain": r["results"]["plain"]["history"][-1]["test_acc"] * 100,
            "img": float(np.mean(img)), "txt": float(np.mean(txt)),
            "ip": (a["img_pass"], a["img_total"]), "tp": (a["txt_pass"], a["txt_total"])}


base = stat("med_color")      # COLOUR without colour concepts (seed 0)
runs = [base] + [stat(f"med_cc_s{s}") for s in (1, 2)]
runs = [r for r in runs if r]
cc = [stat("med_cc")] + [stat(f"med_cc_s{s}") for s in (1, 2)]
cc = [r for r in cc if r]

print("  WITHOUT colour concepts (seed 0 only):")
if base:
    print(f"    acc {base['acc']:.2f}%  img AUROC {base['img']:.4f}  "
          f"img concepts {base['ip'][0]}/{base['ip'][1]}")
print()
print("  WITH colour concepts:")
for r in cc:
    print(f"    seed {r['seed']}  acc {r['acc']:6.2f}%  img AUROC {r['img']:.4f}  "
          f"txt AUROC {r['txt']:.4f}  img concepts {r['ip'][0]}/{r['ip'][1]}  "
          f"plain {r['plain']:.2f}%")
if len(cc) >= 2:
    acc = np.array([r["acc"] for r in cc])
    au = np.array([r["img"] for r in cc])
    print()
    print(f"  3-seed accuracy     {acc.mean():.2f}% +/- {acc.std(ddof=0):.2f}")
    print(f"  3-seed img AUROC    {au.mean():.4f} +/- {au.std(ddof=0):.4f}")
    if base:
        d_acc = acc.mean() - base["acc"]
        d_au = au.mean() - base["img"]
        print()
        print(f"  accuracy change vs no-colour-concepts: {d_acc:+.2f} "
              f"(seed noise {acc.std(ddof=0):.2f})")
        print(f"    -> {'REAL' if abs(d_acc) > acc.std(ddof=0) else 'WITHIN SEED NOISE (no difference)'}")
        print(f"  img AUROC change                     : {d_au:+.4f} "
              f"(seed noise {au.std(ddof=0):.4f})")
        print(f"    -> {'REAL' if abs(d_au) > au.std(ddof=0) else 'WITHIN SEED NOISE'}")
PY
cp STATUS.md LOG.md /home/yahya/ 2>/dev/null
echo "docs copied"
