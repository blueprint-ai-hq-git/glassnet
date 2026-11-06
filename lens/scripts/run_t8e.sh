#!/bin/bash
# T8e (code, final lever): ONE change from T8d — lam-concept 0.3 -> 0.6 (double the pressure
# on the concept layer toward its formulas; the 25-set's boundary pairs sit at margins
# 0.05-0.08 and need ~2x push). Same 25-set, ep32, d192, seed 0, gpu.
# Pre-registered: strict 25/25 AND val >= 39. If it fails -> declare code corpus-limited at
# ~92-93% strict with the weak named (no more drop loops: 35->25 concepts already sacrificed).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CD_VOCAB=32000 CODE_DOC=clean
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_code.py \
  --d 192 --window 16 --epochs 32 --batch 256 --lr 0.4 --lr-decay 0.5 \
  --lam-concept 0.6 --seed 0 --device gpu --out t8e_code25_lam06 \
  --max-windows 1200000