#!/bin/bash
# EXPERIMENT v12k: the ONE change from best1 — vocabulary 3,500 -> 12,000. Everything else identical
# (d=128, W=16, 24 epochs, lr 0.4 with decay 0.5, lam_concept 0.3, 1.5M windows).
#
# WHY THIS ONE CHANGE: the baseline measurement showed a tradeoff nobody had both halves of --
#   best1 (V=3500): 6.04 bits/token, 14.5% <unk> in replies   <- good recipe, small vocab
#   big1  (V=12000): 7.21 bits/token,  8.5% <unk> in replies   <- big vocab, BAD recipe (W=8, 12ep, no decay)
# Neither has both. This run is the good recipe WITH the big vocab.
#
# PRE-REGISTERED (written BEFORE the run, binary):
#   v1  <unk> rate in generated replies FALLS below best1's 14.5%   (the point of the big vocab)
#   v2  bits/token stays within +0.30 of best1's 6.04              (we accept a small ppl cost for
#                                                                    fewer unks, but not a collapse)
#   v3  both v1 and v2 pass -> v12k is the new talk model, and the decoder sweep re-runs on it.
#       v1 passes but v2 fails -> vocab cost too high; report and stop (do NOT hide it).
#       v1 fails -> big vocab does not help once the recipe is fixed; that kills this direction
#                   and we go to window length instead. Reported as a FAILURE, not spun.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
CD_VOCAB=12000 TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 16 --epochs 24 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --device gpu --out v12k \
  --max-windows 1500000 2>&1 | grep -E "^  ep |final VAL|saved|Error|Traceback" | tail -30
