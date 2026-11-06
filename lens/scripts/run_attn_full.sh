#!/bin/bash
# THE REAL ATTENTION ARM: the one architectural lever this project has never trained at scale.
# Predictions P82a-f written BEFORE this line was queued: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt
# The smoke that was supposed to license it came out INCONCLUSIVE and is graded honestly in
# lens/runs/GRADE-P79-SMOKE.txt (P79a held, P79b's letter held with attention 2.83 points WORSE, P79c
# FALSIFIED: the control landed at 9.76% against a predicted 12-22% band, which voids P79b by its own wording).
#
# EXACT BEHAVIOUR AGAINST THE STORED TWIN (lens/scripts/run_tied_full169_maskfix.sh, out ct_d256_full169_fix3):
# identical flags, ONE addition (--attention) and the checkpoint interval. The twin's finished numbers:
#   ep3 train 38.57% held-out by-position 35.26% legacy 26.31% (gap 3.31 points, TIED-FULLCORPUS-169M-MASKFIX.txt)
# Its own --out tag, its own log, so neither the twin nor any stored file is touched.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== THE REAL ATTENTION ARM (P82a-f) $(date '+%F %H:%M')"
  echo "predictions: lens/runs/ATTN-FULL-PREDICTIONS-P82.txt (written before this line was queued)"
  echo "twin (no attention, same everything): ct_d256_full169_fix3, ep3 train 38.57% / by-position 35.26% /"
  echo "  legacy 26.31% / gap 3.31 points -- lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt; talk 6 hedges 0 answers"
  echo "smoke verdict: INCONCLUSIVE (lens/runs/GRADE-P79-SMOKE.txt) -- this arm is run because the smoke could"
  echo "  not decide and the arm is cheap (~1.5 h), not because the smoke passed"
  echo "smoke's weak direction, on the record: attention 6.93% vs control 9.76% by position at 200k windows/1 ep"
  echo "checkpoint every 1800 s (P81a-f verified); resumes with --resume-mid auto if it is ever re-queued"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --attention \
    --ckpt-seconds 1800 --out ct_d256_full169_attn
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
  echo "NEXT, on a completed arm: pad-free talk test on ct_d256_full169_attn_best.npz with"
  echo "lens/scripts/run_talk_finalbest.sh pointed at it (env TALK_FAIR_CKPT), and the P65d/P65c context and"
  echo "free-running numbers compared against the twin's 33.07-point context drop and 3.9% step-8."
} >> lens/runs/ATTN-FULL.txt 2>&1
