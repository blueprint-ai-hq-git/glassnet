#!/bin/bash
# run_agreement_seeds.sh -- the three-seed half of PLAN item (d2d). The seed-0 screen
# (lens/scripts/run_agreement_test.sh) measured the DEFAULT detector 'vs_probe' refusing 0.383 of 400 steps on
# the memorisable corpus against 0.013 of 237 on the learnable one. This project's standing rule is THREE SEEDS
# before anything is called a win, and the mechanism that produced that gap (A_train dipping below the probe's
# own coherence) works on a thin margin near the independence floor, so it is exactly the kind of result that
# must not be believed from one seed. These are the seeds 1 and 2 arms, same commands otherwise.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONPATH="$PWD/lens/scripts"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2
OUT=lens/runs/AGREEMENT-TEST.txt
SMALL="$PWD/data/chat/prepared_gate_small"
CLEAN="$PWD/data/chat/prepared_clean"
COMMON="--d 128 --window 32 --batch 256 --lr 0.4 --lam-concept 0.3 --device cpu --gates agreement"

{
  echo "=== AGREEMENT TEST SEEDS START $(date '+%F %H:%M:%S')"
  for S in 1 2; do
    echo "--- agree_m_vs_s${S}: M, DEFAULT detector vs_probe, seed ${S}"
    CHAT_DATA_DIR="$SMALL" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
      --epochs 80 --seed $S --out agree_m_vs_s${S}
    echo "agree_m_vs_s${S} EXIT=$?"
    echo "--- agree_l_vs_s${S}: L, DEFAULT detector vs_probe, seed ${S}"
    CHAT_DATA_DIR="$CLEAN" nice -n 19 ionice -c3 python3 -u lens/scripts/train_chat.py $COMMON \
      --epochs 3 --max-windows 20000 --seed $S --out agree_l_vs_s${S}
    echo "agree_l_vs_s${S} EXIT=$?"
  done
  echo "=== AGREEMENT TEST SEEDS DONE $(date '+%F %H:%M:%S')"
} >> "$OUT" 2>&1
