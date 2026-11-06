#!/bin/bash
# THE COST OF THE GATES, AT MATCHED SCALE. Two arms, ONE script, so the watchdog can never have two trainers on
# the GPU at once (that is what OOM-killed runs on this machine before).
# PREDICTIONS P87a-h were written BEFORE this file was queued: lens/runs/GATE-COST-PREDICTIONS-P87.txt.
# Log: lens/runs/GATE-COST.txt   Own --out tags: ours3M_ctrl_std, ours3M_relgrad_nogates (nothing overwritten).
#
# REFERENCE ARM (already on disk, NOT re-run): --method ours, --out ours_d256_3M_2ep, ep1 held-out BY POSITION
# 12.68% (lens/runs/chat/ours_d256_3M_2ep_history.jsonl), gap gate refused 84.3% of updates (GRADE-P86.txt).
# ARM 1 = the old loop at the same cap/epochs/seed -> isolates the WHOLE method.
# ARM 2 = the same step rule with the gates OFF -> isolates the TWO GATES from the step rule.
# NOTE the flags deliberately: --method ours would FORCE a.gates='gap+agreement' (train_chat.py block at
# 'OUR METHOD, AS ONE SWITCH'), so ARM 2 is spelled as --method standard --opt relgrad --gates off and its log
# header says so -- the trainer's own printed label for ARM 2 will read 'standard (the old loop ...)', which is
# wrong for that arm and is called out in the log here rather than left to be misread.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
COMMON="--d 256 --window 32 --epochs 2 --batch 1024 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
--seed 0 --device gpu --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 3000000"
{
  echo "=== GATE-COST ABLATION (P87a-h) $(date '+%F %H:%M:%S') ==="
  echo "predictions: lens/runs/GATE-COST-PREDICTIONS-P87.txt (written before this line was queued)"
  echo "reference arm (on disk, not re-run): ours_d256_3M_2ep ep1 by-position 12.68%, gates refused 84.3%"
  echo "  sources: lens/runs/chat/ours_d256_3M_2ep_history.jsonl + lens/runs/GRADE-P86.txt"
  echo "--- ARM 1: the old loop at matched scale (plain SGD, gates off) ---"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $COMMON --method standard --out ours3M_ctrl_std
  echo "ARM1 exit=$?  $(date '+%F %H:%M:%S')"
  echo "--- ARM 2: the same step rule, GATES OFF (isolates the gates from relgrad) ---"
  echo "    (the trainer's own METHOD label will say 'standard'; the flags below are what it really runs)"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $COMMON \
    --method standard --opt relgrad --opt-lr 0.0441256 --gates off --out ours3M_relgrad_nogates
  echo "ARM2 exit=$?  $(date '+%F %H:%M:%S')"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
  echo "NEXT: grade P87a-g from lens/runs/chat/ours3M_ctrl_std_history.jsonl and"
  echo "      lens/runs/chat/ours3M_relgrad_nogates_history.jsonl, then run the same pad-free instrument on both"
  echo "      best checkpoints (CKPT_LIST env, new PADFREE_OUT) for P87e and P87g."
} >> lens/runs/GATE-COST.txt 2>&1
