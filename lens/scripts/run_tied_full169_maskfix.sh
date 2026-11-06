#!/bin/bash
# CRITICAL PATH 1, UNCAPPED, RE-MEASURED WITH THE FIXED ASSISTANT MASK (2026-10-22 18:0x self-wake).
#
# WHY THIS EXISTS AT ALL: BUGS.md [BUG-MASK-TOKEN-ID]. `--assistant-only` indexed a POSITIONAL array with
# TOKEN IDS (`is_asst[Ytr]`), so (a) the loss weights were an arbitrary function of the target's token id
# instead of whose turn the token is in, i.e. human-turn text was trained on with weight 1, and (b) the
# "MASKED" held-out column was a fixed vocabulary subset (44,418 of 87,035 windows, 33.9% overlap with the
# real assistant turns) and roughly 2x optimistic (35.82% legacy against 17.95% by position on the same
# checkpoint: lens/runs/MASK-INDEX-BUG.txt). Fixed, verified: lens/runs/MASKFIX-SMOKE.txt, MASKFIX-RESULTS.txt.
#
# THE RUN IN FLIGHT KEEPS ITS OLD BEHAVIOUR: ct_d256_full169_3 (started 17:33 by the watchdog) loaded the
# pre-change file, so its numbers are LEGACY and are labelled as such. THIS arm is the same recipe with the
# fixed mask, in its OWN file (two turns editing one runner is how experiments got mixed before) and with its
# own --out tag so no checkpoint of the earlier arm is overwritten.
#
# PREDICTIONS, written BEFORE this line is queued (P67a-e), so the result cannot be interpreted after the fact:
#  P67a THE COLUMN TO READ IS ASST(by-position). If the weight bug was costing real quality, this run's ASST
#       held-out number lands ABOVE the stored legacy-equivalent (the stored arms' by-position accuracy has
#       never been measured for any arm except ct_d256_6M_6ep_best = 17.95%); I predict 18-24%.
#  P67b the LEGACY column still prints at about TWICE the ASST column. That doubling is arithmetic, not
#       progress, and reporting it as a result would be the verdict-check bug this project keeps hitting.
#  P67c peak RSS 6.5 GB or lower, read from this arm's own scope MemoryPeak (P64d, re-checked).
#  P67d the first epoch line IS written (P64g, re-checked; the width fix is in this path).
#  P67e THE TALK TEST IS STILL WORD SALAD: the mask fix changes the weights and the metric, nothing in the
#       decoder, so my honest prediction is that generation is unchanged in kind. If it is still salad, the
#       next lever is NOT the mask and not the data volume - it is the context mechanism itself
#       (lens/scripts/talk_fair_test.py measures whether the model reads its own 32-token context at all).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS (16,878,132 windows) d=256 3 epochs, FIXED ASSISTANT MASK $(date '+%F %H:%M') ==="
  echo "mask bug: BUGS.md [BUG-MASK-TOKEN-ID]; evidence lens/runs/MASK-INDEX-BUG.txt; fix verified lens/runs/MASKFIX-RESULTS.txt"
  echo "reference (LEGACY by-token-id numbers, NOT comparable to this run's ASST column): 34.92% (6M/d256), 33.94% (4M/d512), 35.82% (6M/6ep) peak"
  echo "reference (BY POSITION, the fixed column, measured on ct_d256_6M_6ep_best): 17.95% over 58,191 windows"
  echo "predictions P67a-e are in this file's header"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --out ct_d256_full169_fix3
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TIED-FULLCORPUS-169M-MASKFIX.txt 2>&1
