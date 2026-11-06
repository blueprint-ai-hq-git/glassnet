#!/bin/bash
# THE TIED, FULL-RANK DIRECT PATH - the principled version, after the rank-16 path failed to move the talk test.
#
# WHY THIS VERSION: the low-rank run measured val 28.06% final / 28.56% peak against the baseline's 28.30% /
# 28.52% - identical within noise - and the train loss came out WORSE (3.58 vs 3.48), with the talk test still
# pure word salad. The reason is arithmetic: a rank-16 bottleneck cannot say WHICH of 12,002 tokens it was.
# In the toy test rank 8 worked because the vocabulary was 50 tokens. So now every position mixes its own
# D numbers into D numbers (full rank, D=256) and the readout is the model's OWN embedding matrix, TIED.
# Gradient check in float64 before trusting it: dmix 4.14e-09, tied readout 2.15e-09 (experiments/tied_path_check.py).
# Cost: W*D*D = 32*256*256 = 2,097,152 parameters, so about 5.9M total - still inside the 50M cap.
#
# PREDICTIONS, written BEFORE the run:
#  1. train loss should beat the low-rank run's 3.58 and come closer to the baseline's 3.48, because token
#     identity can finally flow into the answer at full rank. If it is worse than 3.58, this version is wrong.
#  2. honest masked val should hold at or above 28.30% (the baseline). A drop means the tied path is
#     interfering with the concept path and that is a finding against it.
#  3. THE TALK TEST DECIDES. Prediction: the text should be visibly less salad - real words, fewer fragments
#     like "5 * * sample input". If it is STILL pure salad, then the answer-layer bottleneck was NOT the whole
#     story, and the leading remaining explanation is that the model is too small for the data (the trainer
#     itself printed "tokens per parameter: 0.36 (UNDER-TRAINED)" for the 4.1M model). That would be tested
#     next by the same run at d=512, not by another guess.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED FULL-RANK DIRECT PATH (d=256, 12 ep, clean corpus) $(date '+%F %H:%M')"
  echo "compared against: baseline 28.30% final / 28.52% peak / loss 3.48"
  echo "                  lowrank-16 28.06% final / 28.56% peak / loss 3.58 / talk STILL SALAD"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 12 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 1500000 --out cl_d256_tied_12
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/DIRECT-TIED.txt 2>&1
