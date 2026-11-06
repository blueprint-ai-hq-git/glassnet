#!/bin/bash
# CRITICAL PATH 2, ON THE FIXED CODE AND THE FULL CORPUS (2026-10-22 18:5x self-wake).
#
# WHAT IT IS: the capacity test - same recipe as CRITICAL PATH 1 (tied direct path, best-val checkpointing) but
# d=512, on the FULL 16,900,000-window cap and with the FIXED assistant mask (BUGS.md [BUG-MASK-TOKEN-ID]).
#
# WHY IT IS A NEW FILE: run_tied_d512.sh is the old 4M-window / legacy-mask arm. Its stored numbers are legacy
# (lens/runs/TIED-D512.txt) and re-running that line would have produced numbers nobody can compare to the
# mask-fix runs. Two turns editing one runner file is how experiments got mixed together before (see the note in
# run_tied_full169.sh), so this arm gets its own file, its own --out tag (ct_d512_full169_fix3) and its own log
# file. Nothing existing is overwritten or deleted.
#
# PREDICTIONS P69a-e were written BEFORE this file existed: lens/runs/TIED-D512-FULL-PREDICTIONS-P69.txt
#
# MEMORY: the chunked window sampler (lens/scripts/chat_data.py, verified bit-identical 12/12,
# lens/runs/MEMFIX-TEST.txt) is what makes the full corpus buildable - the earlier d=512 full-data attempt
# peaked at 11.2 GB on the pre-fix code. The watchdog gives this job its own scope with MemoryMax=10G.
#
# IT IS QUEUED, NOT RUN INLINE: the GPU is already occupied by the mask-fix d=256 full-corpus arm
# (ct_d256_full169_fix3, pid 1184011). Two trainers at once caused an OOM kill on this machine before.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS (16,900,000 windows) d=512 3 epochs, FIXED ASSISTANT MASK $(date '+%F %H:%M') ==="
  echo "predictions: lens/runs/TIED-D512-FULL-PREDICTIONS-P69.txt (P69a-e); mask fix: lens/runs/MASK-INDEX-BUG.txt"
  echo "reference: d=512 4M windows LEGACY mask 33.94% (lens/runs/TIED-D512.txt) - not comparable to the ASST column below"
  echo "declared confound: this arm uses batch 128 / lr 0.3 (the established d=512 settings) where the d=256 full arm used batch 256 / lr 0.4"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 512 --window 32 --epochs 3 --batch 128 --lr 0.3 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --out ct_d512_full169_fix3
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TIED-D512-FULL.txt 2>&1
