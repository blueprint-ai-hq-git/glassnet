#!/bin/bash
# Diagnostic runner: why does the mask-fixed ep1 checkpoint emit <unk>? (read-only, CPU)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
{
  echo "=== WHY <unk>? diagnostic $(date '+%F %H:%M') (read-only, no stored results touched) ==="
  echo "hypothesis H71: <unk> is a real argmax at turn-start prompts, logits finite, no degenerate output row."
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/_inspect_unk_argmax.py
  echo "exit=$?"
} >> lens/runs/UNK-ARGMAX-DIAG.txt 2>&1
