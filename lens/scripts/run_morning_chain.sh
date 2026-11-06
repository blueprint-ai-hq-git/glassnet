#!/bin/bash
# MORNING CHAIN - resume exactly where the night stopped, skipping anything already finished.
#
# Where it stopped: the merge and the ChatML control tokens are DONE (data/chat/prepared_mix_ctrl exists),
# the disjoint-split repair is NOT (data/chat/prepared_mix_ctrl_honest is missing), and therefore neither the
# small smart model on the combined corpus nor the big rungs ever ran. Every step here is skipped if its
# output already exists, and every step prints its exit code, so a silent kill cannot hide again.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=4
PY=.venv-gpu/bin/python3
OUT=lens/runs/MORNING-CHAIN.txt
{
  echo "=== MORNING CHAIN $(date '+%F %H:%M')"
  echo "memory at start: $(free -g | awk 'NR==2{print $3"G used of "$2"G"}')  disk free: $(df -h . | tail -1 | awk '{print $4}')"

  if [ -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ]; then
    echo "--- 1/4 corpus already repaired, skipping"
  else
    echo "--- 1/4 make the held-out split disjoint from train at the training window (32)"
    nice -n 19 ionice -c3 $PY -u lens/scripts/repair_split.py \
      data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32
    echo "exit=$?"
    [ -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ] || {
      echo "## CHAIN STOPPED: the repair produced no corpus, so nothing downstream can run"; exit 1; }
    du -sh data/chat/prepared_mix_ctrl_honest
  fi

  echo "--- 2/4 small smart model on the combined corpus (d=256, about 4M parameters)"
  echo "predictions registered in run_mix_small.sh: MMLU above 26.4%, code above 4.0%, GSM8K still near zero,"
  echo "chat accuracy dips, readability still FAILS, P(<|im_end|>) above zero"
  bash lens/scripts/run_mix_small.sh
  echo "exit=$?"

  echo "--- 3/4 big rungs d=1536 / 2048 / 3072 (about 21M / 28M / 41M parameters, inside the 50M cap)"
  echo "predictions: smartness and accuracy improve with size; readability still FAILS; d=3072 may be the first"
  echo "real VRAM limit on a 6 GB card and would be reported as such"
  bash lens/scripts/run_ladder_big.sh
  echo "exit=$?"

  echo "--- 4/4 summary of what this chain produced"
  for f in lens/runs/MIX-SMALL.txt lens/runs/LADDER-BIG.txt; do
    [ -f "$f" ] || continue
    echo "  == $(basename $f)"
    grep -E "^## |MMLU|GSM8K|code|READABILITY GATE|false fires|assistant-masked|WORD SALAD|VERDICT|INSIDE|OVER" "$f" \
      | tail -12 | sed "s/^/     /"
  done
  echo "=== DONE $(date '+%F %H:%M')"
} >> "$OUT" 2>&1
