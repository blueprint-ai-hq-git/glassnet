#!/bin/bash
# Finish the combined corpus: steps 2-4 only. Step 1 (the build) already succeeded and produced
# data/chat/prepared_kn2 with 979 MB of ids, so re-running it would waste ten minutes for nothing.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=/tmp/mix_finish.log
{
  echo "=== COMBINED CORPUS: FINISH $(date '+%F %H:%M')"
  echo "--- base split for the new corpus"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/make_split.py data/chat/prepared_kn2 0.02 0
  echo "exit=$?"
  echo "--- merge chat + knowledge/maths/code"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/merge_corpora.py \
    data/chat/prepared_mix data/chat/prepared_ext_honest data/chat/prepared_kn2
  echo "exit=$?"
  echo "--- standard control tokens (ChatML)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/add_control_tokens.py \
    data/chat/prepared_mix data/chat/prepared_mix_ctrl
  echo "exit=$?"
  echo "--- split, then make val disjoint from train at the training window"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/make_split.py data/chat/prepared_mix_ctrl 0.02 0
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32
  echo "exit=$?"
  echo "--- sizes:"
  du -sh data/chat/prepared_kn2 data/chat/prepared_mix data/chat/prepared_mix_ctrl \
         data/chat/prepared_mix_ctrl_honest 2>/dev/null
  echo "=== DONE $(date '+%F %H:%M')"
} > "$LOG" 2>&1
tail -30 "$LOG"
