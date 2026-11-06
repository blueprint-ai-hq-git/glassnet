#!/bin/bash
# The warm tier in the generation loop (PLAN item (c2)).
# PREDICTIONS, written before the run: lens/runs/WARM-DECODE-PREDICTIONS.txt (P38a..P38i)
# Output: lens/runs/WARM-DECODE.txt (APPENDED, never overwritten: earlier blocks are earlier passes).
#   part 1: the identity/cost/size/readability measurements in the decoder
#   part 2: the same store over real HTTP in the server (lens/scripts/warm_serve_probe.py)
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR=data/chat/prepared_mix_honest
{
  echo "=== WARM TIER IN THE GENERATION LOOP $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 python3 -u lens/scripts/warm_decode_test.py
  echo "decoder exit=$?"
  nice -n 19 ionice -c3 timeout 500 python3 -u lens/scripts/warm_serve_probe.py
  echo "server probe exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/WARM-DECODE.txt 2>&1
