#!/bin/bash
# H2 PRE-FLIGHT (P103a-d). WIRING ONLY: does --lam-concept 0 reach the loss, and does the concept column read
# 0.0000 while the same command with 0.3 reads non-zero? Predictions: lens/runs/H2-PREFLIGHT-PREDICTIONS-P103.txt,
# written BEFORE this file existed. Output lens/runs/H2-PREFLIGHT.txt (new file; nothing stored is written to).
# CPU ONLY (--device cpu), tiny corpus (data/chat/prepared_gate_small: ids_v12000.npy is 243 KB), d=64, 3,000
# windows — deliberately NOT the real corpus, because one train_chat process loads its whole corpus cache
# [BUG-TRAINCHAT-MEMORY-IS-THE-CORPUS] and a training arm is on the GPU right now.
# NO QUALITY, MEMORY OR TALK CLAIM COMES OUT OF THIS FILE.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/H2-PREFLIGHT.txt
{
  echo "================================================================================================"
  echo "H2 PRE-FLIGHT (wiring only: does --lam-concept 0 reach the loss?)  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/H2-PREFLIGHT-PREDICTIONS-P103.txt (written before this file existed)"
  echo "corpus: data/chat/prepared_gate_small (tiny, on purpose)   d=64   window 32   device cpu   3000 windows"
  echo "--- RAM before ---"; free -h | head -2
  echo "================================================================================================"
  for LC in 0 0.3; do
    echo "--- run with --lam-concept $LC ---"
    CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small" \
    taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
      --d 64 --window 32 --epochs 1 --batch 64 --lr 0.4 --lam-concept "$LC" --lam-act 0.0 --seed 0 \
      --device cpu --assistant-only --train-turn-end --direct --direct-mode tied \
      --max-windows 3000 --method standard --out "h2pf_lc${LC}" 2>&1 | grep -E "^  ep |model:|CONCEPT|concept|Traceback|Error|error" | head -20
    echo "RUN-EXIT(pipe status)=$?  for --lam-concept $LC   $(date '+%F %H:%M:%S')"
  done
  echo "--- RAM after ---"; free -h | head -2
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
  echo "NOTE: d=64 wiring evidence only. The full-scale control is P102b (the concept column must read 0.0000 on"
  echo "      the real corpus), and no quality number here is a finding."
} >> "$LOG" 2>&1
echo "--- pre-flight log tail ---"
tail -n 25 "$LOG"
