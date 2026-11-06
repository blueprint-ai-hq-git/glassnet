#!/bin/bash
# Run the turn-boundary probe on the ships we care about: the arm trained WITH --train-turn-end and the
# earlier one trained without it, so the flag's real effect is measured, not assumed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/turnend_probe.py \
  lens/runs/chat/te_T1_turnend_d256_24_ep24.npz lens/runs/chat/sb_S0_d256_noattn_ep96.npz 2>&1 \
  | tee lens/runs/TURNEND-PROBE.txt
echo "exit=$?"