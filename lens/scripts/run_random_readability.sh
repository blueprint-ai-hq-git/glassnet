#!/bin/bash
# The readability gate on RANDOM held-out windows, for a given ship (default: our best honest model).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"
SHIP="${1:-lens/runs/chat/sb_S0_d256_noattn_ep96.npz}"
export SHIP
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/random_readability.py
echo "exit=$?"