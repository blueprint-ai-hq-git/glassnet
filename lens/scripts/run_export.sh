#!/bin/bash
# Export a ship in the normal formats and verify every file by reading it back.
# safetensors comes from the CUDA venv (the only place it is installed); the model itself is loaded
# on CPU, so nothing here touches the GPU.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
SHIP="${1:-lens/runs/chat/sb_S0_d256_noattn_ep96.npz}"
nice -n 19 ionice -c3 /home/yahya/hermes-env/bin/python3 -u lens/export.py "$SHIP" lens/runs/export
echo "exit=$?"