#!/bin/bash
# Rebuild the disjoint val split for the control-token corpus (294,225,090 tokens). The stale destination
# from the morning is moved aside by repair_split itself, which now validates bytes instead of trusting
# existence. CPU only, memory-lean (fingerprints), so it runs alongside the GPU training.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
{
  echo "=== REPAIR CTRL CORPUS $(date '+%F %H:%M')"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/repair_split.py \
    data/chat/prepared_mix_ctrl data/chat/prepared_mix_ctrl_honest 32
  echo "exit=$?"
  .venv-gpu/bin/python3 -c "
import numpy as np
for f in ('ids_v12000.npy','bounds_v12000.npy','split_v12000.npy'):
    a=np.load('data/chat/prepared_mix_ctrl_honest/'+f, mmap_mode='r')
    print(f'  {f:20s} {str(a.shape):18s} {a.dtype}')"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/REPAIR-CTRL.txt 2>&1
