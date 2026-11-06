#!/bin/bash
# THE DATA JUDGE RUN (plan ask #4): 30k real train windows + 4 controlled corruptions each,
# 10k held-out val windows, 30 epochs, CPU (the GPU is busy with the distillation).
# Gates pre-registered in data_judge.py's header: G1 AUROC >= 0.90 vs corrupted; G2 the MLP
# must not beat the readable linear by > 0.02; G3 must beat the best single feature by +0.05;
# G4 agreement with the reference model's worst-loss quartile >= 0.65.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CD_VOCAB=12000
export DJ_N_TRAIN=30000 DJ_N_VAL=10000 DJ_EPOCHS=30 DJ_FEATSET=v2
exec nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/data_judge.py
