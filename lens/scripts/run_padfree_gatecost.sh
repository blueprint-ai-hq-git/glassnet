#!/bin/bash
# PAD-FREE TALK INSTRUMENT ON THE THREE GATE-COST-ABLATION CHECKPOINTS (P87e and P87g; grading predictions
# lens/runs/GATE-COST-GRADE-PREDICTIONS-P97.txt, P97d/e/f).
#
# WHY THIS RUN EXISTS: P87e (the decider) says every arm here still fails coherence, and P87g says ARM 1's <unk>
# share should be at or below 20% against the reference arm's 37.5%. Both are answered by ONE run of the SAME
# instrument on the three checkpoints, so the three numbers cannot come from three differently-set-up runs.
# THE REFERENCE ARM IS RE-MEASURED IN THIS RUN ON PURPOSE, as the instrument's own control: it must reproduce
# lens/runs/PADFREE-TALK-OURS3M.txt's 37.5% mean <unk> (P97d).
#
# READ-ONLY: this script writes ONLY lens/runs/PADFREE-TALK-GATECOST.txt (PADFREE_OUT) and
# lens/runs/PADFREE-TALK-GATECOST-RUN.txt. It never edits talk_fair_v3_ship.py (the CKPT_LIST env is the v3
# instrument's own documented interface), never trains, never uses the GPU, and touches no stored number.
# CPU only, 2 BLAS threads, nice 19, ionice -c3, pinned to cores 2-3 (the live trainer owns 0-1).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export PADFREE_OUT="$PWD/lens/runs/PADFREE-TALK-GATECOST.txt"
export CKPT_LIST="ARM1 gates-OFF old loop (plain SGD, held-out by-position 29.41% ep2, train 31.16%)=lens/runs/chat/ours3M_ctrl_std_best.npz;ARM2 gates-OFF relgrad step rule (held-out by-position 24.54% ep2, train 22.38%)=lens/runs/chat/ours3M_relgrad_nogates_best.npz;REFERENCE method-ours WITH both gates (held-out by-position 12.68% ep1, the stored arm P87b's control)=lens/runs/chat/ours_d256_3M_2ep_best.npz"
{
  echo "=== PADFREE GATE-COST runner $(date '+%F %H:%M:%S') ==="
  echo "predictions: lens/runs/GATE-COST-GRADE-PREDICTIONS-P97.txt (P97d/e/f) written BEFORE this run"
  echo "output: lens/runs/PADFREE-TALK-GATECOST.txt"
  echo "checkpoints (mtime + size recorded BEFORE the run so the read-only claim can be checked after it):"
  stat -c '  %y  %s bytes  %n' lens/runs/chat/ours3M_ctrl_std_best.npz lens/runs/chat/ours3M_relgrad_nogates_best.npz lens/runs/chat/ours_d256_3M_2ep_best.npz
  start=$(date +%s)
  taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_v3_ship.py
  rc=$?
  echo "python exit=$rc   wall=$(( $(date +%s) - start ))s"
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> lens/runs/PADFREE-TALK-GATECOST-RUN.txt 2>&1
