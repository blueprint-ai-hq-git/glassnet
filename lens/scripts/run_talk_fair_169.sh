#!/bin/bash
# THE FAIR TALK TEST ON THE BIGGEST-DATA CHECKPOINT (2026-10-22 18:4x self-wake).
#
# WHY: ct_d256_full169_3_best.npz is the only checkpoint trained on the FULL 16,878,132-window corpus
# (tied direct path, d=256, 3 epochs, legacy by-token-id mask), and it finished at 18:26 exit 0
# (lens/runs/TIED-FULLCORPUS-169M.txt). No talk test has ever been run on it, and plan priority 4 says the
# talk test DECIDES. Predictions P68a-e were written BEFORE this line existed:
# lens/runs/TALK-FAIR-PREDICTIONS-P68.txt
#
# It writes its OWN outputs (TALK-FAIR-169M.txt/.json) so the stored 18:04 TALK-FAIR-TEST.txt and its
# .json are NOT overwritten. The instrument is lens/scripts/talk_fair_test.py with CKPT/OUT/logged made
# env-overridable this turn (defaults unchanged -> re-running with no env vars reproduces the stored file).
# CPU only, threads capped: the GPU arm ct_d256_full169_fix3 (mask-fix full corpus, pid 1184011) is training
# beside us. nice 19 / ionice -c3.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
export TALK_FAIR_CKPT="$PWD/lens/runs/chat/ct_d256_full169_3_best.npz"
export TALK_FAIR_OUT="$PWD/lens/runs/TALK-FAIR-169M.txt"
export TALK_FAIR_LOGGED=0.3834
export TALK_FAIR_EP="ep3, lens/runs/TIED-FULLCORPUS-169M.txt line 65"
export TALK_FAIR_PREDICTIONS="lens/runs/TALK-FAIR-PREDICTIONS-P68.txt"
{
  echo "=== TALK FAIR TEST (P68a-e) on ct_d256_full169_3_best.npz $(date '+%F %H:%M') ==="
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/talk_fair_test.py
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M') ==="
} >> lens/runs/TALK-FAIR-169M.txt 2>&1
