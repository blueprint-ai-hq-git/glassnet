#!/bin/bash
# CRITICAL PATH 1, UNCAPPED: EVERY TRAINING WINDOW (16,878,132), d=256, 3 epochs, best-val checkpointing.
#
# WHY THIS IS A SEPARATE FILE (2026-10-22 17:0x, self-wake). The first version of this arm lived in
# lens/scripts/run_tied_fullcorpus.sh, and while it was being queued A PARALLEL TURN REWROTE THAT FILE to run a
# different arm (--epochs 6 --max-windows 6000000 --out ct_d256_6M_6ep, which is what is running now). Two turns
# editing one runner is how experiments get mixed up silently, so this arm gets its OWN file and the other file
# is left exactly as that turn set it. Nothing was overwritten; both intents survive.
#
# ITS HISTORY, measured, so nobody has to guess why it is queued again:
#   16:09:58 launched by the watchdog, OOM-KILLED at 16:13:56 inside its own scope (glassnet-q-2591701866):
#            "Result: oom-kill", Mem peak 10G, swap 2.7G, dead inside the data build with NO epoch line
#            (lens/runs/TIED-FULLCORPUS.txt). P63b ("peak RSS below 9.0 GB") is a FAIL: 10G measured.
#   16:52    re-queued; the launch SILENTLY FAILED ("Unit ... was already loaded"), the line was consumed, and
#            nothing ran (SUPERVISOR.txt, QUEUE-DONE.txt). Fixed in lens/scripts/watchdog.sh: unique unit name
#            per launch, the launch is VERIFIED, and the line is restored at the front if it did not start.
#
# THE MEMORY FIX THIS ARM NOW CARRIES (lens/runs/MEMFIX2-PREDICTIONS.txt, written BEFORE the edit):
#   X, Y int64 -> int32 (every id in this corpus is 0..12001) and is_asst/is_turnend float32 -> bool.
#   VERIFIED: values identical 12/12 cases (lens/runs/MEMFIX2-TEST.txt, P64a); the real corpus ids are already
#   int32 on disk, range 0..12001 (P64e); the mask's printed numbers and every epoch line are digit-identical
#   between the pre-change and changed trainer on real dialogue (lens/runs/MEMFIX2-MASK.txt, P64b); int32 vs
#   int64 window arrays give BIT-IDENTICAL logits, loss and 6/6 gradient tensors on cpu AND gpu
#   (lens/runs/D-TYPE-EQUIV.txt, P64c). Pre-change copies: lens/runs/_tmp/*_prefix_memfix2.py.
#
# PREDICTIONS (the three originals from 2026-10-22 15:40, unchanged, then the ones added before this re-queue):
#  1. if the undertraining explanation is right, held-out beats the 1.5M-window tied run's peak 29.71% and the
#     talk test improves.
#  2. if held-out stays in the 27-30% band, data volume was NOT the binding constraint and the next test is
#     capacity (d=512).
#  3. the train/held-out gap is expected to stay wide; that is the memorisation question, measured here and
#     attacked separately, not hidden by the gate.
#  P63a this arm reaches epoch 1 and writes its first per-epoch checkpoint.
#  P63b peak RSS stays below 9.0 GB.  ->  ALREADY MEASURED AS A FAIL (10G) on the pre-fix code; P64d replaces it.
#  P63c the first epoch's wall time is 2.8-4.0x the capped arm's first epoch (the window count is 2.81x).
#  P63d the masked held-out number is reported from the DISJOINT val split only (87,035 windows, 0 overlap with
#       train: lens/runs/VAL-TRAIN-LEAK-mix.txt) and it names that file, or it is not reported at all.
#  P64d peak RSS 6.5 GB or LOWER, read from this arm's own scope: `systemctl --user show -p MemoryPeak
#       glassnet-q-<unit>.scope`. If it is again near 10G, the width fix was not the binding term and the next
#       step is to stop holding all windows in RAM at once.
#  P64g the FIRST EPOCH LINE is written this time. Three earlier uncapped attempts died inside the data build
#       with no epoch line at all (TIED-ALLDATA.txt, TIED-D512.txt, TIED-FULLCORPUS.txt).
#  P64h THE TALK TEST IS STILL WORD SALAD. Predictions from today's two measured arms: the capped d=256 arm
#       reached masked held-out 34.92% and the d=512 arm 33.94%, and BOTH produced degenerate generation
#       (TIED-ALLDATA.txt, TIED-D512.txt "CAN IT TALK?"). 2.8x more data on the same recipe is not expected to
#       fix generation on its own, and this run is not to be called progress on its loss - only on its English.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
{
  echo "=== TIED + FULL CORPUS (16,878,132 windows, UNCAPPED) d=256 3 epochs $(date '+%F %H:%M')"
  echo "reference: tied/1.5M-windows peaked 29.71% (ep4), ended 27.37%, train acc 44.14%, talk still salad"
  echo "reference: capped 6M d=256 best 34.92% (TIED-ALLDATA.txt); d=512 4M best 33.94% (TIED-D512.txt); both talk=FALSE"
  echo "memory width fix verified: lens/runs/MEMFIX2-TEST.txt, MEMFIX2-MASK.txt, D-TYPE-EQUIV.txt"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end \
    --direct --direct-mode tied --max-windows 16900000 --out ct_d256_full169_3
  echo "exit=$?"
  echo "=== DONE $(date '+%F %H:%M')"
} >> lens/runs/TIED-FULLCORPUS-169M.txt 2>&1
