#!/bin/bash
# run_agree_arm.sh -- PLAN item (d2d) follow-up: the AGREEMENT gate on the REAL recipe and the REAL corpus, as
# ONE queue entry. Predictions P61a-e are in lens/runs/AGREEMENT-GATE-PREDICTIONS-APPENDIX.txt, written BEFORE
# this line was queued.
#
# WHY IT IS QUEUED AND NOT RUN INLINE: it is a 12-epoch GPU arm (~40 minutes ungated, and the agreement pass
# costs a measured 4.53x the per-step time on the CPU measurement arms - lens/runs/AGREEMENT-REPORT.txt P60l -
# so this arm is expected to be the longest in the queue). The watchdog starts it only when no other train_chat
# is running, and the checkpoint per epoch means a shutdown cannot lose the run. It is LAST in QUEUE.txt on
# purpose: the 16-arm comparison, its collector and the KV-DECODE regression are what the stored numbers
# depend on, and this arm must not sit inside them.
#
# THE ARM IS THE STORED sgd04 SEED-2 ARM WITH EXACTLY ONE CHANGE: --gates agreement (the DEFAULT detector,
# vs_probe, tol 0.0, 8 groups). Same corpus (data/chat/prepared_mix_ctrl_honest, val measured disjoint from
# train at the training window: 0 of 87,035 windows overlap, lens/runs/VAL-TRAIN-LEAK-mix.txt), same d=256,
# same 1.5M windows, same seed 2, same mask, same step rule (--opt sgd --lr 0.4 --lr-decay 0.5). Its ungated
# twin is lens/runs/chat/opt_sgd04_s2_*, final honest masked val 0.284636858931064 (n=44,418).
#
# WHY IT MATTERS: on the small corpora the default detector refused 0.383/0.190/0.158 of the steps where
# memorisation was possible against 0.013/0.110/0.076 where learning was possible, and the SEPARATION moved
# 4.6x across seeds (lens/runs/AGREEMENT-REPORT.txt). Whether it fires at all, and what it costs, at the real
# scale and on the real corpus is not measured - that is what this arm answers.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export PYTHONUNBUFFERED=1
OUTF=lens/runs/AGREE-ARM.txt
TAG="agree_sgd04_s2"
{
  echo "=== AGREEMENT ARM (real recipe) start $(date '+%F %H:%M:%S')"
  echo "    the stored opt_sgd04_s2 arm plus --gates agreement (AgreementGate vs_probe, tol 0.0, 8 groups),"
  echo "    probe 512 windows drawn once from the held-out split with seed 2, 64 of them scored for A_probe,"
  echo "    per-step log lens/runs/chat/${TAG}_agreement.jsonl, summary lens/runs/chat/${TAG}_agreement_summary.json"
  echo "    EXPECTED COST: up to ~4.5x the per-step time (P60l), so up to ~3 hours; the checkpoint per epoch"
  echo "    is what makes that safe. If it runs past 2 hours the epoch checkpoints are the record."
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 12 --batch 256 --lam-concept 0.3 \
    --seed 2 --device gpu --assistant-only --train-turn-end --max-windows 1500000 \
    --opt sgd --lr 0.4 --lr-decay 0.5 \
    --gates agreement --probe-windows 512 --agree-probe-windows 64 --agree-groups 8 \
    --agree-log-every 200 --out "$TAG"
  echo "ARM EXIT=$?"
  echo "=== AGREEMENT ARM done $(date '+%F %H:%M:%S')"
  echo "    per-epoch rows: lens/runs/chat/${TAG}_history.jsonl   checkpoints: lens/runs/chat/${TAG}_ep*.npz"
} >> "$OUTF" 2>&1
