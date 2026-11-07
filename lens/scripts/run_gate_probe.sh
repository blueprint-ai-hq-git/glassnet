#!/bin/bash
# Validate the new --no-gates control flag before it is used for a real arm.
# Runs two 1-epoch, 20k-window probes (~40 s each): gates ON (the normal method) and gates OFF.
# What must be true afterwards:
#   on  : log says "METHOD: OURS (relgrad step rule + gap gate + agreement gate", and agreement/gate step lines exist
#   off : log says "OURS STEP RULE, GATES OFF", and NO agreement or gate step lines exist
# Both must print the same step rule (relgrad) and the same param count, so the only difference is the gates.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/GATE-PROBE.txt"
exec 9>"lens/runs/.lock_gate_probe"
flock -n 9 || { echo "$(date '+%F %T') REFUSED: probe already running" >> "$LOG"; exit 0; }

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }
: > "$LOG"
say "===== GATE PROBE start (validates --no-gates before any real arm uses it)"

for mode in on off; do
  extra=""
  [ "$mode" = "off" ] && extra="--no-gates"
  tag="gateprobe_${mode}"
  say "----- probe $mode : --method ours $extra --max-windows 20000 --epochs 1  $(date '+%H:%M:%S')"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 1 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
    --train-turn-end --max-windows 20000 --gate-tol 0.05 \
    --out "$tag" --method ours --opt-lr 1.0 $extra > "lens/runs/GATE-PROBE-${mode}.txt" 2>&1
  rc=$?
  say "  probe $mode exit=$rc"
  so=$(grep -c "agreement step" "lens/runs/GATE-PROBE-${mode}.txt" 2>/dev/null || true); so=${so:-0}
  go=$(grep -c "    gate step" "lens/runs/GATE-PROBE-${mode}.txt" 2>/dev/null || true); go=${go:-0}
  ml=$(grep -m1 "METHOD:" "lens/runs/GATE-PROBE-${mode}.txt" | cut -c1-120)
  pp=$(grep -m1 "model: " "lens/runs/GATE-PROBE-${mode}.txt" | cut -c1-80)
  say "  agreement-step lines=$so  gate-step lines=$go"
  say "  $ml"
  say "  $pp"
done

on_s=$(grep -c "agreement step" lens/runs/GATE-PROBE-on.txt 2>/dev/null || true); on_s=${on_s:-0}
off_s=$(grep -c "agreement step" lens/runs/GATE-PROBE-off.txt 2>/dev/null || true); off_s=${off_s:-0}
if [ "$on_s" -gt 0 ] && [ "$off_s" -eq 0 ]; then
  say "===== GATE PROBE PASS: gates fire with --method ours ($on_s agreement lines) and do not exist with --no-gates (0)"
else
  say "===== GATE PROBE FAIL: on=$on_s off=$off_s - the flag does not isolate the gates, do NOT use it"
fi
say "===== GATE PROBE DONE $(date '+%F %H:%M')"
