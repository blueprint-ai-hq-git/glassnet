#!/bin/bash
# STOP EVERYTHING for the night (Yahya: "so I can be chill the whole night because it uses too much
# power"). Kills the GPU trainer and all three queued chains, then reports the idle power draw.
# All completed work stays on disk; B1 resumes from its newest checkpoint tomorrow via run_tomorrow.sh.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG=lens/runs/STOP-2026-10-16.md

echo "=== BEFORE:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
pgrep -af "[t]rain_chat.py" | head -2

echo "=== stopping queued chains (so they cannot start the next arm):"
for pat in "[r]un_full_schedule" "[r]un_corpus_ab" "[r]un_small_big_queue"; do
  pkill -f "$pat" && echo "  killed $pat" || echo "  $pat not running"
done
sleep 2
echo "=== stopping the GPU trainer:"
pkill -f "[t]rain_chat.py" && echo "  killed trainer" || echo "  no trainer"
sleep 6

echo "=== AFTER (should be ~0% and a few watts):"
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
pgrep -af "[t]rain_chat|[r]un_small_big_queue|[r]un_corpus_ab|[r]un_full_schedule" || echo "  no project jobs running"

B1=$(ls -t lens/runs/chat/sb_B1_d512_attn_ep*.npz 2>/dev/null | head -1)
{
  echo "# STOPPED FOR THE NIGHT — 2026-10-16 (power saving, Yahya's request)"
  echo
  echo "Stopped: the d512+attention arm (B1) at its newest checkpoint, plus the corpus-isolation and"
  echo "full-schedule chains before they started any arm. Nothing was deleted."
  echo
  echo "Newest B1 checkpoint on disk: ${B1:-none}"
  echo "Completed and recorded earlier today: S0 (13.81% masked, salad), S1 (13.03%, ppl 1892, salad),"
  echo "B0 (13.87%, salad); leakage audits 0/40 and 0/41 leaky; talk-quality meter validated on a control ladder."
  echo
  echo "TO RESUME TOMORROW: bash lens/scripts/run_tomorrow.sh"
} > "$LOG"
echo "=== wrote $LOG"
cat "$LOG"