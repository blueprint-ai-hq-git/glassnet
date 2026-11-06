#!/bin/bash
# PIPELINE WATCH — a heartbeat trail with STALL DETECTION, so "keeping an eye on it" is a fact on
# disk rather than a promise. Appends one line every 5 minutes: time, supervisor state, trainer
# state, GPU use, and the newest checkpoint of the active run. If a trainer is alive but no new
# checkpoint has appeared for >15 minutes, that is recorded as a STALL (the failure mode that would
# otherwise sit unnoticed for hours).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
exec >> /tmp/pipeline_heartbeat.log 2>&1
EVENTS="lens/runs/PIPELINE_EVENTS.md"
LAST_CKPT=""; LAST_CKPT_TIME=0; LAST_EVENT=""
echo "=== WATCH START $(date '+%F %H:%M:%S') pid=$$"
while true; do
  NOW=$(date '+%F %H:%M:%S'); EPOCH=$(date +%s)
  SUP=$(pgrep -f "[r]un_morning_supervisor" | head -1)
  TR=$(pgrep -f "[t]rain_chat.py" | head -1)
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')
  CK=$(ls -t lens/runs/chat/*_ep*.npz 2>/dev/null | head -1)
  CKT=""
  [ -n "$CK" ] && CKT=$(date -r "$CK" '+%H:%M:%S')
  STALL=""
  if [ -n "$TR" ]; then
    if [ "$CK" != "$LAST_CKPT" ]; then
      LAST_CKPT="$CK"; LAST_CKPT_TIME=$EPOCH
    elif [ $((EPOCH - LAST_CKPT_TIME)) -gt 900 ]; then
      STALL="STALL: no new checkpoint in $(( (EPOCH-LAST_CKPT_TIME)/60 )) min while a trainer is alive"
    fi
  fi
  echo "$NOW sup=${SUP:-none} train=${TR:-none} gpu=[$GPU] newest=$(basename "$CK" 2>/dev/null)@$CKT ${STALL}"
  # events worth telling a human about, written once each
  if [ -z "$SUP" ] && [ -z "$TR" ]; then
    if [ "$LAST_EVENT" != "idle" ]; then
      echo "- $NOW  pipeline IDLE: no supervisor and no trainer running" >> "$EVENTS"; LAST_EVENT=idle
    fi
  elif [ -n "$STALL" ] && [ "$LAST_EVENT" != "stall" ]; then
    echo "- $NOW  $STALL (newest $CK)" >> "$EVENTS"; LAST_EVENT=stall
  elif [ -n "$TR" ]; then
    LAST_EVENT="running"
  fi
  grep -q "MATRIX DONE" lens/runs/MATRIX.txt 2>/dev/null && {
    echo "- $NOW  matrix finished; results in lens/runs/MATRIX.txt" >> "$EVENTS"
    grep -q "matrix finished" "$EVENTS" || true
  }
  sleep 300
done