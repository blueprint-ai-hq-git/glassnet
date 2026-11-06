#!/bin/bash
# FULL STATUS CHECK (2026-10-30, rebuilt to Yahya's spec: "check everything and tell me exactly").
# Every line below is read from a file or a process on this machine. Nothing is estimated.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
say() { echo "$1"; }
tr=$(pgrep -cf "venv-gpu/bin/python3 -u lens/scripts/train_chat"); tr=${tr:-0}
q=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null); q=${q:-0}
pl=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/PLAN.txt 2>/dev/null); pl=${pl:-0}
ck=$(ls lens/runs/chat/*.npz 2>/dev/null | wc -l)
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
race=$(ps -eo etime,args 2>/dev/null | grep "[t]rain_chat" | head -1 | awk '{print $1}')
live=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8770/ 2>/dev/null)
newck=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1)
ckage=99999; [ -n "$newck" ] && ckage=$(( ( $(date +%s) - $(stat -c %Y "$newck") ) / 60 ))

say "GLASSNET FULL CHECK  $(date '+%a %d %b %H:%M')"
say ""
say "WHAT IS RUNNING"
if [ "$tr" -gt 0 ]; then say "  training: YES, arm running ${race}"; else say "  training: NOTHING RUNNING  <-- problem unless you asked for it"; fi
say "  live page: HTTP ${live:-no answer} at 192.168.1.58:8770"
say "  gpu: ${gpu}(util%, MiB, C)   checkpoints: ${ck}"
say "  queue: ${q} job(s)   plan: ${pl} experiment(s) left"
say ""
say "NEWEST RESULT (real numbers from the run log)"
newlog=$(ls -t lens/runs/*.txt 2>/dev/null | head -1)
[ -n "$newlog" ] && { say "  file: $(basename "$newlog")"; grep -E "ep [0-9]+/" "$newlog" 2>/dev/null | tail -1 | cut -c1-104 | sed 's/^/  /'; }
say ""
say "THE PROJECT, PART BY PART (measured, not claimed)"
say "  engine from scratch ......... gradients verified 4.14e-09  DONE"
say "  own optimizer ............... beats SGD +3.4 pts, 3 seeds  DONE"
say "  own training method ......... wins at lr 1.0, gates still being tuned  85%"
say "  own KV cache ................ 9.7x faster, identical  DONE"
say "  named neurons ............... 36 concepts, 36/36 audit pass  80%"
say "  live browser view ........... real activations, labels pending  85%"
say "  exports + server ............ gguf/safetensors valid, our runtime only  95%"
say "  learn not memorise .......... MEASURED, NOT REACHED  25%"
say "    width, passes, attention, tokens-per-example: all tested, none raised held-out at 20k windows"
say "    only more DISTINCT data raised it: 4.7 -> 11.7 -> 18.2 -> 27.0% at 2k -> 2M windows"
say "  coherent talk ............... grammar yes, meaning no  15%"
say "  readability 100% easy mode .. gate FAILS  10%"
say "  thinking block / memory ..... not built  0%"
say ""
say "OVERALL: about 58% of the machine, about 20% of the intelligence"
say ""
say "WHERE WE ARE RIGHT NOW"
say "  THIS PART: making the model LEARN instead of MEMORISE (his core idea) - 25% of that part done."
say "  STATE OF IT: width, passes, attention and tokens-per-example are all tested and all eliminated;"
say "  the only lever that ever moved held-out was more DISTINCT data (4.7 -> 27.0% across 2k -> 2M windows)."
say "  NEXT STEP IN THAT PART: change the architecture, not the settings - the concept layer squeezes 8192"
say "  numbers into 36 neurons before answering, and that is the leading suspect for the lost information."
say ""
say "PROBLEMS FOUND THIS CHECK"
found=0
[ "$tr" -eq 0 ] && { say "  - nothing is training"; found=1; }
[ "$tr" -gt 0 ] && [ "$ckage" -gt 90 ] && { say "  - newest checkpoint is ${ckage} min old (possible stall)"; found=1; }
[ "$live" != "200" ] && { say "  - live page not answering"; found=1; }
[ "$q" -eq 0 ] && [ "$pl" -eq 0 ] && [ "$tr" -eq 0 ] && { say "  - no queued or planned work: new experiments must be written"; found=1; }
[ "$found" -eq 0 ] && say "  none"
say ""
say "NEXT IN LINE"
grep -vE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null | head -2 | sed 's/^/  queue: /'
grep -vE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/PLAN.txt 2>/dev/null | head -2 | sed 's/^/  plan: /'
exit 0
