#!/bin/bash
# b_state.sh — monitor for the Target B cron job. Prints a DETERMINISTIC single token so
# the scheduler only wakes the agent when the state actually changes:
#   pending   = grid still running
#   DONE      = grid complete, audit not yet run
#   AUDITED   = audit complete (nothing further to do)
# No timestamps, no counts that tick up mid-run -- otherwise every tick would look changed.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || { echo "PROJECT-MISSING"; exit 0; }
PY=".venv-gpu/bin/python3"
[ -x "$PY" ] || PY=python3

if [ -f runs/t3/b_eval.json ] && [ -f runs/t3/b_audit.json ]; then
  echo "AUDITED"
  exit 0
fi

"$PY" - <<'EOF' 2>/dev/null || echo "pending"
import glob, json, os
R = "runs/t3"
def n(pat, model):
    c = 0
    for f in glob.glob(os.path.join(R, pat)):
        try:
            c += len([x for x in json.load(open(f)) if x["model"] == model])
        except Exception:
            pass
    return c
r, nn = n("models_b_read_*.json", "readable"), n("models_b_norm_*.json", "normal")
print("DONE" if (r >= 60 and nn >= 20) else "pending")
EOF