#!/bin/bash
# b_finish.sh — the Target B finisher, designed to run as a NO-AGENT cron script.
#
# Why no-agent: with no_agent=True the scheduler runs ONLY this script. There is no LLM in
# the loop, so the job cannot invent commands. What it can do is exactly what is written
# below, and nothing else -- which is checkable by reading this file.
#
# Behaviour:
#   grid still running -> prints NOTHING (with no_agent, empty stdout sends no message)
#   grid complete      -> runs the audit ONCE, then prints a plain-language verdict
#   already audited    -> prints the recorded verdict, runs nothing
#   anything unexpected-> prints the problem and exits non-zero
#
# It only ever reads/writes inside the GlassNet project folder. It never deletes.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || { echo "GlassNet folder not found"; exit 1; }
PY=".venv-gpu/bin/python3"
[ -x "$PY" ] || PY=python3

STATE=$(bash scripts/b_state.sh 2>/dev/null)
case "$STATE" in
  pending)
    exit 0                      # silent: the grid is still running
    ;;
  DONE)
    # one audit run; its full output goes to a log, not to the message
    bash scripts/b_audit.sh > runs/t3/b_finish.log 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "GlassNet Target B: the audit step FAILED (exit $RC)."
      echo "Log: runs/t3/b_finish.log (last lines follow)"
      tail -8 runs/t3/b_finish.log
      exit 1
    fi
    ;;
  AUDITED)
    :                           # nothing to run; fall through to the verdict
    ;;
  *)
    echo "GlassNet Target B: unexpected grid state '$STATE' -- nothing run."
    exit 1
    ;;
esac

# ---- print the verdict from the RECORDED artifact (never recomputed here) -------------
"$PY" - <<'EOF'
import json, os
R = "runs/t3"
try:
    e = json.load(open(os.path.join(R, "b_eval.json")))
except Exception as ex:
    print(f"GlassNet Target B: no verdict artifact ({ex}).")
    raise SystemExit(0)
r, n = e["readable"], e["normal"]
print("GLASSNET — TARGET B VERDICT")
print(f"readable selected: lc={e['readable_selected']['lc']} lr={e['readable_selected']['lr']} "
      f"({e['readable_selected']['n_seeds']} seeds, {e['readable_selected']['params']:,} params)")
print(f"normal   selected: lr={e['normal_selected']['lr']} "
      f"({e['normal_selected']['n_seeds']} seeds, {e['normal_selected']['params']:,} params)")
print(f"readable: acc {r['acc']:.2f} ± {r['acc_std']:.2f}   ppl {r['ppl']:.4f} ± {r['ppl_std']:.4f}")
print(f"normal  : acc {n['acc']:.2f} ± {n['acc_std']:.2f}   ppl {n['ppl']:.4f} ± {n['ppl_std']:.4f}")
print(f"B-1 perplexity: {e['B1_ppl']}  (readable better by {e['dppl']:+.4f}, "
      f"bar {2*e['pool_p']:.4f})")
print(f"B-2 accuracy  : {e['B2_acc']}  (readable better by {e['dacc']:+.4f}, "
      f"bar {2*e['pool_a']:.4f})")
b3 = os.path.exists(os.path.join(R, "b_audit.json"))
print(f"B-3 audit clauses: {'checked' if b3 else 'not recorded'}")
print("TARGET B = " + ("MET" if e.get("B_met") else "NOT MET"))
print("Parity is never a win. Target A and A2 stay NOT MET; A3 stays MET.")
print("If NOT MET: up to 5 improvement ideas are pre-specified in IDEAS-AFTER-B.md,")
print("to be run ONE change at a time with the prediction written before each run.")
EOF