#!/bin/bash
# run_sentstate_wire_preflight.sh -- P114a/P114e: does the sentstate GRAFT train through the REAL trainer, and is
# it INERT when the flag is off? Tiny corpus, CPU, three short runs. No GPU (agree_sgd04_s2 holds it), no arm.
#
# (NOT the same file as lens/scripts/run_sentstate_preflight.sh, which is the 17:2x turn's pre-flight of the
# sentstate MODULE itself, P111/P112. This one tests the WIRING into the trainer, P114.)
#
# Predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt (written before this file existed).
# Output: lens/runs/SENTSTATE-WIRE.txt (append-only).
#
# WHAT A PASS HERE MEANS, AND WHAT IT DOES NOT: the wiring is correct and the flag-absent path is unchanged.
# It says NOTHING about whether the lever helps a model talk -- the tiny corpus cannot do that, and no quality
# claim of any kind may be read out of these runs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export CHAT_DATA_DIR="$PWD/data/chat/prepared_gate_small"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PY=.venv-gpu/bin/python3
OUT=lens/runs/SENTSTATE-WIRE.txt
COMMON="--d 64 --window 32 --epochs 1 --batch 256 --lam-concept 0.3 --seed 0 --device cpu --max-windows 20000"

{
  echo ""
  echo "####################################################################################################"
  echo "# SENTSTATE WIRE PRE-FLIGHT   $(date '+%F %H:%M:%S')"
  echo "# predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt   corpus: prepared_gate_small (tiny)"
  echo "# three runs, CPU only, d=64 window=32 20,000 windows 1 epoch:"
  echo "#   A  the UNMODIFIED trainer           out _ss_pre_off   (the pre-change reference)"
  echo "#   B  the launcher, SENTSTATE unset    out _ss_wire_off  (P114a: must be field-identical to A)"
  echo "#   C  the launcher, SENTSTATE=1        out _ss_wire_on   (P114e: must run, finite, no NaN)"
  echo "####################################################################################################"

  echo ""
  echo "=== RUN A: unmodified trainer (pre-change reference) $(date '+%H:%M:%S')"
  nice -n 19 ionice -c3 $PY -u lens/scripts/train_chat.py $COMMON --out _ss_pre_off 2>&1
  echo "A EXIT=$?"

  echo ""
  echo "=== RUN B: launcher, SENTSTATE unset (P114a control) $(date '+%H:%M:%S')"
  SENTSTATE=0 nice -n 19 ionice -c3 $PY -u lens/scripts/train_chat_sent.py $COMMON --out _ss_wire_off 2>&1
  echo "B EXIT=$?"

  echo ""
  echo "=== RUN C: launcher, SENTSTATE=1 (the lever ON) $(date '+%H:%M:%S')"
  SENTSTATE=1 SENTSTATE_DS=64 nice -n 19 ionice -c3 $PY -u lens/scripts/train_chat_sent.py $COMMON --out _ss_wire_on 2>&1
  echo "C EXIT=$?"

  echo ""
  echo "=== COMPARISON $(date '+%H:%M:%S')"
  nice -n 19 python3 - "$PWD" <<'PYEOF'
import json, os, sys
CH = os.path.join("lens", "runs", "chat")
def row(tag):
    p = os.path.join(CH, tag + "_history.jsonl")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None
a, b, c = row("_ss_pre_off"), row("_ss_wire_off"), row("_ss_wire_on")
if a is None or b is None or c is None:
    print("  CONTROL FAIL: a history row is missing ->", {k: (v is not None) for k, v in (("A", a), ("B", b), ("C", c))})
    sys.exit(1)
keys = [k for k in a if k in b and k != "seconds" and not k.endswith("_note")]
bad = [(k, a[k], b[k]) for k in keys if a[k] != b[k]]
print("  P114a CONTROL: A (unmodified) vs B (graft installed, flag off), excluding wall time")
for k, x, y in bad:
    print("      %s: A %r  vs B %r" % (k, x, y))
print("  P114a VERDICT: %s  (%d/%d fields identical)" % ("PASS" if not bad else "FAIL", len(keys) - len(bad), len(keys)))
for nm, r in (("A pre-change", a), ("B flag off", b), ("C flag ON ", c)):
    print("      %s train_acc %.6f  val_asst_acc %r  train_loss %.6f  secs %.1f"
          % (nm, r.get("train_acc"), r.get("val_asst_acc"), r.get("train_loss", float("nan")), r.get("seconds", 0.0)))
finite = all(isinstance(c.get(k), (int, float)) and c.get(k) == c.get(k) and abs(c.get(k)) != float("inf")
             for k in ("train_loss", "val_ppl", "val_asst_ppl") if c.get(k) is not None)
print("  P114e FINITE ON C: %s  (train_loss %r)" % (finite, c.get("train_loss")))
print("  P114e VERDICT: %s" % ("PASS" if finite else "FAIL"))
print("  NOTE: a wiring result only. The tiny corpus cannot talk and NO quality claim comes out of it.")
PYEOF
  echo "=== PRE-FLIGHT DONE $(date '+%F %H:%M:%S')"
  echo "# the three runs' artifacts are moved OUT of lens/runs/chat below: the watchdog's staleness instrument is"
  echo "# 'newest *.npz in lens/runs/chat', so a pre-flight left there would mask a stalled arm."
} >> "$OUT" 2>&1

# MOVE THE PRE-FLIGHT'S OWN CHECKPOINTS OUT OF THE TRAINER'S OUTPUT DIRECTORY (measured hazard, 18:1x): the
# watchdog measures progress as the newest lens/runs/chat/*.npz, and these three files are newer than any real
# arm's, so leaving them would have hidden a stalled trainer for as long as they were the newest.
mkdir -p lens/runs/_tmp/ss_preflight
mv lens/runs/chat/_ss_pre_off* lens/runs/chat/_ss_wire_off* lens/runs/chat/_ss_wire_on* \
   lens/runs/_tmp/ss_preflight/ 2>/dev/null

sed -n "/=== COMPARISON/,\$p" "$OUT" | tail -20
