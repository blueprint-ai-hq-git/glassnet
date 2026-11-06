#!/bin/bash
# b_audit.sh — TARGET B step 2: the B-3 audit clauses on the WINNING readable run.
# Runs only after the grid is complete. The winning setting is read from b_eval.json
# (produced by t3_b_report.py applying the frozen rule to the full grid), so the audited
# model cannot be hand-picked.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export T2_CONCEPT_SET=clusters16
export T2_N_TAUGHT=10
export T3_CKPT_TAG=b
export T3_K=16
PY=".venv-gpu/bin/python3"
LOG=runs/t3/b_run.log
fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }

echo "=== TARGET B AUDIT (B-3)  $(date '+%F %T') ===" | tee -a "$LOG"

echo "--- 1. score the grid with the frozen rule (refuses on a partial grid)" | tee -a "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_b_report.py >> "$LOG" 2>&1
RC=$?
echo "    b_report rc=$RC" | tee -a "$LOG"
[ "$RC" -eq 2 ] && fail "grid incomplete -- not scoring"
[ "$RC" -ne 0 ] && fail "b_report failed rc=$RC"

LR=$("$PY" -c "import json;print(json.load(open('runs/t3/b_eval.json'))['readable_selected']['lr'])")
LC=$("$PY" -c "import json;print(json.load(open('runs/t3/b_eval.json'))['readable_selected']['lc'])")
LCS=$(echo "$LC" | tr -d '.')
echo "    winning readable setting: lr=$LR lc=$LC" | tee -a "$LOG"

echo "--- 2. Meter-03 audit on the winning readable run" | tee -a "$LOG"
T3M_B=1 T3M_K=16 T3M_N_SUP=10 nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit_m3.py \
  >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "Meter-03 audit failed"

echo "--- 3. lie probe (train with the _b tag, then audit it)" | tee -a "$LOG"
T3A_BACKEND=cupy T3A_LOAD_DIR="runs/t3/ckpt" T3A_NAMER=pa T3A_MIN_SUPPORT=0.01 \
T3A_K=16 T3A_SEEDS=0,1,2 T3A_EPOCHS=96 T3A_MIN_EPOCH=30 T3A_PATIENCE=15 \
T3A_N_SUP=10 T3A_LIE_COL=2 T3A_FIRING_FLOOR=0.02 T3A_LIVE_TARGET=0.02 \
T3A_FORCE_PROBE=1 T3A_LR="$LR" T2_LAM_CONCEPT="$LC" \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_audit.py >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "probe training failed"

T3P_KIND=clusters16 T3P_K=16 T3P_NSUP=10 \
T3P_CK="lieprobe_K16_lr${LR}_lc${LCS}_clusters16_b_seed1.npz" \
T3P_COL=2 T3P_SEED=1 T3P_OUT=b_lieprobe.json \
  nice -n 19 ionice -c3 "$PY" -u scripts/t3_m3_probe_x.py >> "$LOG" 2>&1
[ $? -ne 0 ] && fail "probe audit failed"

echo "--- 4. collect the B-3 clauses" | tee -a "$LOG"
"$PY" -c "
import json, glob, os
R = 'runs/t3'
probe = json.load(open(os.path.join(R, 'b_lieprobe.json')))
audit = None
for f in glob.glob(os.path.join(R, 'meter3_clusters16*.json')):
    d = json.load(open(f))
    for k, v in d.get('models', {}).items():
        if k.startswith('b '):
            audit = v['agg']
out = {
  'lie_probe_caught_live': bool(probe['caught'] and probe['live']),
  'lie_probe': probe['own'], 'probe_best_is_caught': probe['best_is_caught'],
  'random_control': audit,
  'random_ok': bool(audit and audit['random_sup'] < 5.0 and audit['random_uns'] < 5.0),
}
out['pass'] = out['lie_probe_caught_live'] and out['random_ok']
json.dump(out, open(os.path.join(R, 'b_audit.json'), 'w'), indent=1)
print('  B-3 lie probe caught while live:', out['lie_probe_caught_live'])
print('  B-3 random control:', audit)
print('  B-3 pass:', out['pass'])
" 2>&1 | tee -a "$LOG"

echo "--- 5. final verdict (now with B-3)" | tee -a "$LOG"
nice -n 19 ionice -c3 "$PY" -u scripts/t3_b_report.py 2>&1 | tail -22 | tee -a "$LOG"
echo "=== TARGET B AUDIT DONE  $(date '+%F %T') ===" | tee -a "$LOG"