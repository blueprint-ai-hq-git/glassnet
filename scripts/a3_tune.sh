#!/bin/bash
# a3_tune.sh — TARGET-A3.md §4: re-tune lam_concept and lr for THIS target set using the
# frozen SELECTION-RULE.md (validation perplexity primary, tie-break accuracy within 1%
# relative, then lower lr).
#
# Pre-registered compute-budget decision: all 9 combinations are SCREENED on seed 0; the
# top 2 by the frozen rule are CONFIRMED on seeds 1 and 2; the winner is chosen on the
# 3-seed mean. Running all 9 x 3 seeds is ~3x tonight's budget. Recorded as a limitation.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/t3
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3_BACKEND=cupy
export T3_K=16
export T3_EPOCHS=96
export T3_MIN_EPOCH=30
export T3_PATIENCE=15
export T2_LAM_A=0.2
export T2_LAM_W=1.0
export T2_CONCEPT_SET=clusters16
export T2_N_TAUGHT=10
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a3_run.log

fail() { echo "!!! ABORT: $*" | tee -a "$LOG"; exit 1; }
[ -f runs/t3/concept_clusters16.json ] || fail "no k=16 targets"
"$PY" -c "import cupy" 2>/dev/null || fail "cupy missing"

echo "=== TARGET A3 — TUNING  $(date '+%F %T') ===" | tee -a "$LOG"
for lc in 0.1 0.5 1.0; do
  for lr in 0.1 0.5 1.0; do
    TAG="a3_scr_lc${lc}_lr${lr}"
    echo "--- screen lc=$lc lr=$lr seed0  $(date '+%T')" >> "$LOG"
    T2_LAM_CONCEPT="$lc" T3_LR="$lr" T3_SEEDS=0 T3_MODELS=readable T3_TAG="$TAG" \
      nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
    [ $? -ne 0 ] && fail "screen lc=$lc lr=$lr failed"
  done
done

echo "--- applying the frozen selection rule to the 9 screened combos" | tee -a "$LOG"
"$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import glob, json, os
rows = []
for f in glob.glob("runs/t3/models_a3_scr_*.json"):
    tag = os.path.basename(f)[len("models_"):-len(".json")]
    lc = tag.split("_lc")[1].split("_")[0]
    lr = tag.split("_lr")[1]
    m = json.load(open(f))
    r = [x for x in m if x["model"] == "readable"][0]
    rows.append({"lc": lc, "lr": lr, "ppl": r["val_ppl"], "acc": r["val"],
                 "tag": tag})
# frozen rule: primary validation perplexity (lower better), tie-break accuracy within
# 1% RELATIVE, then lower lr
rows.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4),
                         float(x["lr"])))
print("  screened combos, sorted by the frozen rule:")
for r in rows:
    print(f"    lc={r['lc']} lr={r['lr']}  val ppl {r['ppl']:.4f}  val acc {r['acc']:.2f}%")
top2 = rows[:2]
json.dump({"screened": rows, "top2": top2},
          open("runs/t3/a3_screen.json", "w"), indent=1)
print(f"  TOP2 to confirm on seeds 1,2: "
      f"{[(t['lc'], t['lr']) for t in top2]}")
EOF
[ $? -ne 0 ] && fail "selection rule failed"

echo "--- confirming the top 2 on seeds 1 and 2" | tee -a "$LOG"
for combo in $("$PY" -c "
import json; d=json.load(open('runs/t3/a3_screen.json'))
print(' '.join(f\"{t['lc']}:{t['lr']}\" for t in d['top2']))"); do
  LC="${combo%%:*}"; LR="${combo##*:}"
  TAG="a3_cfm_lc${LC}_lr${LR}"
  echo "--- confirm lc=$LC lr=$LR seeds1,2  $(date '+%T')" >> "$LOG"
  T2_LAM_CONCEPT="$LC" T3_LR="$LR" T3_SEEDS=1,2 T3_MODELS=readable T3_TAG="$TAG" \
    nice -n 19 ionice -c3 "$PY" -u scripts/t3_models.py >> "$LOG" 2>&1
  [ $? -ne 0 ] && fail "confirm lc=$LC lr=$LR failed"
done

echo "--- choosing the winner on the 3-seed mean" | tee -a "$LOG"
"$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob, os
scr = json.load(open("runs/t3/a3_screen.json"))
cands = []
for t in scr["top2"]:
    lc, lr = t["lc"], t["lr"]
    vals = [t]                                   # seed 0 from the screen
    f = f"runs/t3/models_a3_cfm_lc{lc}_lr{lr}.json"
    if os.path.exists(f):
        vals += [x for x in json.load(open(f)) if x["model"] == "readable"]
    ppl = sum(v["val_ppl"] for v in vals) / len(vals)
    acc = sum(v["val"] for v in vals) / len(vals)
    cands.append({"lc": lc, "lr": lr, "ppl": ppl, "acc": acc, "n_seeds": len(vals)})
cands.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4), float(x["lr"])))
for c in cands:
    print(f"    lc={c['lc']} lr={c['lr']}  {c['n_seeds']}-seed mean ppl {c['ppl']:.4f} "
          f"acc {c['acc']:.2f}%")
w = cands[0]
json.dump({"lr": w["lr"], "lc": w["lc"], "ppl": w["ppl"], "acc": w["acc"],
           "n_seeds": w["n_seeds"], "candidates": cands,
           "rule": "frozen SELECTION-RULE.md: val ppl, tie-break acc within 1% rel, then lower lr",
           "budget_note": "9 combos screened on seed 0; top 2 confirmed on seeds 1,2"},
          open("runs/t3/a3_selection.json", "w"), indent=1)
print(f"  WINNER: lr={w['lr']} lam_concept={w['lc']}")
EOF
[ $? -ne 0 ] && fail "winner selection failed"
echo "=== A3 TUNING DONE  $(date '+%F %T') ===" | tee -a "$LOG"