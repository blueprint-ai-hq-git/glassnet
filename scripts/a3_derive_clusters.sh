#!/bin/bash
# a3_derive_clusters.sh — derive the A3 k=12 and k=16 target sets and report how many
# targets qualify at TRAIN support >= 2.5% (the A3 pre-registration's threshold).
# Same frozen method as CONCEPT-TARGETS-02.md; only k changes.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
LOG=runs/t3/a3_clusters.log

echo "=== A3 TARGET DERIVATION  $(date '+%F %T') ===" | tee "$LOG"
T3C_K=12 T3C_OUT=concept_clusters12.json "$PY" -u scripts/t3_concept_clusters.py \
  2>&1 | tail -6 | tee -a "$LOG"
T3C_K=16 T3C_OUT=concept_clusters16.json "$PY" -u scripts/t3_concept_clusters.py \
  2>&1 | tail -6 | tee -a "$LOG"

echo "=== qualification at TRAIN support >= 2.5% ===" | tee -a "$LOG"
"$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import json, os
R = "/run/media/yahya/HARD DISK 4/GlassNet/runs/t3"
THR = 0.025
summary = {}
for k in (12, 16):
    j = json.load(open(os.path.join(R, f"concept_clusters{k}.json")))
    cl = j["clusters"]
    qual = [c for c in cl if c["support"] >= THR]
    summary[k] = {"n_total": len(cl), "n_qual": len(qual),
                  "supports": [c["support"] for c in cl]}
    print(f"\n--- k={k}: {len(cl)} clusters, {len(qual)} qualify at >= {THR:.1%}")
    for i, c in enumerate(cl):
        mark = "QUALIFIES" if c["support"] >= THR else "below"
        lab = "_".join(c["words"][:4])
        print(f"    {i:>3}  support {c['support']:.4f}  members {len(c['words']):>2}  "
              f"{mark:<10} {lab[:30]}")
    qs = sorted(c["support"] for c in qual)
    if qs:
        print(f"    qualifying supports: min {qs[0]:.4f}  median {qs[len(qs)//2]:.4f}  "
              f"max {qs[-1]:.4f}")

# pre-registered primary-arm rule: larger qualifying set; tie-break -> larger k
k12, k16 = summary[12]["n_qual"], summary[16]["n_qual"]
primary = 12 if k12 > k16 else 16
print(f"\nqualifying counts: k=12 -> {k12}, k=16 -> {k16}")
print(f"PRIMARY ARM (pre-registered: larger set, tie-break larger k) = k={primary}")
print(f"PREDICTION (TARGET-A3.md §5.1): k=12 8-12 qualify, k=16 8-14 qualify")
p12 = 8 <= k12 <= 12
p16 = 8 <= k16 <= 14
print(f"  k=12 prediction {'CONFIRMED' if p12 else 'NOT CONFIRMED'}")
print(f"  k=16 prediction {'CONFIRMED' if p16 else 'NOT CONFIRMED'}")
json.dump({"threshold": THR, "n_qual": {str(k): summary[k]["n_qual"] for k in (12, 16)},
           "primary_k": primary}, open(os.path.join(R, "a3_qual.json"), "w"), indent=1)
EOF
echo "=== done  $(date '+%T') ===" | tee -a "$LOG"