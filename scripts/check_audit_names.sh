#!/bin/bash
# check_audit_names.sh — dry check that cluster runs cannot collide with v1 names.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
export T3A_BACKEND=numpy T3A_K=32 T3A_NAMER=pa T3A_FIRING_FLOOR=0.02
export T3A_LOAD_DIR=runs/t3/ckpt

echo "=== v1 (single words) ==="
T2_CONCEPT_SET=words T2_LAM_CONCEPT=0.1 T3A_LR=0.5 "$PY" -c "
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import t3_audit as A
print('  ckpt :', os.path.basename(A.LOAD_DIR + '/readable_K32_lr0.5_lc01' + ('' if A.CONCEPT_SET=='words' else '_'+A.CONCEPT_SET) + '_seed0.npz'))
print('  audit:', os.path.basename(A.OUT))
"
echo "=== v2 (clusters) ==="
T2_CONCEPT_SET=clusters T2_LAM_CONCEPT=0.1 T3A_LR=0.5 "$PY" -c "
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import t3_audit as A
print('  ckpt :', os.path.basename(A.LOAD_DIR + '/readable_K32_lr0.5_lc01' + ('' if A.CONCEPT_SET=='words' else '_'+A.CONCEPT_SET) + '_seed0.npz'))
print('  audit:', os.path.basename(A.OUT))
"
echo
echo "=== do the v2 names collide with any existing file? ==="
ls runs/t3/audit_K32_lr05_pa_lc01_clusters_ff002.json 2>&1 | head -1