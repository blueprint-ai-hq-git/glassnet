#!/bin/bash
# aux_gradcheck.sh — run the auxiliary-loss checks in BOTH switch positions:
# OFF (T1 inertness) and ON (T2 gradient + T3 column isolation).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"

echo "############ SWITCH OFF (default) ############"
T2_LAM_CONCEPT=0 "$PY" scripts/t3_aux_gradcheck.py
echo "OFF exit=$?"

echo
echo "############ SWITCH ON (lam_concept=1.0) ############"
T2_LAM_CONCEPT=1.0 "$PY" scripts/t3_aux_gradcheck.py
echo "ON exit=$?"