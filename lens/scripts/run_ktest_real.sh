#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
# rename the smoke artifact out of the way (mv, never overwrite)
for f in lens/runs/image/ktest_smoke*; do
  [ -e "$f" ] && mv "$f" "${f/ktest_smoke/ktest_SMOKE}"
done
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# SEE-ONLY K-TRANSFER — does the seetalk concept finding hold for the pure classifier?
# PRE-REGISTERED GATES (fixed before running, binary):
#   k1  extra=real  mean test acc > extra=none by > 2.0 pts   (+10 real concepts transfer)
#   k2  extra=dummy mean test acc within +-2.0 of extra=real   (again width, not meaning)
#   k3  k1 fails AND k2 fails -> K-gain was seetalk-specific / caption-position-specific
# 3 seeds x 3 arms x (60k images, 12 epochs, d=64, GPU) — the PROVEN train_image loop untouched.
{
for arm in none real dummy; do
  for seed in 0 1 2; do
    nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_image.py \
      --epochs 12 --device gpu --extra "$arm" --no-plain --seed "$seed" \
      --out "ktest_${arm}_s${seed}" &
    P=$!
    bash lens/scripts/game_guard.sh "$P" 2>/dev/null &
    G=$!
    wait $P
    kill $G 2>/dev/null
  done
done
} 2>&1 | grep -E "EXTRA|readable .* params   TEST|concepts passing|Traceback|Error" | tail -30
