#!/bin/bash
# CPU-ONLY corpus composition audit (predictions P105a-d inside lens/scripts/corpus_mix_audit.py, written before
# its first run). It NEVER touches the GPU and never calls the trainer's corpus loader, so it cannot disturb the
# arm that is training (run_h5l_d256_6ep.sh, scope glassnet-q-2767938400-154238): cores 2,3 (the trainer is
# pinned to 0,1), nice 19, ionice idle. Output: lens/runs/CORPUS-MIX-AUDIT.txt (new file, nothing overwritten).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
{
  echo "======================================================================================================"
  echo "CORPUS MIX AUDIT (CPU only, cores 2-3, nice 19)  $(date '+%F %H:%M:%S')"
  echo "predictions P105a-d are inside the script's own docstring, written BEFORE this runner existed"
  echo "======================================================================================================"
  taskset -c 2,3 nice -n 19 ionice -c3 python3 -u lens/scripts/corpus_mix_audit.py
  echo "AUDIT EXIT=$?  $(date '+%F %H:%M:%S')"
} >> lens/runs/CORPUS-MIX-AUDIT.txt 2>&1
tail -n 5 lens/runs/CORPUS-MIX-AUDIT.txt
