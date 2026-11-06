#!/bin/bash
# DOWNLOAD the license-verified datasets into the project folder. Everything here passed the
# declared-license gate (MIT / Apache-2.0 / CC0 / CC-BY-4.0) in run_license_check.sh.
# Network-bound, so it runs alongside the GPU experiment. Logs a size report at the end.
# Budget: this set is ~4-5 GB of the 20 GB allowance.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
HF=/home/yahya/.hermes/installs/15d969a29be1af88/environments/7c7750f1e3d5471e920a055130314e9b/venv/bin/hf
mkdir -p data/external
log() { echo "=== $(date +%H:%M:%S) $*"; }

# --- benchmarks (small) -----------------------------------------------------------------------
log "gsm8k (MIT)"
$HF download openai/gsm8k --repo-type dataset --local-dir data/external/gsm8k 2>&1 | tail -2
log "mmlu (MIT)"
$HF download cais/mmlu --repo-type dataset --local-dir data/external/mmlu 2>&1 | tail -2
log "humaneval (MIT)"
$HF download openai/openai_humaneval --repo-type dataset --local-dir data/external/humaneval 2>&1 | tail -2
log "mbpp (CC-BY-4.0)"
$HF download google-research-datasets/mbpp --repo-type dataset --local-dir data/external/mbpp 2>&1 | tail -2

# --- training data ----------------------------------------------------------------------------
log "oasst1 (Apache-2.0) -- real human chat"
$HF download OpenAssistant/oasst1 --repo-type dataset --local-dir data/external/oasst1 2>&1 | tail -2
log "ultrachat_200k (MIT)"
$HF download HuggingFaceH4/ultrachat_200k --repo-type dataset --local-dir data/external/ultrachat \
  --include "data/*" 2>&1 | tail -2
log "orca-math-word-problems-200k (MIT)"
$HF download microsoft/orca-math-word-problems-200k --repo-type dataset \
  --local-dir data/external/orca_math --include "data/*" 2>&1 | tail -2

# --- other lines ------------------------------------------------------------------------------
log "mnist (MIT, declared by the author's own mirror)"
$HF download ylecun/mnist --repo-type dataset --local-dir data/external/mnist 2>&1 | tail -2
log "speech_commands (CC-BY-4.0) -- real audio"
$HF download google/speech_commands --repo-type dataset --local-dir data/external/speech_commands \
  --include "data/*" 2>&1 | tail -2

log "SIZE REPORT"
du -sh data/external/* 2>/dev/null
echo "--- total:"; du -sh data/external 2>/dev/null; df -h . | tail -1
log "DONE"