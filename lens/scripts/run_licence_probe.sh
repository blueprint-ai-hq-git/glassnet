#!/bin/bash
# LICENCE GATE for candidate knowledge/maths datasets. Checks the licence ON HUGGING FACE (the API's own
# cardData), not a blog or a memory, and prints a verdict against our allowed list:
# MIT, Apache-2.0, CC0-1.0, CC-BY-4.0. Anything else is SKIPPED, per the project's rule.
# Read-only: no downloads, no writes outside the project.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
OUT=lens/runs/LICENCE-PROBE.txt
: > "$OUT"
check () {
  local id="$1"
  local lic
  lic=$(curl -s --max-time 25 "https://huggingface.co/api/datasets/$id" \
        | python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
except Exception:
    print('UNREADABLE'); raise SystemExit
c=d.get('cardData') or {}
v=c.get('license') or c.get('license_name') or 'NONE STATED'
# some cards return a LIST (e.g. ['mit']); normalise to a single lowercase token, keep the raw text
raw = v
if isinstance(v, list):
    v = v[0] if v else 'NONE STATED'
v = str(v).strip().lower().replace('_','-').replace(' ','-')
print(v + '|' + str(raw).replace(' ',''))" 2>/dev/null)
  local norm="${lic%%|*}"
  local raw="${lic##*|}"
  case "$norm" in
    mit|apache-2.0|cc0-1.0|cc-by-4.0) verdict="ALLOWED" ;;
    *) verdict="SKIPPED (licence not on our allowed list)" ;;
  esac
  printf '%-46s %-18s %s\n' "$id" "$raw" "$verdict" | tee -a "$OUT"
}
echo "dataset                                        licence          verdict" | tee -a "$OUT"
echo "--------------------------------------------------------------------------" | tee -a "$OUT"
for id in \
  openai/gsm8k \
  meta-math/MetaMathQA \
  openai/openai_humaneval \
  google-research-datasets/mbpp \
  manu/project_gutenberg \
  sedthh/gutenberg_english \
  allenai/peS2o \
  cais/mmlu \
  EleutherAI/hendrycks_math \
  open-r1/OpenR1-Math-220k \
  HuggingFaceH4/ultrachat_200k \
  OpenAssistant/oasst1 \
  nampdn-ai/tiny-textbooks \
  HuggingFaceTB/finemath \
  HuggingFaceTB/smollm-corpus \
  wikimedia/wikipedia \
  arxiv-community/arxiv_dataset ; do
  check "$id"
done
echo | tee -a "$OUT"
echo "NOTE: a dataset's HF licence tag is the distributor's statement. Where the underlying text has its" | tee -a "$OUT"
echo "own licence (for example a Wikipedia dump), that is recorded separately and treated as unclear" | tee -a "$OUT"
echo "unless the distributor states an allowed licence. See DATA_LICENSES.md for the full record." | tee -a "$OUT"