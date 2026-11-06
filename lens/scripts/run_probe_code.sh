#!/bin/bash
# Probe CODE datasets for the licence gate: public (no login), licence in the safe list
# (MIT / Apache-2.0 / CC0 / CC BY 4.0), and small enough to take whole shards.
# Read-only. Exam sets (HumanEval, MBPP, CodeContests) are excluded by name -- those are our tests.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/CODE-PROBE.txt
: > "$OUT"
for id in ise-uiuc/Magicoder-OSS-Instruct-75K nvidia/OpenCodeInstruct codeparrot/codeparrot-clean \
          m-a-p/CodeFeedback-Filtered-Instruction bigcode/the-stack-smol HuggingFaceTB/cosmopedia; do
  echo "=== $id" | tee -a "$OUT"
  curl -s --max-time 30 "https://huggingface.co/api/datasets/$id" \
    | python3 -c "
import sys, json
try: d = json.load(sys.stdin)
except Exception as e: print('   no metadata:', e); raise SystemExit
cd = d.get('cardData') or {}
lic = cd.get('license') or d.get('license') or 'NOT STATED'
print(f\"   licence: {lic} | gated: {d.get('gated')} | downloads: {d.get('downloads')}\")
" 2>/dev/null | tee -a "$OUT"
  curl -s --max-time 30 "https://huggingface.co/api/datasets/$id/tree/main?recursive=1" \
    | python3 -c "
import sys, json
try: rows = json.load(sys.stdin)
except Exception: raise SystemExit
files = [r for r in rows if r.get('type') == 'file' and r.get('path','').endswith(('.parquet','.jsonl','.json'))]
files.sort(key=lambda r: r['path'])
for r in files[:2]:
    print(f\"   {(r.get('size') or 0)/1e6:9.1f} MB  {r['path']}\")
tot = sum((r.get('size') or 0) for r in files)
print(f'   {len(files)} files, {tot/1e9:.2f} GB total')
" 2>/dev/null | tee -a "$OUT"
done
echo | tee -a "$OUT"
echo "NEVER train on: HumanEval, MBPP, CodeContests, GSM8K, MMLU -- those are our exam papers." | tee -a "$OUT"