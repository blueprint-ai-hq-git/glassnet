#!/bin/bash
# Probe SUBSTITUTE knowledge sources for the gated tiny-textbooks. Requirements: public (no login),
# licence in the safe list (MIT / Apache-2.0 / CC0 / CC BY 4.0), textbook- or reference-like content.
# Read-only: prints licence tag and file sizes so the choice is made on evidence.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
for id in HuggingFaceTB/cosmopedia nampdn-ai/tiny-codes togethercomputer/RedPajama-Data-1T \
          HuggingFaceTB/finemath allenai/peS2o; do
  echo "=== $id"
  curl -s --max-time 30 "https://huggingface.co/api/datasets/$id" \
    | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
except Exception as e:
    print('   no metadata:', e); raise SystemExit
t = d.get('cardData') or {}
lic = t.get('license') or d.get('license') or 'NOT STATED'
print('   licence:', lic, '| gated:', d.get('gated'), '| downloads:', d.get('downloads'))
" 2>/dev/null
  curl -s --max-time 30 "https://huggingface.co/api/datasets/$id/tree/main?recursive=1" \
    | python3 -c "
import sys, json
try: rows = json.load(sys.stdin)
except Exception: raise SystemExit
tot, n = 0, 0
for r in rows:
    if r.get('type') == 'file' and r.get('path','').endswith('.parquet'):
        tot += r.get('size') or 0; n += 1
        if n <= 2: print(f'   {(r.get(\"size\") or 0)/1e6:9.1f} MB  {r[\"path\"]}')
print(f'   {n} parquet files, {tot/1e9:.2f} GB total')
" 2>/dev/null
done