#!/bin/bash
# cosmopedia educational subsets (Apache-2.0, public): pick whole shards, bounded total.
# These subsets carry a real "prompt" column, so the human side of each conversation is a genuine
# instruction from the source rather than a wrapper I invented -- better than the gated textbooks file.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/COSMOPEDIA-PROBE.txt
: > "$OUT"
for sub in khanacademy openstax stanford auto_math_text web_samples_v2; do
  echo "=== $sub" | tee -a "$OUT"
  curl -s --max-time 40 "https://huggingface.co/api/datasets/HuggingFaceTB/cosmopedia/tree/main/data/$sub?recursive=1" \
    | python3 -c "
import sys, json
try: rows = json.load(sys.stdin)
except Exception as e: print('   unreadable:', e); raise SystemExit
rows = [r for r in rows if r.get('type')=='file' and r.get('path','').endswith('.parquet')]
rows.sort(key=lambda r: r['path'])
for r in rows[:3]:
    print(f\"   {(r.get('size') or 0)/1e6:9.1f} MB  {r['path']}\")
print(f'   {len(rows)} shards, {sum((r.get(\"size\") or 0) for r in rows)/1e9:.2f} GB total')
" | tee -a "$OUT"
done