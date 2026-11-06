#!/bin/bash
# Probe the ALLOWED knowledge/maths datasets: what files exist, how big, so a bounded slice can be
# chosen deliberately instead of downloading whatever comes first. Read-only.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
OUT=lens/runs/DATA-PROBE.txt
: > "$OUT"
probe () {
  local id="$1"
  echo "=== $id" | tee -a "$OUT"
  curl -s --max-time 30 "https://huggingface.co/api/datasets/$id/tree/main?recursive=1" \
    | python3 -c "
import sys, json
try:
    rows = json.load(sys.stdin)
except Exception as e:
    print('   could not read file list:', e); raise SystemExit
tot = 0
for r in rows:
    if r.get('type') != 'file':
        continue
    p, sz = r.get('path',''), r.get('size') or 0
    if not p.endswith(('.parquet', '.jsonl', '.json', '.jsonl.gz', '.csv')):
        continue
    tot += sz
    print(f'   {sz/1e6:9.1f} MB  {p}')
print(f'   ---- total listed: {tot/1e9:.2f} GB')
" | tee -a "$OUT"
}
probe meta-math/MetaMathQA
probe EleutherAI/hendrycks_math
probe open-r1/OpenR1-Math-220k
probe nampdn-ai/tiny-textbooks
probe sedthh/gutenberg_english
echo | tee -a "$OUT"
echo "Choosing rule: take whole files only, prefer one or two shards per source, and keep the total" | tee -a "$OUT"
echo "under ~6 GB so the download is quick and the disk stays comfortable." | tee -a "$OUT"