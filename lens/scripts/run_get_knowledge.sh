#!/bin/bash
# Download the ALLOWED knowledge/maths slice. Every source below passed the licence gate
# (lens/runs/LICENCE-PROBE.txt): MIT, Apache-2.0 or CC0-1.0 only.
#
# Bounded on purpose: whole files, one or two shards per source, about 2.3 GB total, so the download is
# quick and 101 GB of free disk stays comfortable. Each file's size is verified after download; a
# mismatch is reported, never silently accepted.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
D=data/knowledge
mkdir -p "$D"
LOG=/tmp/knowledge_dl.log
exec > "$LOG" 2>&1
BASE=https://huggingface.co/datasets
echo "=== KNOWLEDGE SLICE DOWNLOAD $(date '+%F %H:%M')"

get () {   # get <url> <filename> <min-bytes-acceptable>
  local url="$1" out="$D/$2" want="$3"
  if [ -f "$out" ]; then
    local have
    have=$(stat -c %s "$out")
    if [ "$have" -ge "$want" ]; then echo "  have     $2 ($have bytes, >= minimum)"; return 0; fi
    echo "  REDOWNLOAD $2 (have $have, below minimum $want)"
  fi
  curl -sL --max-time 1800 -o "$out.part" "$url" || { echo "  FAILED curl $2"; return 1; }
  local have
  have=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
  # the sizes printed by the probe are rounded, so this is a floor check, not an exact match: a
  # truncated download is the thing worth catching, and the real size is printed for the record
  if [ "$have" -lt "$want" ]; then
    echo "  FAILED $2 size $have < floor $want (kept as .part, not trusted)"; return 1
  fi
  mv "$out.part" "$out"
  echo "  ok       $2 ($have bytes)"
}

echo "--- maths"
get "$BASE/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json" \
    "metamathqa_395k.json" 380000000
get "$BASE/EleutherAI/hendrycks_math/resolve/main/algebra/train-00000-of-00001.parquet" \
    "hendrycks_algebra.parquet" 400000
get "$BASE/open-r1/OpenR1-Math-220k/resolve/main/all/train-00000-of-00020.parquet" \
    "openr1_math_shard0.parquet" 190000000

echo "--- knowledge text"
get "$BASE/nampdn-ai/tiny-textbooks/resolve/main/train-00000-of-00001.parquet" \
    "tiny_textbooks.parquet" 800000000
get "$BASE/sedthh/gutenberg_english/resolve/main/data/train-00000-of-00037-f5fce855b93d2d02.parquet" \
    "gutenberg_shard0.parquet" 300000000
get "$BASE/sedthh/gutenberg_english/resolve/main/data/train-00001-of-00037-9f227d74fc154ce9.parquet" \
    "gutenberg_shard1.parquet" 300000000

echo
echo "--- landed:"
ls -la "$D" | tail -8
du -sh "$D"
echo "=== DONE $(date '+%F %H:%M')"