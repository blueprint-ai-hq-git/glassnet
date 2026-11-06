#!/bin/bash
# Fetch the cosmopedia educational shards that replace the gated tiny-textbooks file.
# Licence: Apache-2.0, public (checked against Hugging Face metadata, not memory).
# Bounded: whole shards only, about 1.4 GB total.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
D=data/knowledge
mkdir -p "$D"
LOG=/tmp/cosmopedia_dl.log
exec > "$LOG" 2>&1
BASE=https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/resolve/main/data
echo "=== COSMOPEDIA DOWNLOAD $(date '+%F %H:%M')"

get () {   # get <subset/path> <localfile> <min-bytes>
  local rel="$1" out="$D/$2" want="$3"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" -ge "$want" ]; then
    echo "  have     $2 ($(stat -c %s "$out") bytes)"; return 0
  fi
  curl -sL --max-time 1800 -o "$out.part" "$BASE/$rel" || { echo "  FAILED curl $2"; return 1; }
  local have
  have=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
  if [ "$have" -lt "$want" ]; then
    echo "  FAILED $2 size $have < floor $want"; head -c 120 "$out.part"; echo; return 1
  fi
  mv "$out.part" "$out"
  echo "  ok       $2 ($have bytes)"
}

get "khanacademy/train-00000-of-00001.parquet"  "cosmo_khanacademy.parquet"   44000000
get "openstax/train-00000-of-00002.parquet"     "cosmo_openstax_0.parquet"    155000000
get "openstax/train-00001-of-00002.parquet"     "cosmo_openstax_1.parquet"    155000000
get "stanford/train-00000-of-00013.parquet"     "cosmo_stanford_0.parquet"    228000000
get "stanford/train-00001-of-00013.parquet"     "cosmo_stanford_1.parquet"    228000000
get "auto_math_text/train-00000-of-00018.parquet" "cosmo_math_0.parquet"      223000000
get "auto_math_text/train-00001-of-00018.parquet" "cosmo_math_1.parquet"      223000000

echo
echo "--- landed:"; ls -la "$D"/cosmo_*.parquet 2>/dev/null | awk '{print "   ", $5, $9}'
du -sh "$D"
echo "=== DONE $(date '+%F %H:%M')"