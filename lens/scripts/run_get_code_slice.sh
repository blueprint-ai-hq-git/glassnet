#!/bin/bash
# Download the licence-cleared CODE slice (2026-10-18). Licences checked against Hugging Face metadata,
# not memory:
#   ise-uiuc/Magicoder-OSS-Instruct-75K        MIT          (public)
#   m-a-p/CodeFeedback-Filtered-Instruction    Apache-2.0   (public)
#   nvidia/OpenCodeInstruct                    CC BY 4.0    (public)
# SKIPPED: codeparrot/codeparrot-clean and bigcode/the-stack-smol -- licence NOT STATED (smol is also
# gated). Never trained on: HumanEval, MBPP, CodeContests (our exam papers).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
D=data/knowledge
mkdir -p "$D"
LOG=/tmp/code_slice_dl.log
exec > "$LOG" 2>&1
BASE=https://huggingface.co/datasets
echo "=== CODE SLICE DOWNLOAD $(date '+%F %H:%M')"

get () {
  local url="$1" out="$D/$2" want="$3"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" -ge "$want" ]; then
    echo "  have     $2 ($(stat -c %s "$out") bytes)"; return 0
  fi
  curl -sL --max-time 1800 -o "$out.part" "$url" || { echo "  FAILED curl $2"; return 1; }
  local have
  have=$(stat -c %s "$out.part" 2>/dev/null || echo 0)
  if [ "$have" -lt "$want" ]; then
    echo "  FAILED $2 size $have < floor $want"; head -c 120 "$out.part"; echo; return 1
  fi
  mv "$out.part" "$out"
  echo "  ok       $2 ($have bytes)"
}

get "$BASE/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl" \
    "magicoder_75k.jsonl" 180000000
get "$BASE/m-a-p/CodeFeedback-Filtered-Instruction/resolve/main/CodeFeedback-Filtered-Instruction.jsonl" \
    "codefeedback.jsonl" 330000000
get "$BASE/nvidia/OpenCodeInstruct/resolve/main/data/train-00000-of-00050.parquet" \
    "opencodeinstruct_0.parquet" 118000000
get "$BASE/nvidia/OpenCodeInstruct/resolve/main/data/train-00001-of-00050.parquet" \
    "opencodeinstruct_1.parquet" 103000000

echo
echo "--- landed:"
ls -la "$D"/magicoder_75k.jsonl "$D"/codefeedback.jsonl "$D"/opencodeinstruct_*.parquet 2>/dev/null \
  | awk '{print "   ", $5, $9}'
du -sh "$D"
echo "=== DONE $(date '+%F %H:%M')"