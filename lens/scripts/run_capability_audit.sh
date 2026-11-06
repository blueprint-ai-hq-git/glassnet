#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
echo "############ 1. CAN EACH MODEL GENERATE TEXT, OR DOES IT ONLY PICK FROM A LIST? ############"
.venv-gpu/bin/python3 - <<'PY'
import json, os, glob
import numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"

print("  VL MODEL (vqa2) — what is its output shape?")
z = np.load(os.path.join(P, "vl", "vqa2_small_ep6.npz"), allow_pickle=True)
m = json.loads(str(z["__meta__"]))
print(f"    answer.W {z['answer.W'].shape}  -> it CHOOSES 1 of {z['answer.W'].shape[1]} answers")
print(f"    those answers: {m.get('avocab')}")
print("    => it CANNOT write free text. It picks from a fixed list.")
print()
print("  CHAT MODEL (best1) — what is its output?")
for f in sorted(glob.glob(os.path.join(P, "chat", "*_log.json")))[:3]:
    d = json.load(open(f))
    print(f"    {os.path.basename(f):<28} vocab {d.get('vocab_size', d.get('concepts', {}).get('unit', '?'))}")
PY

echo
echo "############ 2. IS THERE A SINGLE MODEL THAT BOTH SEES AND TALKS? ############"
ls -la lens/runs/vl/*.npz lens/runs/chat/*.npz 2>/dev/null | awk '{print "  "$NF}' | head -12
echo "  (a VL checkpoint has answer.W over 12-14 classes; a chat checkpoint predicts 12,000 words)"

echo
echo "############ 3. ANY TEST AGAINST AN OUTSIDE/BIGGER MODEL? ############"
grep -ril "llama\|qwen\|gpt-2\|gpt2\|mistral\|benchmark.*external\|vs.*baseline.*model" \
  --include=*.md --include=*.json --include=*.py lens/runs STATUS.md LOG.md 2>/dev/null | head -5 \
  || echo "  no external-model comparison files"
echo "  --- what 'bigger' was actually tested (our own models only) ---"
.venv-gpu/bin/python3 - <<'PY'
import json, os, glob
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs"
rows = []
for f in glob.glob(os.path.join(P, "chat", "*_log.json")):
    try:
        d = json.load(open(f))
        h = d.get("history", [{}])[-1]
        rows.append((os.path.basename(f).replace("_log.json", ""), d.get("params", 0),
                     h.get("val_ppl", 0)))
    except Exception:
        pass
for name, pr, ppl in sorted(rows, key=lambda r: r[1])[-6:]:
    print(f"    {name:<26} {pr:>12,} params   val ppl {ppl:>9.2f}")
PY
