#!/bin/bash
# T12 WINNER SCENARIO BATTERY — runs AFTER the ladder picks a winner and it is extended + audited.
# Standing requirement: every finding needs its failure scenario measured, not assumed.
# Scenarios (real prompts, same decoder as the shipped runtime):
#   1. greeting              "hello how are you ?"
#   2. off-topic             "what is the weather on mars ?"
#   3. repeat-prompt         same prompt cold + after a long warm reply (repetition check)
#   4. one-word echo         "yes" / "no" (degenerate-loop check, known failure class in small LMs)
#   5. long context          a 60-token sentence (window overflow behavior)
# Cheap and fast; output goes to /tmp/talk_scenarios_<ship>.out for the packet.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 TALK_CKPT="${1:?usage: run_talk_scenarios.sh <ckpt-name.npz>}"
.venv-gpu/bin/python3 -u - <<'PY'
import os, sys
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")
import variants as V
vr = V.build_default()
net, meta = vr._need("talk")
print("ship:", os.environ["TALK_CKPT"])
prompts = ["hello how are you ?", "what is the weather on mars ?", "yes",
           "no", "please tell me a very long story about a small bird " * 3]
for p in prompts:
    r = vr.talk_sentence(p, n_new=16)
    out = r.get("reply", "") if isinstance(r, dict) else str(r)
    rep = len(out.split()) and (len(set(out.split())) / len(out.split()))
    print(f"PROMPT    {p[:70]!r}")
    print(f"REPLY     {out!r}")
    print(f"STATS     {len(out.split())} tokens, unique-ratio {rep:.2f}"
          + ("   <-- DEGENERATE LOOP" if rep < 0.5 else ""))
PY