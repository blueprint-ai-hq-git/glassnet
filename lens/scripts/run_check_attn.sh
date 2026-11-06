#!/bin/bash
# Which architecture did the shipped talk line actually use? The trainer exposes --attention
# (causal self-attention before the concept layer) as "the architecture lever". If the t11/t12
# lineage never enabled it, the model is a bag-of-window concept net with NO token order, which
# explains 56% next-token accuracy together with word-salad generation.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u - <<'PY'
import json
for f in ("lens/runs/chat/t11_talk41_log.json", "lens/runs/chat/t12_d512_full_log.json"):
    try:
        d = json.load(open(f))
        a = d.get("args", {})
        print(f"{f.split('/')[-1]:<28} attention={a.get('attention')} d={a.get('d')} window={a.get('window')} "
              f"assistant_only={a.get('assistant_only')} lam_concept={a.get('lam_concept')}")
    except Exception as e:
        print(f, "->", e)
PY