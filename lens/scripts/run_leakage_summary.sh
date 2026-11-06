#!/bin/bash
# Summarise a leakage JSON: own-label AUROC spread and the worst residualised confound overall.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import json, sys
p = "lens/runs/leakage_t12_splitfix_d512_ep192.json"
j = json.load(open(p))
rows = j["rows"]
print(f"  ship {j['ship']}   windows {j['n_windows']:,}   concepts {len(rows)}")
print(f"  bars: leak >= {j['bars']['leak']}, and within {j['bars']['close_to_own']} of own-label AUROC")
own = sorted(r["own_label_auroc"] for r in rows)
print(f"  own-label AUROC: min {own[0]:.3f}  median {own[len(own)//2]:.3f}  max {own[-1]:.3f}")
worst = max(rows, key=lambda r: r["worst_residual"])
print(f"  worst residualised confound anywhere: {worst['worst_residual']:.3f} "
      f"(concept {worst['concept']!r}, own-label {worst['own_label_auroc']:.3f})")
print(f"  raw (non-residualised) confound means: "
      f"pos {sum(r['raw']['pos'] for r in rows)/len(rows):.3f}  "
      f"len {sum(r['raw']['len'] for r in rows)/len(rows):.3f}  "
      f"early {sum(r['raw']['early'] for r in rows)/len(rows):.3f}")
ctrl = [r["control"] for r in rows]
print(f"  control (permuted activations): min {min(ctrl):.3f} max {max(ctrl):.3f} "
      f"-> {'METER SOUND' if abs(max(ctrl)-0.5) < 0.05 and abs(min(ctrl)-0.5) < 0.05 else 'METER BROKEN'}")
print(f"  leaky concepts: {sum(1 for r in rows if r['verdict'] == 'LEAKY')} of {len(rows)}")
PY