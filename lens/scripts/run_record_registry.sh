#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 - <<'PY'
LOG = """
---

## 2026-10-08 late — load_vision extended-registry; dummy checkpoints REFUSED; see arm = K23

- `lens/vision.py load_vision` now resolves concepts from the FROZEN registry (base SPEC +
  extended + more + band) by (name, formula) — same strictness as SeeTalkNet.load. Verified:
  ladder K=13 loads · ktest_real K=23 loads · forward fires named concepts · explain identity
  on the extended see-only model = 0.00e+00.
- **Deny-by-construction for free:** `ktest_dummy` weights REFUSE to load as readable ("concept
  'dummy_1' is not defined") — a random-projection model CANNOT masquerade as an explainable
  product, even if someone points a client at its file.
- `variants.build_default()` see arm -> ktest_real_s0 (84.36 test, +2.5 proven k1). Full
  regression after change: variants smoke (80.2% test noun position, refusal OK, talk_sentence
  deterministic) + selfcheck 28/28 PASS.
"""
open("LOG.md", "a").write(LOG)
print("appended")
PY
cp LOG.md STATUS.md API.md /home/yahya/ 2>/dev/null; echo synced; date '+%H:%M'; nvidia-smi --query-gpu=memory.used --format=csv,noheader
