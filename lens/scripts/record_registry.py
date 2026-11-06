"""record_registry.py — load_vision extended registry; dummy refusal; see arm = K23.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 late — load_vision extended registry; dummy checkpoints REFUSED; see arm = K23

- `lens/vision.py load_vision` resolves concepts from the FROZEN registry (base SPEC +
  extended + more + band) by (name, formula) — same strictness as SeeTalkNet.load. Verified:
  ladder K=13 loads · ktest_real K=23 loads · forward fires named concepts · explain identity
  on the extended see-only model = 0.00e+00.
- **Deny-by-construction, free:** `ktest_dummy` weights REFUSE to load as readable ("concept
  'dummy_1' is not defined") — a random-projection model cannot masquerade as an explainable
  product even if someone points a client at its file.
- `variants.build_default()` see arm -> ktest_real_s0_ep12 (84.36% test, the +2.5 proven k1
  gain). Full regression after the change: variants smoke 80.2% test noun position, refusal OK,
  talk_sentence deterministic; selfcheck 28/28 PASS.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended", len(LOG))
