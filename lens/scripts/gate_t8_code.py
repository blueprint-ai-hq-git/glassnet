"""gate_t8_code.py — TRACK 3 gate: does epochs 16->32 beat the best code model (30/150)?

Pre-registered BEFORE the run: PASS iff t8_code_ep32's syntax count at n=150 (own vocab,
v32000_clean) is STRICTLY greater than the current best's 30/150. Binary. Reported, never assumed.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import verdict_code_clean as V  # noqa: E402

BASELINE = 30  # cclean32k_ep16 (n=150, own vocab) — the measured current best
runs = os.path.join(PROJ, "lens", "runs", "code")
CAND = "t8_code_ep32"

m = V.measure(CAND, "v32000_clean")
if m is None:
    print("t8_code_ep32 not found or not measured yet — run the training first")
    sys.exit(3)
print(json.dumps({k: m[k] for k in ("ran", "syntax", "same", "unk") if k in m}, indent=1))
got = m.get("syntax", 0)
print(f"\n  gate (pre-registered): t8 syntax > {BASELINE} -> "
      f"{'PASS — epochs was a lever; promote t8 as best code model' if got > BASELINE else 'FAIL — not the lever; keep cclean32k_ep16'}")
print(f"  measured: {got} vs best {BASELINE}")
sys.exit(0 if got > BASELINE else 2)