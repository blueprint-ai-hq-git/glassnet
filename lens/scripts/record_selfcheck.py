"""record_selfcheck.py — the one-command stack proof, and the guard bug it caught on first run.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — SELFCHECK: ONE COMMAND PROVES THE WHOLE STACK (28 checks, all PASS, 2 s)

`lens/scripts/selfcheck.py` (+ `run_selfcheck.sh`). Exit 0 only if every check passes; missing
artifacts count as FAIL, not SKIP. Coverage: cleaner counts reconcile + OFF pass-through · all four
seetalk checkpoints (load with registry resolution, bit-identical round-trip, explain identity at
0.0, forward determinism) · TAMPERED concept formula must be refused (it is) · all three variants
answer · see-only asked to talk refuses · talk refuses a vocab-size mismatch · image-line
checkpoints load through load_vision.

**It caught a real bug on its first run:** the talk() vocab guard checked `id < len(vocab)`, which
still printed words from a TRUNCATED vocab — the camera-vocab bug class (sizes agree, contents
don't). The selfcheck probe swapped in a 3-word vocab and the interface happily answered. Guard is
now `len(vocab) == len(logits)` exactly, else refuse. 28 PASS after the fix.

This is the ship-gate artifact: any future change that breaks transparency, determinism, or
refusals fails one of these checks in 2 seconds instead of a video render.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
