"""record_to_cpu_fix.py — B.to_cpu on a Tensor silently returned a 0-d OBJECT array; fixed.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — B.to_cpu() BUG: a Tensor passed to to_cpu silently became a 0-d OBJECT array.

FOUND WHILE DRIVING THE FULL TEST SUITE (run_all_tests.py)
`verify_all_shipped.py` reported see-only logits shape () instead of (5, 10). The model was fine —
`net.forward()` returns dicts of **Tensor** objects, while `forward_arrays()` returns plain
ndarrays. `to_cpu()` did `np.asarray(tensor)`, and a Tensor has no __array__ hook going the expected
way, so numpy wrapped it as a **0-dimensional object array** (shape ()) instead of the numbers.
The talk path worked only because its logits came from `forward_arrays` (real ndarrays).

WHY THIS MATTERED BEYOND ONE TEST
Any code path that called `B.to_cpu(x)` with a Tensor and then compared shapes, argmax'ed, or
indexed would silently compare/transform the wrong object. `verify_all_shipped` caught it; the
gradcheck scripts call `B.to_cpu(p.grad)` where p.grad is a Tensor — the float64 gradient audits of
today would have been affected by the same trap before the fix (their PASS states were re-verified
AFTER the fix in the 14/14 suite run).

THE FIX (lens/backend.py to_cpu)
Detect a lens Tensor by `hasattr(a, 'requires_grad')` and unwrap `a.data` (which may itself be a
cupy array; .get() it in that case). NOT `hasattr(a, 'data')`: a numpy scalar also exposes a .data
memoryview, and calling .get() on that crashed all five gradcheck scripts the first way I wrote it.
Both mistakes are now guarded by the same 14-script suite that caught them.

LESSON
Helpers that "convert to numpy" must handle the engine's Tensor type explicitly, and the tests that
exercise them must actually pass Tensors (not just ndarrays) — otherwise a scalar-vs-array mixup
rounds silently to the wrong shape instead of failing loudly.

## 2026-10-10 — ALL 14 RUNNABLE TESTS PASS (first time)
```
  gradcheck_full_ops / concept_weight / extra_ops / vision / wce   PASS
  verify_all_shipped / explain_fidelity / talk_ship / mask         PASS
  verify_clean_cache (now CODE_DOC=clean aware)                    PASS
  selfcheck (incl. the new gradient-integrity gate)                PASS
  test_constraint / test_face_detect / test_face_real              PASS
```
The run_all_tests.py runner is now part of the workflow: a script that exists but is never executed
is not a test, and that is exactly how the gradient bug shipped for months.
"""
with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended to_cpu fix + 14/14 suite record")