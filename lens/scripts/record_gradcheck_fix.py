"""record_gradcheck_fix.py — the gradient check was the foundation, and it was NOT 100%: two real
bugs, found by the numeric check the user said to trust. Both fixed, both gated.
"""
import os
import json

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-10 — THE GRADIENT CHECK WAS NOT 100%. TWO REAL BUGS FOUND. THIS IS THE FOUNDATION FIX.

### WHY IT WAS FOUND
The user said: "the small tests should get 100% before you move on." The gradient check had a
recorded value "0.003457" and was treated as passing. Running it fresh in the session showed it
FAILING at max relative difference 1.0. That is not float32 noise; that is a wrong gradient. Two
separate bugs, both real, both in the engine's verification story:

### BUG 1 — the verification was IMPOSSIBLE, not just failing: Tensor always cast to float32
`Tensor.__init__` forced `data.astype(np.float32)` on EVERY input, including float64 arrays. So
`gradcheck_vision.py`, which builds float64 tensors and compares against central differences at
eps=1e-6, could never pass — eps=1e-6 is below float32 resolution of the accumulated sums, and the
"numeric" gradient came back QUANTISED in steps of 2^-22/(2*eps) ≈ 0.119, differing from the
(documented-as-float64, actually-float32) analytic gradient by up to 1.0. The module docstring
claimed "the gradient gate runs in float64" — the code silently contradicted it.
FIX: `T.set_dtype("float64")` / `current_dtype()`; Tensor honours the mode; float32 remains the
training default. Verified: reshape-op check went from FAIL 4.34e-01 to PASS 5.1e-10, and the full
op-table audit passes at 1e-9.

### BUG 2 — the concept-loss backward was wrong by a factor of pos_weight for support != 0.5
The forward computes  bce = -(w*t*log(h+eps) + (1-t)*log(1-h+eps))  with w = (1-support)/support,
but the backward used  dh = w*(h-t)/(h*(1-h)). Expanding at t=0, the old formula gives w/(1-h) where
the truth is 1/(1-h) — wrong by a factor of w. w==1 only at support==0.5, and w ranges to 49 for
rare concepts. THE CHECK MISSED IT FOR MONTHS because its test concepts were built with
support=0.5, which makes w=1 and hides the exact error the check exists to find. A gradient gate
that tests a single configuration can certify a wrong formula.
FIX: dh = (-w*t/(h+eps) + (1-t)/(1-h+eps)) / n, exact for every w. Added `gradcheck_concept_weight.py`
and a full op-table audit (`gradcheck_full_ops.py`) that sweeps MULTIPLE input ranges per op,
including supports 0.02/0.2/0.5/0.9 -> w = 49/4/1/0.111. All PASS at ~1e-9.

### SCOPE OF THE CONCEPT-LOSS BUG — which claims are now under suspicion
* medical lam_concept=0.15 checkpoints: 34 of 35 concepts had w != 1 (w up to 5.7). The readable
  arm's accuracy and the +24.62-vs-plain claim were trained against a distorted gradient.
* SHIPPING seetalk_k22_big: 0 of 22 concepts had w==1 (supports 0.02-0.97 -> w 0.03-48.7), trained
  with lam_concept=0.3. The shipped artifact's concept layer learned under a distorted gradient.
* lam_concept=0.00 models are UNAFFECTED (no concept loss).
RETRAIN LAUNCHED (2026-10-10): medical 3 seeds x (0.00, 0.15) => mfix_*; shipping K22 => 
seetalk_k22_big_fixedgrad. The pre-fix numbers are recorded as SUSPECT until the retrains land.

### THE GATE IS NOW REAL
`selfcheck.py` (the ship gate) NEVER ran a gradient check — bug 2 shipped for months because of it.
A gradient-integrity section now runs `gradcheck_full_ops.py` inside selfcheck, so a wrong formula
can no longer pass silently.

### LESSONS (recorded for the skill)
1. A gradient gate must test RANGES, not one config: a formula right only at one magic parameter
   (w=1) passes a single-config check while being wrong everywhere else.
2. A numerical check whose dtype is silently rounded is not a check at all: verify the array is the
   dtype the docstring claims (assert it) — the float32 cast made float64 central differences
   quantised, and the failure mode looked like "numeric noise" when it was "verification impossible".
3. The ship gate must include the foundation checks, or the foundation can rot without anyone
   noticing a FAILING check was not even in the gate.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended gradient-fix record")
print("  bug1: Tensor forced float32 -> set_dtype('float64') added; reshape check 4.3e-1 -> 5.1e-10")
print("  bug2: concept_bce backward wrong by factor w for w!=1 -> fixed; w-sweep gate added")
print("  scope: medical lc0.15 arm + SHIPPING K22 both retraining with the fix")