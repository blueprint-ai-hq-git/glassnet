"""record_regressions.py — record the regression suite and the audit-formula correction.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 14:40 — REGRESSION SUITE: 4/4. THE WRONG-AUDIT NEAR-MISS IS PINNED.

Four defects hit on 2026-10-06. Each now has a test, so none can return silently. **The audit
formula bug is the important one** -- it nearly put a wrong number in front of the owner.

**Bug 1 -- embedding got no gradient.** `ArithNet.forward()` built the reshape node with
`_backward=None`, so the embedding never received a gradient; the optimiser raised
`no gradient for 'emb'`. Caught by a smoke run before the 5-seed Target G sweep.
*Test 1* checks the gradient exists AND matches central differences, so a silently-zero gradient
cannot pass either: `max |g| 1.522e-01, worst rel diff 2.84e-08`.

**Bug 2 -- readable `loss()` returns a tuple.** `(loss, parts)`, so calling `.backward()` on the
return value crashes with `'tuple' object has no attribute 'backward'`. Killed the first G2 sweep.
*Test 2* asserts the signature is `(Tensor, dict)` and that `backward()` works on the first element.

**Bug 3 -- `CONCEPT_TARGETS` is None under `T3M_IMPORT_ONLY=1`.** Code that read that module global
crashed with `'NoneType' object is not iterable`. *Test 3* asserts the 9 targets load from the
pre-registered JSON in training order, and documents that the module global is legitimately None.

**Bug 4 -- THE WRONG AUDIT FORMULA.** The first G2 audit used `|h * W|` as contribution and the
raw (non-residualised) activation for AUROC, and reported **0/9 concepts at D=32**. The standing
meter is:
  * AUROC on the **length-residualised** activation: `resid = h[:,c] - A @ lstsq(A, h[:,c])`,
    `A = [1, log1p(non-pad tokens)]`
  * contribution as **logit gain**: `bk = concept.W[:,c].reshape(WIN,D)`, `wvec = bk @ emb[i]`,
    summed over fired windows containing token i, divided by the number of fired windows
  * firing `> 0.5`, floor 2%; pass = a >= 0.55 AND contrib >= 0.25 AND firing >= 0.02
**The 0/9 was my statistic, not the model's.** Caught by running a KNOWN-ANSWER CONTROL before
trusting anything: the rewritten audit reproduces Target E's recorded numbers to four decimals
(`pronoun_subject 0.5125/0.0769`, `is_archaic_address 0.7485/0.4147`, **R1 7/9 = recorded 7/9**).
*Test 4* asserts exactly that, and the G2 runner now refuses to start if the control fails.

**Fifth, harmless bug found while pinning the others:** the readable loss `parts` dict uses key
`ce`, not `lm`. The concept diagnostic printed `lm 0.0000 aux 0.0000` every epoch because it read
keys that do not exist. The AUROC columns in that diagnostic are unaffected (they never used
`parts`), but the loss columns were junk and are noted rather than left standing.

**Test totals: 21 passing** -- 7 library (`test_library.py`) + 10 readable-model
(`test_readable.py`) + 4 regressions (`test_regressions.py`).
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
anchor = "10. **Still open:**"
new = ("10. **Test totals: 21 passing** — 7 library + 10 readable-model + 4 regressions. The "
       "regression suite pins all four of today's defects, including the audit-formula bug that "
       "nearly produced a wrong number (the audit now has a known-answer control that runs as a "
       "hard gate).\n11. **Still open:**")
assert anchor in txt, "STATUS anchor not found"
txt = txt.replace(anchor, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
