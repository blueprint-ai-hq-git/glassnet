"""record_l3.py — record the L3 gate and the full rebuild session.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 21:00 — L3 GATE: 8/8. THE READABLE LAYER WORKS ON THE NEW ENGINE. L0-L3 ALL DONE.

```
PASS  concepts declared            has_marker_a (0.500), has_marker_b (0.500), seven_in_position_1 (0.401)
PASS  model builds                 1155 parameters, K=3 named concepts
PASS  1. concept loss is ACTIVE    parts = {'ce': 2.82004, 'concept_aux': 0.7379, 'concept_weight': 0.1}
PASS  2. its gradient reaches the layer   max |grad_with - grad_without| = 6.011e-03 on concept.W
PASS  2b. concept.W gets a non-zero gradient  max |grad| = 5.104e-02
PASS  3. explain() is EXACT        worst residual 5.960e-07 over 200 windows
PASS  4. coverage 100% by construction   reported WITH the audit, never alone
PASS  4b. at least half the concepts pass  3/3 (AUROC 0.8194 / 0.8660 / 0.6407)
L3 GATE: 8/8 passed
```

**Sample explanation -- the whole point of the project in six lines:**
```
IT SAYS  : '8'  (score +3.2822)
EXACT    : residual 1.19e-07 -> EXACT
BECAUSE  : 1 of 3 named concepts fired
   seven_in_position_1        fires 0.991   contribution +2.4804
   has_marker_a               fires 0.124   contribution -0.2296
```

**The two checks that exist because the old project got them wrong:**
* **the concept loss is ACTIVE and logged** -- the old code never logged it, so a run that trained
  on the wrong concept set looked identical to one that did not
* **its gradient provably reaches the concept layer** -- measured by differencing the gradient
  with and without the concept term (6.011e-03). A loss VALUE proves nothing; the gradient does.

**TWO more bugs, both caught by the gate:**
10. **A closure re-introduced out of habit.** The model attached a `_backward` closure to the
    embedding node -- exactly the pattern the new engine removed -- and crashed instantly with
    "'Tensor' object has no attribute '_backward'". The new engine has no such attribute BY
    DESIGN, so the old mistake cannot survive even when I reach for it. Fixed by registering an
    `embedding` op in the op table.
11. **A biased audit sample.** The gate audited `X[:500]`, which is entirely the corpus's first
    half, so one concept had no positives and another no negatives -- "no positives in this
    sample". Shuffled the sample; 3/3 then pass. A defect in the test, not the model.

## SESSION TOTALS

```
L0  our own C kernels         18/18 PASSED   liblens.so, no BLAS
L1  our own tensors+autograd  12/12 PASSED   cycle bug fixed by design, no leak
L2  our own GPU layer         14/14 PASSED   full op set, exact, 4.09x our CPU
L3  the readable layer         8/8 PASSED   exact explain, concept loss provably active
TOTAL                         52/52 PASSED
```

**ELEVEN bugs today, every one caught by a gate, a control, a contradiction, or a self-check.
Zero were caught by being careful.** That is the single most useful fact about this session.

**Nothing borrowed anywhere:** our own C kernels, our own tensors, our own autograd, our own CUDA
kernels compiled by our own NVRTC binding through the CUDA driver API by ctypes. No BLAS, no
cupy, no torch, no autograd library, no pretrained weights.

**Next: L4 (the library API), and a full GPU training loop, then optimise the tiling past 2.6% of
the card's peak.**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-06 21:00. **REBUILD: `lens` — L0, L1, L2, L3 ALL DONE. 52/52 gates passing.**

1. **Owner instruction:** start over, everything from scratch — libraries, GPU, all of it — fully readable, tested fully, faster. Accepted. **Nothing deleted**; the rebuild sits alongside the old work.
2. **The name is `lens`** — one syllable, universally known, and it means something: a lens is glass (transparency) and a lens is for seeing (`explain()`). Parts stay plain, like `torch.nn`.
3. **L0 DONE — our own C kernels, 18/18** against NumPy as the reference oracle. No BLAS.
4. **L1 DONE — our own autograd, 12/12.** The old engine's reference-cycle bug is **fixed by design** (no closures; the gradient is passed in), proven by a reachability test and a leak test.
5. **L2 DONE — our own GPU layer, 14/14 on the FULL op set.** add/mul/mul_scalar/relu **exact**, matmul **exact** at three shapes, matmul_abt 4.4e-07, fused softmax+CE probs 2.2e-08 / gradient 6.0e-08 / loss 4.8e-07.
6. **L3 DONE — the readable layer, 8/8.** `explain()` is **EXACT** (residual 5.96e-07); the concept loss is **provably active and provably reaching the layer**; 3/3 concepts pass the meter; coverage reported **with** the audit, never alone.
7. **Nothing borrowed:** ctypes → CUDA driver API → our own kernels compiled at runtime by NVRTC (a *compiler*, like gcc). No BLAS, no cupy, no torch, no autograd library, no pretrained weights.
8. **Speed, honestly:** GPU matmul **129.32 GFLOPS** vs our CPU 31.63 (4.09×) and vs OpenBLAS 61.91 (2.1×) — but only **~2.6% of the card's ~5000 GFLOPS peak**. CPU kernels are **slower** than OpenBLAS (3.1×). Not dressed up.
9. **ELEVEN bugs today, every one caught by a gate or a self-check — none by being careful.** The new engine even *refuses* the old mistake: a closure re-introduced out of habit crashed instantly because `_backward` does not exist by design.
10. **Next: L4 (the library API) + a full GPU training loop**, then optimise the tiling. Still owed from you: the **Target G decision** and the **human R4 readability score**.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
