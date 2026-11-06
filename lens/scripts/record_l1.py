"""record_l1.py — record the L1 gate (our own autograd) and the two bugs it caught.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 20:20 — L1 GATE: our own autograd PASSES 13/13 (after two real bugs and one bad instrument)

**`lens/tensor.py`** — our own tensors and our own autograd, written from scratch, using the L0
kernels.

**THE DESIGN CHANGE THAT FIXES THE OLD ENGINE'S BUG.** The old engine gave every op a `backward`
CLOSURE capturing its output, which creates output -> closure -> output reference cycles that only
the cycle collector can reclaim. This engine has **no closures at all**: a node stores only its
parents, an op name, and the small data the backward needs, and the gradient is **passed in as an
argument**. Two tests prove it:
* `no cycle: output not reachable` — walked every ancestor of an output; the output never reappears
* `200 graphs freed` — live Tensor count delta **-1** over 200 build/backward/drop cycles (was the leak)

**Two real bugs, both caught by the gate:**
1. **Scalar shape.** `sum_all` and `softmax_ce` returned shape-`(1,)` tensors while the backward
   did `float(grad)`, which raises on a 1-element array. Fixed to true 0-d scalars, and the
   extraction is now robust to either.
2. **Wrong matmul backward.** I used `matmul_atb` for BOTH gradients, but `dA = grad @ b^T` is a
   different kernel — the gate failed with `matmul_atb shape mismatch: (12,8)^T @ (16,8)`, which
   is the shapes not even matching. Added `gc_matmul_abt_f32` (A @ B^T) to the C, bound it, and
   used the correct kernel for each side. The C comment records the failure.

**A third finding — an instrument that could not measure.** The gate first reported
`softmax_ce (fused) worst rel diff 3.879e-02` as a FAILURE. An eps sweep showed the textbook
U-shape (truncation left, float noise right, minimum 3.8e-06 at eps=3e-3), and the analytic
gradient matches an independently-computed **float64 reference to 1.121e-09**. So the gradient was
always correct and the finite difference was the wrong instrument: one logit of a mean-over-32
loss moves the loss by ~3e-6, at the edge of float32's ~1e-7 precision. Evidence: the same check
gives **3.8e-06 in float64** and **1.29e-01 in float32** at the same eps — a 34,000x gap from
arithmetic alone.
**The finite-difference check for `softmax_ce` is therefore removed and replaced with the float64
reference**, which is strictly stronger. Every other op keeps its finite difference (no mean over
rows, so the signal is 32x larger). This is stated plainly rather than quietly dropped.

**Also:** my own diagnostic script printed the conclusion "there is a REAL BUG" from data that
showed the opposite — a hair-trigger ratio test (`worst[1e-2] > worst[best]*2`) that failed by
7.64e-06 vs 6.97e-06. Replaced with a shape test. **That is the fourth time today a wrong
conclusion came from my own code and was caught by reading the numbers rather than trusting the
verdict line.**

**L1 GATE: 13/13** — add, mul, matmul, matmul+bias, relu, sigmoid, a 2-layer net, and the fused
softmax+CE, all with printed numbers.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = "9. **Next in the rebuild:** L1 — our own tensors and autograd, with the reference-cycle issue the old engine had."
new = ("9. **L1 DONE — our own autograd PASSES 13/13.** The reference-cycle bug is fixed by design "
       "(no closures; the gradient is passed in), proven by a reachability test and a leak test "
       "(live Tensors delta -1 over 200 graphs). The gate caught two real bugs: a scalar-shape "
       "bug in the losses, and a wrong matmul backward that used the wrong kernel entirely. A "
       "third finding: the finite-difference check on the fused softmax was an instrument that "
       "could not measure (float32 mean-loss signal ~3e-6 vs ~1e-7 precision) — it is replaced "
       "by a float64 reference that agrees to 1.1e-09. **Next: L2, our own GPU layer.**")
assert old in txt
txt = txt.replace(old, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
