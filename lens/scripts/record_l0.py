"""record_l0.py — record the L0 gate and the lens rename.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 19:20 — REBUILD STARTED: `lens` L0 GATE PASSES 18/18

**Owner instruction:** "start over and make everything from scratch, libraries every single thing,
even the GPU compatibility, and fully readable — it takes more time but we get more control, and
each thing is tested fully." Plus: "make it much faster."

**Naming (owner asked for memorable, useful, stable names).** The library is **`lens`**:
one syllable, universally known, and it means something here — a lens is glass (transparency) and a
lens is for seeing (`explain()`). The parts stay plain and obvious, exactly as `torch.nn` /
`torch.optim` do: `lens/kern.c`, `lens/cuda/`, `lens/tensor.py`, `lens/concepts.py`.
**Nothing deleted** — the rebuild sits alongside the old work; the old `.so` was moved, not removed.

**L0 = our own compute kernels.** `lens/csrc/kern.c` compiled to `liblens.so`
(`-O3 -march=native -ffast-math -fopenmp -funroll-loops`), driven by `ctypes`. No BLAS. NumPy is
used ONLY as the allocation and as the reference oracle in the gate.

**L0 GATE: 18/18 PASSED** — every op checked against NumPy:
```
add/sub/mul/add_scalar/mul_scalar/relu     max abs diff 0.000e+00
div                                        rel 1.037e-08
sigmoid                                    rel 1.567e-07
sum                                        rel 3.265e-08
matmul + matmul_atb, 3 shapes              rel ~5e-07
fused softmax+CE: loss / probs / gradient  rel ~1e-08
```

**One real bug found and fixed by the gate.** The first run was 17/18: `softmax_ce loss` was off
by 5.755 while **probs and dlogits matched to 1e-8**. That pattern localised the fault to a single
line — the loss had `- log(s)` where cross-entropy needs `+ log(s)`. Fixed; loss now matches to
4.8e-08. **The gate earned its keep on the first run.**

**SPEED, reported as measured, not as hoped:**
```
matmul 512x784x256   ours 19.86 GFLOPS   numpy 61.91 GFLOPS   3.12x slower
add 5,000,000 elems  ours 8.99 ms        numpy 5.10 ms        1.76x slower
```
**Our kernels are correct but slower than OpenBLAS.** That is the honest position: a clean loop
nest does not beat twenty years of hand-tuned assembly. Two routes to real speed, in order:
1. **register blocking + AVX2 intrinsics** (6x16 micro-kernel, packing) — the standard route to
   3-5x on this CPU, and real work
2. **the GPU layer** — the GTX 1660 SUPER is roughly **100x** the CPU's FLOPs, so Layer 2 is where
   "much faster" actually lives

**Next: L1 (our own tensors + autograd, with the reference-cycle issue the old engine had).**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
new = """# GLASSNET — STATUS

Updated 2026-10-06 19:20. **REBUILD STARTED — the library is now `lens`.**

1. **Owner instruction:** start over, everything from scratch (libraries, GPU, all of it), fully readable, tested fully, and much faster. Accepted and started. **Nothing deleted** — the rebuild sits alongside the old work.
2. **The name is `lens`** — one syllable, universally known, and it means something: a lens is glass (transparency) and a lens is for seeing (`explain()`). Parts stay plain, like `torch.nn`: `lens/kern.c`, `lens/cuda/`, `lens/tensor.py`, `lens/concepts.py`.
3. **L0 = our own C kernels, built and PASSING 18/18** against NumPy as the reference oracle. No BLAS.
4. **The gate caught a real bug on its first run:** the softmax cross-entropy loss was off by 5.755 while probs and gradient matched to 1e-8 — a `- log(s)` that should have been `+ log(s)`. Fixed; now matches to 4.8e-08.
5. **Speed, honestly: our kernels are SLOWER than OpenBLAS** — 3.12× on matmul (19.86 vs 61.91 GFLOPS), 1.76× on elementwise. Not dressed up.
6. **Where "much faster" actually comes from:** register blocking + AVX2 intrinsics on the CPU (3–5× available), and the GPU layer — the GTX 1660 SUPER is roughly **100×** the CPU. Layer 2 is the real win.
7. **Old record unchanged and still on disk:** A/A2 = NOT MET, A3 = MET (parity), C = PARTIAL, D = BEHIND, E = PARITY, F = LM collapse (concept claim withdrawn), G = NO WIN, G2 = FAILED on perplexity (coverage + readability PASSED at 3 sizes).
8. **Still owed from you:** the **Target G decision** (easier task / bigger model / both / stop) and the **human R4 readability score**.
9. **Next in the rebuild:** L1 — our own tensors and autograd, with the reference-cycle issue the old engine had.
10. **Machine:** GPU idle. Old work and new work both intact.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md` · `TARGET-G2-RESULTS.md`.
"""
open(p, "w").write(txt and new)
print("STATUS.md rewritten")
