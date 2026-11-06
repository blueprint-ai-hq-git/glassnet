"""record_crossover.py — the measured GPU crossover.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 23:15 — CROSSOVER FOUND: THE GPU WINS ABOVE ~30,000 PARAMETERS.

The CPU-vs-GPU gate said the GPU was 0.24x at toy size. Instead of calling that a defect or
hiding it, the crossover was measured on the SAME architecture at increasing sizes:

```
size      params      CPU s     GPU s    speedup   verdict
toy        2,379     0.028     0.195     0.14x    CPU wins
small     29,475     0.354     0.116     3.06x    GPU wins
medium   266,147     1.708     0.106    16.05x    GPU wins

CROSSOVER: the GPU starts winning at small (29,475 params, batch 512).
```

**Read the two columns, not just the ratio.** The GPU time is essentially FLAT -- 0.195, 0.116,
0.106 -- while the CPU time grows with the work -- 0.028, 0.354, 1.708. That flat line is the
launch-overhead signature: the GPU pays a fixed per-step cost (roughly 200 kernel launches at
~15 us each) whether the matrices are tiny or large, so as soon as there is real arithmetic to
hide behind it, it wins. At 266k parameters it is **16.05x faster**.

**So the answer to "run everything on the GPU" is: yes, at model sizes worth running.** The toy
size is the exception and its cause is measured rather than asserted.

**The rebuilt `lens` stack now stands at:**
```
L0  our own C kernels         18/18 PASSED   (kept; no longer on the tensor path)
L1  our own tensors+autograd  12/12 PASSED   (still the engine)
L2  GPU, resident arrays       8/9 PASSED    correctness exact; the one FAIL is the toy-size
                                             speed line, now explained by the crossover above
L3  the readable layer         8/8 PASSED    explain() EXACT on BOTH devices
```

**A note on what the rebuild changed, in one line:** the ops are array expressions now, so the
same code runs on the CPU and the GPU, `explain()` is exact on both, and the CPU side got 4x
faster as a side effect because the hand-written C kernels were the bottleneck.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-06 23:15. **REBUILD: `lens`. L0-L3 done. The GPU now runs the real work.**

1. **Your rule change:** local GPU only (no cloud/L4); everything runs on the GPU; from-scratch relaxed **for the compute backend only** — **the model architecture and the library stay ours**. Implemented exactly that way and stated plainly: the GPU arithmetic is **CuPy** (which your original brief allowed); the concept layer, `explain()`, the losses, the optimizer, the autograd and the training loop are ours.
2. **Why the first GPU attempt was SLOWER (0.69x):** it copied operands to the card and back once per op. Measured: resident arrays give **4.0x** on matmul and **19.4x** on elementwise; with transfers, **0.5x and 0.8x**. The fix was keeping arrays where the work happens.
3. **Measured honestly:** CuPy **1900 GFLOPS** vs my hand-written CUDA kernels **129** vs NumPy **475**. **My hand-written GPU math was 14.7x slower than CuPy's.** Hand-writing it was the wrong bet.
4. **The model trains on the GPU with the SAME NUMBERS:** loss curves match to **4.8e-07**, trained parameters to **2.4e-07**, concept activations to **2.7e-07**. `explain()` is **EXACT on both devices** (residual 2.98e-08), coverage 100% on both.
5. **CROSSOVER, measured:** toy 2,379 params **0.14x** (CPU wins) · small 29,475 **3.06x** · medium 266,147 **16.05x**. **The GPU wins above ~30,000 parameters.** Its time is FLAT (0.195/0.116/0.106) while the CPU's grows (0.028/0.354/1.708) — launch overhead, not arithmetic.
6. **Side finding:** the CPU got **4x faster** in the same change, because the ops became array expressions. **My hand-written C kernels were the CPU bottleneck** (3.1x slower than NumPy). They stay in the tree and still pass their gates.
7. **Gates:** L0 18/18 · L1 12/12 · L2 GPU 8/9 (the one FAIL is the toy-size speed line, now explained by the crossover) · L3 8/8.
8. **14 bugs so far, every one caught by a gate or a self-check — none by being careful.** Latest: mixing cupy and numpy in one expression; parameters left on the host after the device switch; and a **process-global device** (documented as a real constraint in `lens/backend.py`).
9. **Next: L4 — the library** (the 5–20 line training API + `explain()`), then training at a size where the GPU actually pays, then pushing the readable model past the perplexity blocker.
10. **Still owed from you:** the **Target G decision** and the **human R4 readability score**.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
