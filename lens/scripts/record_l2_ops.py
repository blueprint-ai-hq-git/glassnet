"""record_l2_ops.py — record the full L2 op gate.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 20:10 — L2 COMPLETE: THE FULL OP SET RUNS ON OUR OWN GPU. 14/14.

```
PASS  GPU add == CPU                   max abs diff 0.000e+00
PASS  GPU mul == CPU                   max abs diff 0.000e+00
PASS  GPU mul_scalar == CPU            max abs diff 0.000e+00
PASS  GPU relu == CPU                  max abs diff 0.000e+00
PASS  GPU sigmoid == CPU               max abs diff 1.788e-07
PASS  GPU matmul 64x784x128            max rel diff 0.000e+00
PASS  GPU matmul_abt 64x784x128        max rel diff 1.279e-06
PASS  GPU matmul 128x128x128           max rel diff 0.000e+00
PASS  GPU matmul_abt 128x128x128       max rel diff 4.399e-07
PASS  GPU matmul 256x256x256           max rel diff 0.000e+00
PASS  GPU matmul_abt 256x256x256       max rel diff 5.633e-07
PASS  GPU softmax_ce probs == CPU      max abs diff 2.235e-08
PASS  GPU softmax_ce dlogits == CPU    max abs diff 5.960e-08
PASS  GPU softmax_ce loss == CPU       GPU 5.528581  CPU 5.528581  diff 4.768e-07
L2 OPS GATE: 14/14 passed
```

**One more real kernel bug found and fixed.** `k_matmul_abt` failed on the GPU with rel diff
~1.4 -- nothing matched -- while the CPU version was right. Cause: the B tile was loaded using
`col` (which depends on `threadIdx.x`) where the tile row must depend on `threadIdx.y`. Every
thread read the wrong row of B. Fixed with an explicit `bRow = blockIdx.x * TS + threadIdx.y`,
and the comment in kernels.cu records the failure and the fix. Now 4.4e-07 .. 1.3e-06.

**NINE bugs today, every one caught by a gate, a control, a contradiction, or a self-check:**
1. sign error in the fused softmax loss (`- log(s)` for `+ log(s)`)
2. scalar-shape bug in both losses
3. wrong matmul backward (used the same kernel for both gradients)
4. my diagnostic's verdict line said "REAL BUG" from data showing the opposite
5. my rename script defeated its own guard by being written into the target directory
6. NVRTC could not find libnvrtc-builtins (fixed with an RTLD_GLOBAL preload)
7. numpy scalar kernel arguments rejected; width matters (long vs int)
8. **a 4-argument kernel called with 3 arguments** -- silent segfault until a minimal-repro
   diagnostic isolated it
9. **k_matmul_abt tile row index** -- rel diff ~1.4, fixed to threadIdx.y

**Zero of the nine were caught by being careful.** Every one came from a test that compares two
independent implementations, or from reading numbers instead of trusting a verdict.

**L2 is done.** The GPU layer runs the complete op set needed for training, on our own kernels,
compiled at runtime by our own NVRTC binding, driven through the CUDA driver API by ctypes.
Nothing borrowed: no cupy, no torch, no cuda-python.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = "**Next: remaining kernels, then a full GPU training loop, then optimise the tiling.**"
new = ("**L2 COMPLETE: the FULL op set runs on our own GPU — 14/14.** add/mul/mul_scalar/relu "
       "exact (0.000e+00), sigmoid 1.8e-07, matmul exact at three shapes, matmul_abt 4.4e-07.."
       "1.3e-06, fused softmax+CE probs 2.2e-08 / gradient 6.0e-08 / loss 4.8e-07. One more real "
       "kernel bug found and fixed: `k_matmul_abt` loaded its B tile with an index that depended "
       "on threadIdx.x where it needed threadIdx.y, giving rel diff ~1.4. **NINE bugs today, all "
       "caught by gates — none by being careful.** Next: L3, the readable layer on the new "
       "engine, then a GPU training loop.**")
assert old in txt
txt = txt.replace(old, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
