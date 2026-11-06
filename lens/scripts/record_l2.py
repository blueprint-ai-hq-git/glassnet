"""record_l2.py — record the L2 GPU milestone.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 19:40 — L2 MILESTONE: OUR OWN GPU LAYER WORKS. 9/9, EXACT, 4.09x OUR CPU.

**The whole from-scratch GPU path is proven end to end:** ctypes -> CUDA driver API -> our own
kernels compiled at runtime by NVRTC -> module load -> launch -> correct answer.

```
PASS  driver: device + context            NVIDIA GeForce GTX 1660 SUPER, 5746 MiB
PASS  nvrtc: compiler available           version 13.0 (from an existing venv's libnvrtc)
PASS  nvrtc: our kernels.cu compiles      PTX 33049 bytes
PASS  driver: module load + get_function  7/7 kernels found
PASS  GPU add == CPU add                  max abs diff 0.000e+00 over 1,000,000 elements
PASS  GPU relu == CPU relu                max abs diff 0.000e+00
PASS  GPU sigmoid == CPU sigmoid          max abs diff 1.788e-07
PASS  GPU tiled matmul 256x256x256        max rel diff 0.000e+00
PASS  GPU faster than our CPU kernel      SPEEDUP 4.09x
L2 SMOKE: 9/9 passed
```

**SPEED, honestly:**
```
matmul 512x784x256   GPU 129.32 GFLOPS   our CPU 31.63 GFLOPS   speedup 4.09x
```
That is also **2.1x faster than OpenBLAS** (61.91 GFLOPS measured earlier on the same shape).
**But it is only ~2.6% of the card's ~5000 GFLOPS peak** -- the tiling is basic: TS=16, no
double buffering, no vectorized loads, no register blocking. Large headroom remains.

**Nothing borrowed.** No cupy, no torch, no cuda-python. The one external component is **NVRTC**,
which is a COMPILER -- the GPU equivalent of gcc, which the CPU kernels already use. No kernel,
operator, or tensor routine comes from anywhere but this project.

**Three bugs, all mine, all caught by the smoke:**
1. **NVRTC could not find `libnvrtc-builtins`** -- it looks the companion library up through the
   dynamic loader rather than next to itself. Fixed by preloading it `RTLD_GLOBAL` before loading
   libnvrtc. Error text was `failed to open libnvrtc-builtins.so.13.0`.
2. **numpy scalar kernel arguments were rejected**, and width matters: a kernel taking `long n`
   needs 8 bytes while one taking `int m` needs 4. Passing the wrong width reads garbage from the
   argument block. Fixed with dtype-aware marshalling, not a blanket cast.
3. **The crash.** `k_add` takes FOUR arguments `(a, b, out, n)`; my `_ew` wrapper passed three, so
   the kernel read `b` from a garbage pointer and segfaulted silently. **A minimal-repro
   diagnostic isolated it**: a hand-built 4-argument launch worked while the wrapper crashed.
   Without that diagnostic the segfault said nothing at all.

**Files:** `lens/cuda/driver.py` (driver API), `lens/cuda/nvrtc.py` (runtime compiler),
`lens/cuda/kernels.cu` (our kernels), `lens/cuda/__init__.py` (device + launch wrappers),
`lens/tests/test_l2_smoke.py` (the gate), `lens/tests/diag_gpu_launch.py` (the minimal repro).

**Next in L2:** the remaining kernels (mul, mul_scalar, matmul_abt, softmax_ce) through the GPU
path, then a full GPU training loop, then optimise the tiling past 2.6% of peak.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = "**Next: L2, our own GPU layer.**"
new = ("**L2 MILESTONE: our own GPU layer WORKS — 9/9, exact, 4.09x our CPU and 2.1x faster than "
       "OpenBLAS (129 vs 62 GFLOPS), but only ~2.6% of the card's peak, so headroom remains. "
       "Nothing borrowed: ctypes -> driver API -> our own kernels compiled at runtime by NVRTC "
       "(a compiler, like gcc). Three bugs caught by the smoke, all mine — the worst being a "
       "4-argument kernel called with 3 arguments, which segfaulted silently until a "
       "minimal-repro diagnostic isolated it. Next: remaining kernels, then a full GPU training "
       "loop, then optimise the tiling.**")
assert old in txt
txt = txt.replace(old, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
