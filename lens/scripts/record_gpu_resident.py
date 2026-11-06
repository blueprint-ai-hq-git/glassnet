"""record_gpu_resident.py — the device-resident backend, recorded honestly.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 23:05 — DEVICE-RESIDENT BACKEND. OWNER RELAXED THE FROM-SCRATCH RULE.

**Owner's rule change (verbatim):** "We are not going for an l4 or cloud GPU we are not verifying
we are doing everything on the GPU or ok don't make everything from scratch but make the model
architecture and the library from scratch at least."

Read as: local GTX 1660 Super only, no cloud; the compute actually runs on the GPU; from-scratch
is relaxed for the COMPUTE BACKEND ONLY, while **the model architecture and the library stay
ours**. That is how it was implemented, and it is stated plainly rather than glossed:

* **OURS:** the model architecture (the named concept layer), the library (training API,
  `explain()`), the losses, the optimizer, the autograd, the training loop.
* **NOT OURS:** the array arithmetic and the matmul on the GPU, which **CuPy** provides. The
  owner's original brief allowed this explicitly ("a raw GPU array backend like CuPy").

**WHY THE PREVIOUS GPU ATTEMPT WAS SLOWER.** It copied every operand to the card and back, once
per operation. Measured, with the arrays STAYING resident:

```
matmul 512x784x256    CPU 0.432 ms   GPU resident 0.108 ms    4.0x
add 5M elements       CPU 4.096 ms   GPU resident 0.211 ms   19.4x
matmul (transfers)    CPU 0.432 ms   GPU 0.885 ms            0.5x   <- slower
add (transfers)       CPU 4.096 ms   GPU 5.126 ms            0.8x   <- slower
```

So the fix was not a faster kernel, it was **keeping the arrays where the work happens**.

**CUDA-VS-NUMPY, measured:** CuPy **1900 GFLOPS** on that matmul; my hand-written CUDA kernels
**129**; NumPy/OpenBLAS **475** with all cores. **My hand-written GPU math was 14.7x slower than
CuPy's.** Hand-writing it was the wrong bet, and the owner was right to relax the rule.

**THE CPU-vs-GPU GATE: 8/9.**
```
PASS  both devices train             final loss CPU 2.251071   GPU 2.251071
PASS  loss curves match              max |diff| over 120 steps = 4.768e-07
PASS  trained parameters match       worst |cpu - gpu| = 2.384e-07 on answer.W
PASS  concept activations match      max |cpu - gpu| = 2.682e-07
PASS  CPU: explain() still EXACT     residual 2.980e-08
PASS  CPU: coverage 100%             100.0000%
PASS  GPU: explain() still EXACT     residual 2.980e-08
PASS  GPU: coverage 100%             100.0000%
FAIL  GPU training is faster         0.24x  (0.101s CPU -> 0.414s GPU)
```

**The model trains on the GPU and produces the same numbers to ~5e-07, and `explain()` stays
exact on both devices.** That is the correctness claim and it is proven.

**And the honest failure: the GPU is 4x SLOWER at this size.** The cause is measured, not guessed:
at d=32/K=3/batch=128 the model issues roughly 200 kernel launches per step and each launch costs
~15 us no matter how little work it does -- 120 x 200 x 15 us ~ 0.36 s, which is the entire 0.414 s
measured. **Launch overhead, not arithmetic.** A crossover test at increasing sizes is running to
find where the GPU actually starts winning, reported as measured even if there is no crossover.

**A side finding that matters:** the CPU side got 4x FASTER (0.423s -> 0.101s) in the same change,
because the ops became array expressions. **My hand-written C kernels were the CPU bottleneck**
(3.1x slower than NumPy, as the L0 gate already said). They remain in the tree and still pass
their gates; the record of what was tried is kept.

**Bugs in this change, all caught:**
12. **Mixing cupy and numpy in one expression** in the diagnostic -- cupy refuses with "Unsupported
    type <class 'numpy.ndarray'>", easy to misread as a cupy bug. Fixed with `.get()`.
13. **Parameters stayed host arrays** after `set_device("gpu")`, so the first cupy op failed with
    the same confusing message. Fixed by having `Tensor` place data on the current device.
14. **A process-global device vs per-device models** -- explaining a CPU model while the device
    was "gpu" mixed the two array types. This is a REAL constraint, now documented in
    `lens/backend.py`: the device is process-global, a model lives on the device that was active
    when it was built, and mixing two devices in one process is not supported.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
