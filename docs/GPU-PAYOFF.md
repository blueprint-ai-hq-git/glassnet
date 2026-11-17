
## A GPU port of the recurrent loop would lose, and the loop is not the bottleneck (2026-11-17)

Before writing the 247-line CuPy port, the payoff was measured with the real shapes from `scan_direct_batched.py` (B=32, T=128, D=128, H=256):

```
CPU (numpy):   forward 0.011 s   backward 0.012 s   total 0.024 s     (0.09 ms per timestep)
GPU (cupy):    forward 0.014 s   backward 0.015 s   total 0.028 s     (0.11 ms per timestep, incl. launch + sync)

SPEEDUP from porting this loop to GPU: 0.84x   -> GPU LOSES

off-loop reference, ONE batched (4096,128)@(128,256) matmul:
  CPU 0.0051 s   GPU 0.0001 s   speedup 37.50x
```

## Two conclusions, and the second one is the useful one

**1. Do not port the recurrent loop.** The forward is a Python `for t in range(128)` doing two small matmuls and a tanh per step. That is 128 kernel launches of microsecond-sized work, and launch overhead exceeds the arithmetic: **0.84x, slower than the CPU.** This is the same failure the project already recorded once, when per-op hand-written CUDA kernels measured 0.69x. The array library is not the problem; the loop is.

**2. But the loop is only 2.4% of the step, so it was never the bottleneck.** A measured step costs **10.5 s**. The loop accounts for **0.024 s** of that. So my own hypothesis - that Python-loop launch overhead was why run 9 is slow - is **wrong**, and the honest conclusion is that the cost lives somewhere else entirely, in the vocab-sized work: the readout is `(B,K,V)` with **K=64 and V=12,002**, plus einsums of the same order.

**Where a GPU port would actually pay** is exactly that bulk work: one batched matmul of the same arithmetic is **37.5x faster** on the GPU. So the port is worth doing *for the readout and the einsums*, and pointless for the recurrence. Those are different jobs, and only the first one is worth its engineering.

## Why this is worth recording even though it produced no model number

It cost two minutes and it **cancelled a 247-line change** that would have made the arm slower while looking like progress - and it also **falsified my own explanation** from the previous wake with a number, in the same run. A benchmark that only ever confirms a plan is not a benchmark; this one contradicted one plan and redirected it.
