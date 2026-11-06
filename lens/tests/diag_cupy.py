"""diag_cupy.py — what GPU backend do we actually have available?

The from-scratch rule was relaxed by the owner for the COMPUTE BACKEND only: the model
architecture and the library stay ours. CuPy is a raw GPU array backend, which the original
project brief explicitly allowed ("a raw GPU array backend like CuPy").

This checks what is actually installed and how fast it is, so the choice is made on measurements
rather than on a guess about what "should" be faster.
"""
import sys
import time

import numpy as np

print("=" * 88)
print("GPU BACKEND DIAGNOSTIC")
print("=" * 88)
print(f"python {sys.version.split()[0]}")

try:
    import cupy as cp
    print(f"cupy {cp.__version__}")
except Exception as e:
    print(f"cupy NOT available: {type(e).__name__}: {e}")
    sys.exit(1)

n = cp.cuda.runtime.getDeviceCount()
print(f"devices: {n}")
p = cp.cuda.runtime.getDeviceProperties(0)
name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
print(f"device 0: {name}")
free, total = cp.cuda.runtime.memGetInfo()
print(f"memory: {free/2**20:.0f} MiB free of {total/2**20:.0f} MiB")

print("\n--- correctness against numpy ---")
# NOTE: never mix a cupy array with a numpy array in one expression -- cupy refuses (correctly).
# Bring the GPU result back with .get() and compare on the CPU.
a = np.random.default_rng(0).standard_normal((512, 784)).astype(np.float32)
b = np.random.default_rng(1).standard_normal((784, 256)).astype(np.float32)
ga, gb = cp.asarray(a), cp.asarray(b)
gc = ga @ gb
print(f"matmul max abs diff vs numpy: {float(np.abs(gc.get() - (a @ b)).max()):.3e}")
gs = cp.sum(cp.exp(ga[:, :16]), axis=1)
print(f"exp/sum max abs diff vs numpy: {float(np.abs(gs.get() - np.exp(a[:, :16]).sum(1)).max()):.3e}")

print("\n--- speed: matmul 512x784x256 ---")


def timeit(fn, reps=50):
    fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / reps


flops = 2 * 512 * 784 * 256
t_np = timeit(lambda: a @ b, reps=20)
t_cp = timeit(lambda: ga @ gb, reps=50)
t_cp_transfer = timeit(lambda: cp.asarray(a) @ cp.asarray(b), reps=20)
print(f"  numpy (CPU)            {t_np*1e3:8.3f} ms   {flops/t_np/1e9:8.2f} GFLOPS")
print(f"  cupy (GPU, resident)   {t_cp*1e3:8.3f} ms   {flops/t_cp/1e9:8.2f} GFLOPS   "
      f"speedup {t_np/t_cp:.1f}x")
print(f"  cupy (with transfers)  {t_cp_transfer*1e3:8.3f} ms   "
      f"speedup {t_np/t_cp_transfer:.1f}x")

print("\n--- speed: elementwise add, 5M elements (the op I left on the CPU) ---")
big = np.random.default_rng(2).standard_normal(5_000_000).astype(np.float32)
gb1, gb2 = cp.asarray(big), cp.asarray(big)
t_np_add = timeit(lambda: big + big, reps=20)
t_cp_add = timeit(lambda: gb1 + gb2, reps=50)
t_cp_add_tr = timeit(lambda: cp.asarray(big) + cp.asarray(big), reps=20)
print(f"  numpy (CPU)            {t_np_add*1e3:8.3f} ms")
print(f"  cupy (GPU, resident)   {t_cp_add*1e3:8.3f} ms   speedup {t_np_add/t_cp_add:.1f}x")
print(f"  cupy (with transfers)  {t_cp_add_tr*1e3:8.3f} ms   speedup {t_np_add/t_cp_add_tr:.1f}x")

print("\nVERDICT: resident GPU arrays are the difference. Transfers per op destroy the win --")
print("which is exactly why the current per-op-copy path measured 0.69x (slower) at toy size.")
