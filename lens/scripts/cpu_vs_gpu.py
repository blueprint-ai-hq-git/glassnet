"""WHY the CPU was busy and the GPU idle - measured, not argued.

Our new-architecture comparison does 32 SEQUENTIAL small matmuls per window (256x256). A GPU wins on big
parallel work and loses on many tiny separate operations, because each one pays a fixed launch cost. This
measures the crossover on this exact machine so the split is a decision, not a guess.
"""
import time, numpy as np
try:
    import cupy as cp
    has_gpu = True
except Exception:
    has_gpu = False

D = 256
rng = np.random.default_rng(0)
def bench(n, reps=200):
    a = rng.standard_normal((n, D)).astype(np.float32)
    w = (rng.standard_normal((D, D)) * 0.06).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(reps): np.maximum(a @ w, 0)
    cpu = (time.perf_counter() - t0) / reps * 1000
    if not has_gpu:
        return cpu, None
    ag, wg = cp.asarray(a), cp.asarray(w)
    ag @ wg; cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        cp.maximum(ag @ wg, 0)
    cp.cuda.Stream.null.synchronize()
    gpu = (time.perf_counter() - t0) / reps * 1000
    return cpu, gpu

print(f"GPU available: {has_gpu}  ({cp.cuda.runtime.getDeviceProperties(0)['name'].decode() if has_gpu else 'none'})")
print(f"{'batch':>6} {'CPU ms':>9} {'GPU ms':>9} {'winner':>8} {'speedup':>8}")
for n in (32, 64, 128, 256, 512, 1024, 2048, 4096):
    cpu, gpu = bench(n)
    if gpu is None:
        print(f"{n:>6} {cpu:>9.4f} {'-':>9}")
    else:
        who = "GPU" if gpu < cpu else "CPU"
        sp = (cpu / gpu) if gpu < cpu else (gpu / cpu)
        print(f"{n:>6} {cpu:>9.4f} {gpu:>9.4f} {who:>8} {sp:>7.1f}x")
