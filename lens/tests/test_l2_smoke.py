"""lens/tests/test_l2_smoke.py — THE FIRST GPU MILESTONE.

Goal tonight is narrow and honest: prove the whole from-scratch GPU path works end to end --
driver API, NVRTC compilation, module load, kernel launch, memory copy -- by getting the RIGHT
ANSWER on the GPU for a handful of ops, compared against our own CPU kernels.

If this passes, L2 is proven viable and the remaining work is filling in kernels and optimising.
If it fails, the failure is the finding and it gets reported, not worked around.

NOT claimed here: speed. This is correctness of the path only. Speed comes after.
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

RNG = np.random.default_rng(0)
RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} {detail}")


print("=" * 96)
print("L2 SMOKE — our own GPU layer, end to end")
print("=" * 96)

# ---------------------------------------------------------------- 1. driver
try:
    from lens.cuda import driver, nvrtc
    d = driver.Device(0)
    report("driver: device + context", True,
           f"{d.name}, {d.total_mem / 2**20:.0f} MiB, ctx {d.ctx.value:#x}")
except Exception as e:
    report("driver: device + context", False, f"{type(e).__name__}: {e}")
    print("\nCannot continue without a device.")
    sys.exit(1)

# ---------------------------------------------------------------- 2. nvrtc
try:
    i = nvrtc.info()
    report("nvrtc: compiler available", True,
           f"version {i['version'][0]}.{i['version'][1]} from {i['loaded_from']}")
except Exception as e:
    report("nvrtc: compiler available", False, f"{type(e).__name__}: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- 3. compile our kernels
try:
    from lens.cuda import ptx, module, fn
    p = ptx()
    report("nvrtc: our kernels.cu compiles", True, f"PTX {len(p)} bytes")
except Exception as e:
    report("nvrtc: our kernels.cu compiles", False, f"{type(e).__name__}: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- 4. load + find functions
try:
    m = module()
    names = ["k_add", "k_mul", "k_relu", "k_sigmoid", "k_matmul", "k_matmul_abt", "k_softmax_ce"]
    found = []
    for n in names:
        try:
            fn(n)
            found.append(n)
        except Exception:
            pass
    report("driver: module load + get_function", len(found) == len(names),
           f"{len(found)}/{len(names)} kernels found: {', '.join(found)}")
except Exception as e:
    report("driver: module load + get_function", False, f"{type(e).__name__}: {e}")
    sys.exit(1)

# ---------------------------------------------------------------- 5. the actual compute
from lens import _capi as C          # our CPU kernels, the reference
from lens import cuda as G           # our GPU layer

n = 1_000_000
a = RNG.standard_normal(n).astype(np.float32)
b = RNG.standard_normal(n).astype(np.float32)

try:
    gpu_out = np.zeros(n, dtype=np.float32)
    G.add(a, b, gpu_out)                    # gpu_out = a + b
    cpu_out = np.zeros(n, dtype=np.float32)
    C.add(a, b, cpu_out)
    dmax = float(np.abs(gpu_out - cpu_out).max())
    report("GPU add == CPU add", dmax == 0.0, f"max abs diff {dmax:.3e} over {n} elements")
except Exception as e:
    report("GPU add == CPU add", False, f"{type(e).__name__}: {e}")

x = RNG.standard_normal(n).astype(np.float32)
try:
    g = np.zeros(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    G.relu(x, g)
    C.relu(x, c)
    dmax = float(np.abs(g - c).max())
    report("GPU relu == CPU relu", dmax == 0.0, f"max abs diff {dmax:.3e}")
except Exception as e:
    report("GPU relu == CPU relu", False, f"{type(e).__name__}: {e}")

try:
    g = np.zeros(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    G.sigmoid(x, g)
    C.sigmoid(x, c)
    dmax = float(np.abs(g - c).max())
    report("GPU sigmoid == CPU sigmoid", dmax < 1e-6, f"max abs diff {dmax:.3e}")
except Exception as e:
    report("GPU sigmoid == CPU sigmoid", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 6. tiled matmul
m_, k_, n_ = 256, 256, 256
A = RNG.standard_normal((m_, k_)).astype(np.float32)
B = RNG.standard_normal((k_, n_)).astype(np.float32)
try:
    g = np.zeros((m_, n_), dtype=np.float32)
    c = np.zeros((m_, n_), dtype=np.float32)
    G.matmul(A, B, g)
    C.matmul(A, B, c)
    scale = max(float(np.abs(c).max()), 1.0)
    rel = float(np.abs(g - c).max()) / scale
    report(f"GPU tiled matmul {m_}x{k_}x{n_}", rel < 1e-4, f"max rel diff {rel:.3e}")
except Exception as e:
    report(f"GPU tiled matmul {m_}x{k_}x{n_}", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 7. speed, reported honestly
try:
    m_, k_, n_ = 512, 784, 256
    A = RNG.standard_normal((m_, k_)).astype(np.float32)
    B = RNG.standard_normal((k_, n_)).astype(np.float32)
    g = np.zeros((m_, n_), dtype=np.float32)
    G.matmul(A, B, g)
    t0 = time.perf_counter()
    for _ in range(5):
        G.matmul(A, B, g)
    t_gpu = (time.perf_counter() - t0) / 5
    c = np.zeros((m_, n_), dtype=np.float32)
    C.matmul(A, B, c)
    t0 = time.perf_counter()
    for _ in range(5):
        C.matmul(A, B, c)
    t_cpu = (time.perf_counter() - t0) / 5
    flops = 2.0 * m_ * k_ * n_
    print(f"\n  SPEED  matmul {m_}x{k_}x{n_}:  GPU {t_gpu*1e3:8.2f} ms "
          f"({flops/t_gpu/1e9:7.2f} GFLOPS)   CPU {t_cpu*1e3:8.2f} ms "
          f"({flops/t_cpu/1e9:7.2f} GFLOPS)   speedup {t_cpu/t_gpu:5.2f}x")
    report("GPU faster than our CPU kernel", t_gpu < t_cpu,
           f"{t_cpu/t_gpu:.2f}x speedup ({flops/t_gpu/1e9:.1f} vs {flops/t_cpu/1e9:.1f} GFLOPS)")
except Exception as e:
    report("GPU faster than our CPU kernel", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 96)
npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"L2 SMOKE: {npass}/{len(RESULTS)} passed")
for nm, st, dd in RESULTS:
    if st != "PASS":
        print(f"  !! {nm}: {dd}")
print("=" * 96)
sys.exit(0 if npass == len(RESULTS) else 1)
