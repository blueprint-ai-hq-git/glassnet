"""diag_gpu_launch.py — minimal repro for the kernel launch crash.

The smoke got as far as compiling and loading our kernels, then died silently on the first
launch. "Silently" is the problem: a segfault in cuLaunchKernel tells you nothing. This does
ONE tiny launch with every step printed, so the failing call is visible.

Suspects, in order:
  1. the context is not CURRENT on this thread (legacy cuCtxCreate vs primary context)
  2. the kernel argument block is malformed (cuLaunchKernel wants an array of POINTERS)
  3. the kernel itself is wrong (grid/block mismatch, bad indexing)
"""
import ctypes
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

print("=" * 88)
print("GPU LAUNCH DIAGNOSTIC — one tiny kernel, every step printed")
print("=" * 88)

from lens.cuda import driver, nvrtc   # noqa: E402

print("1. device + context")
d = driver.Device(0)
print(f"   device {d.name}, ctx {d.ctx.value:#x}")

print("2. compile a trivial kernel")
src = r'''
extern "C" __global__ void k_smoke(const float *a, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * 2.0f;
}
'''
ptx = nvrtc.compile_cuda(src, arch="compute_75")
print(f"   PTX {len(ptx)} bytes")

print("3. load module + get function")
mod = driver.load_module(ptx)
fn = driver.get_function(mod, "k_smoke")
print(f"   module {mod.value:#x}, function {fn.value:#x}")

print("4. allocate")
host = np.arange(8, dtype=np.float32)
out = np.zeros(8, dtype=np.float32)
ba = driver.Buffer(host.nbytes)
bo = driver.Buffer(out.nbytes)
print(f"   {ba}  {bo}")

print("5. copy host -> device")
ba.from_host(host)
print("   ok")

print("6. launch (grid 1, block 8, n as c_long)")
args = [ba, bo, ctypes.c_long(8)]
holders = []
for a in args:
    if isinstance(a, driver.Buffer):
        holders.append(ctypes.c_void_p(a.ptr.value))
    elif isinstance(a, ctypes._SimpleCData):
        holders.append(a)
    else:
        holders.append(ctypes.c_long(int(a)))
print(f"   holder types: {[type(h).__name__ for h in holders]}")
print(f"   holder values: {[getattr(h, 'value', None) for h in holders]}")
pa = (ctypes.c_void_p * len(holders))(
    *[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders])
print(f"   ptr array {ctypes.sizeof(pa)} bytes")
rc = driver._cuLaunchKernel(fn, 1, 1, 1, 8, 1, 1, 0, None, pa, None)
print(f"   cuLaunchKernel returned {rc} ({driver._STATUS.get(rc, rc)})")
if rc != 0:
    sys.exit(1)

print("7. sync")
rc = driver._cuCtxSynchronize()
print(f"   cuCtxSynchronize returned {rc} ({driver._STATUS.get(rc, rc)})")
if rc != 0:
    sys.exit(1)

print("8. copy device -> host")
bo.to_host(out)
print(f"   got {out}")
print(f"   expected {host * 2}")
print(f"\nRESULT: {'PASS' if np.array_equal(out, host * 2) else 'WRONG ANSWER'}")
