"""lens.cuda — our own GPU layer.

The stack, bottom to top:
    driver.py   ctypes -> CUDA driver API (cuInit, cuMemAlloc, cuLaunchKernel, ...)
    nvrtc.py    ctypes -> NVRTC, which compiles our kernels.cu to PTX at runtime
    kernels.cu  our own CUDA kernels (elementwise, tiled matmul, fused softmax+CE)
    this file   device setup, one-time compile, kernel launch wrappers

Nothing is imported from cupy, torch, or cuda-python. NVRTC is a compiler, the GPU equivalent
of gcc; the compute is entirely ours.

LAZY BY DESIGN: importing lens.cuda does not touch the GPU. The device and the compiled module
are created on first use, so the CPU layers keep working on a machine with no GPU.
"""
import os

import ctypes

import numpy as np

from lens.cuda import driver, nvrtc

_HERE = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SRC = os.path.join(_HERE, "kernels.cu")

_device = None
_module = None
_functions = {}
_ptx = None

# launch configuration for this card. 22 SMs on a GTX 1660 SUPER; 512 threads per block for
# the elementwise kernels, and a 16x16 tile for the matmul kernels (matching TS in kernels.cu).
BLOCK_1D = 512
GRID_CAP = 65535
TS = 16


def device():
    """The CUDA device, created once."""
    global _device
    if _device is None:
        _device = driver.Device(0)
    return _device


def ptx(force=False):
    """Compile our kernels once and cache the PTX."""
    global _ptx
    if _ptx is None or force:
        src = open(_KERNEL_SRC).read()
        _ptx = nvrtc.compile_cuda(src, arch="compute_75")
    return _ptx


def module():
    global _module
    if _module is None:
        device()                       # a context must exist before loading a module
        _module = driver.load_module(ptx())
    return _module


def fn(name):
    if name not in _functions:
        _functions[name] = driver.get_function(module(), name)
    return _functions[name]


def _grid_1d(n):
    return min((n + BLOCK_1D - 1) // BLOCK_1D, GRID_CAP)


def _ew(kname, a, b, out):
    """Elementwise op: out = a <op> b, all same size.

    The first version of this passed only TWO buffers to k_add, which takes three (a, b, out)
    plus n. The kernel then read b from a garbage pointer and segfaulted. The minimal-repro
    diagnostic isolated it: a hand-built 4-argument launch worked while this wrapper crashed.
    """
    n = a.size
    if b.size != n or out.size != n:
        raise ValueError(f"{kname}: sizes must match, got {a.size}, {b.size}, {out.size}")
    buf_a = driver.Buffer(a.nbytes)
    buf_b = driver.Buffer(b.nbytes)
    buf_o = driver.Buffer(out.nbytes)
    buf_a.from_host(a)
    buf_b.from_host(b)
    driver.launch(fn(kname), (_grid_1d(n), 1, 1), (BLOCK_1D, 1, 1),
                  [buf_a, buf_b, buf_o, np.int64(n)])
    device().sync()
    buf_o.to_host(out)
    buf_a.free()
    buf_b.free()
    buf_o.free()
    return out


def add(a, b, out):
    """out = a + b on the GPU."""
    return _ew("k_add", a, b, out)


def relu(x, out):
    n = x.size
    bx = driver.Buffer(x.nbytes)
    bo = driver.Buffer(out.nbytes)
    bx.from_host(x)
    driver.launch(fn("k_relu"), (_grid_1d(n), 1, 1), (BLOCK_1D, 1, 1),
                  [bx, bo, np.int64(n)])
    device().sync()
    bo.to_host(out)
    bx.free()
    bo.free()
    return out


def sigmoid(x, out):
    n = x.size
    bx = driver.Buffer(x.nbytes)
    bo = driver.Buffer(out.nbytes)
    bx.from_host(x)
    driver.launch(fn("k_sigmoid"), (_grid_1d(n), 1, 1), (BLOCK_1D, 1, 1),
                  [bx, bo, np.int64(n)])
    device().sync()
    bo.to_host(out)
    bx.free()
    bo.free()
    return out


def mul(a, b, out):
    """out = a * b on the GPU."""
    return _ew("k_mul", a, b, out)


def mul_scalar(x, s, out):
    """out = x * s on the GPU. `s` is a Python float, passed as a kernel argument value."""
    n = x.size
    bx = driver.Buffer(x.nbytes)
    bo = driver.Buffer(out.nbytes)
    bx.from_host(x)
    driver.launch(fn("k_mul_scalar"), (_grid_1d(n), 1, 1), (BLOCK_1D, 1, 1),
                  [bx, ctypes.c_float(float(s)), bo, np.int64(n)])
    device().sync()
    bo.to_host(out)
    bx.free()
    bo.free()
    return out


def matmul_abt(A, B, out):
    """out[m,n] = A[m,k] @ B[n,k]^T, the dA gradient, using the tiled kernel."""
    m, k = A.shape
    n, k2 = B.shape
    if k != k2:
        raise ValueError(f"matmul_abt shape mismatch: {A.shape} @ {B.shape}^T")
    bA = driver.Buffer(A.nbytes)
    bB = driver.Buffer(B.nbytes)
    bC = driver.Buffer(out.nbytes)
    bA.from_host(A)
    bB.from_host(B)
    grid = ((n + TS - 1) // TS, (m + TS - 1) // TS, 1)
    driver.launch(fn("k_matmul_abt"), grid, (TS, TS, 1),
                  [bA, bB, bC, np.int32(m), np.int32(k), np.int32(n)])
    device().sync()
    bC.to_host(out)
    bA.free()
    bB.free()
    bC.free()
    return out


def softmax_ce(logits, labels, probs_out, dlogits_out):
    """Fused softmax + cross-entropy on the GPU. One block per row, 256 threads.

    Returns the mean loss. `labels` must be int32 and contiguous.
    Rows longer than MAXV in kernels.cu (4096) are rejected here rather than silently truncated.
    """
    rows, V = logits.shape
    if V > 4096:
        raise ValueError(f"softmax_ce: vocab {V} exceeds the kernel's MAXV of 4096")
    bl = driver.Buffer(logits.nbytes)
    blab = driver.Buffer(labels.nbytes)
    bp = driver.Buffer(probs_out.nbytes)
    bd = driver.Buffer(dlogits_out.nbytes)
    bl.from_host(logits)
    blab.from_host(labels)
    shared = V * 4                                  # the row staged in shared memory
    driver.launch(fn("k_softmax_ce"), (rows, 1, 1), (256, 1, 1),
                  [bl, blab, np.int32(rows), np.int32(V), bp, bd], shared=shared)
    device().sync()
    bp.to_host(probs_out)
    bd.to_host(dlogits_out)
    for b in (bl, blab, bp, bd):
        b.free()
    # the mean loss, from the probabilities we already have
    import numpy as _np
    p = probs_out[_np.arange(rows), labels]
    return float(-_np.log(_np.maximum(p, 1e-12)).mean())


def matmul(A, B, out):
    """out[m,n] = A[m,k] @ B[k,n], using the tiled kernel."""
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {A.shape} @ {B.shape}")
    bA = driver.Buffer(A.nbytes)
    bB = driver.Buffer(B.nbytes)
    bC = driver.Buffer(out.nbytes)
    bA.from_host(A)
    bB.from_host(B)
    grid = ((n + TS - 1) // TS, (m + TS - 1) // TS, 1)
    driver.launch(fn("k_matmul"), grid, (TS, TS, 1),
                  [bA, bB, bC, np.int32(m), np.int32(k), np.int32(n)])
    device().sync()
    bC.to_host(out)
    bA.free()
    bB.free()
    bC.free()
    return out


def info():
    d = device()
    return {"device": d.name, "total_mib": d.total_mem / 2**20,
            "nvrtc": nvrtc.info(), "ptx_bytes": len(ptx())}
