"""lens._capi — ctypes bindings to our own kernels in liblens.so.

What is ours: every kernel, the loop order, the blocking, the fusion, the threading.
What is not: the raw memory block. NumPy arrays are used here ONLY as contiguous float32
buffers and as a reference oracle in the tests. No NumPy arithmetic is used to compute a
result — if you grep this file, the only numpy calls are np.zeros/np.empty (allocation) and
.ctypes.data_as (handing a pointer to our C). Replacing the allocator with our own is a
later step, deliberately deferred so the kernels can be verified first.

Nothing here allocates inside C. Every buffer is owned by the caller.
"""
import ctypes
import os

import numpy as np  # allocation + pointer access only

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBPATH = os.path.join(_HERE, "liblens.so")

if not os.path.exists(_LIBPATH):
    raise RuntimeError(
        f"liblens.so not found at {_LIBPATH}\n"
        "Build it with: bash lens/scripts/build_lens.sh")

_lib = ctypes.CDLL(_LIBPATH)

_f32 = ctypes.POINTER(ctypes.c_float)
_i32 = ctypes.POINTER(ctypes.c_int)
_sz = ctypes.c_size_t


def _sig(name, argtypes, restype=None):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


# ---------------------------------------------------------------- signatures
_add = _sig("gc_add_f32", [_f32, _f32, _f32, _sz])
_sub = _sig("gc_sub_f32", [_f32, _f32, _f32, _sz])
_mul = _sig("gc_mul_f32", [_f32, _f32, _f32, _sz])
_div = _sig("gc_div_f32", [_f32, _f32, _f32, _sz])
_adds = _sig("gc_add_scalar_f32", [_f32, ctypes.c_float, _f32, _sz])
_muls = _sig("gc_mul_scalar_f32", [_f32, ctypes.c_float, _f32, _sz])
_relu = _sig("gc_relu_f32", [_f32, _f32, _sz])
_sigmoid = _sig("gc_sigmoid_f32", [_f32, _f32, _sz])
_sum = _sig("gc_sum_f32", [_f32, _sz], ctypes.c_float)
_matmul = _sig("gc_matmul_f32", [_f32, _f32, _f32, ctypes.c_int, ctypes.c_int, ctypes.c_int])
_matmul_atb = _sig("gc_matmul_atb_f32",
                   [_f32, _f32, _f32, ctypes.c_int, ctypes.c_int, ctypes.c_int])
_matmul_abt = _sig("gc_matmul_abt_f32",
                   [_f32, _f32, _f32, ctypes.c_int, ctypes.c_int, ctypes.c_int])
_softmax_ce = _sig("gc_softmax_ce_f32",
                   [_f32, _i32, ctypes.c_int, ctypes.c_int, _f32, _f32], ctypes.c_float)


def ptr(a):
    """Pointer to a contiguous float32 array's data."""
    return a.ctypes.data_as(_f32)


def iptr(a):
    """Pointer to a contiguous int32 array's data."""
    return a.ctypes.data_as(_i32)


# ---------------------------------------------------------------- public API
def add(a, b, out):
    _add(ptr(a), ptr(b), ptr(out), a.size)


def sub(a, b, out):
    _sub(ptr(a), ptr(b), ptr(out), a.size)


def mul(a, b, out):
    _mul(ptr(a), ptr(b), ptr(out), a.size)


def div(a, b, out):
    _div(ptr(a), ptr(b), ptr(out), a.size)


def add_scalar(a, s, out):
    _adds(ptr(a), ctypes.c_float(s), ptr(out), a.size)


def mul_scalar(a, s, out):
    _muls(ptr(a), ctypes.c_float(s), ptr(out), a.size)


def relu(x, out):
    _relu(ptr(x), ptr(out), x.size)


def sigmoid(x, out):
    _sigmoid(ptr(x), ptr(out), x.size)


def total(x):
    """Sum of a contiguous float32 array."""
    return float(_sum(ptr(x), x.size))


def matmul(A, B, out):
    """out[m,n] = A[m,k] @ B[k,n]. Shapes are checked in Python, not in C."""
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {A.shape} @ {B.shape}")
    _matmul(ptr(A), ptr(B), ptr(out), ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(n))


def matmul_atb(A, B, out):
    """out[m,n] = A[k,m]^T @ B[k,n]. The backward-pass workhorse."""
    k, m = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError(f"matmul_atb shape mismatch: {A.shape}^T @ {B.shape}")
    _matmul_atb(ptr(A), ptr(B), ptr(out), ctypes.c_int(k), ctypes.c_int(m), ctypes.c_int(n))


def matmul_abt(A, B, out):
    """out[m,n] = A[m,k] @ B[n,k]^T. Needed for dA in the backward pass."""
    m, k = A.shape
    n, k2 = B.shape
    if k != k2:
        raise ValueError(f"matmul_abt shape mismatch: {A.shape} @ {B.shape}^T")
    _matmul_abt(ptr(A), ptr(B), ptr(out), ctypes.c_int(m), ctypes.c_int(k), ctypes.c_int(n))


def softmax_ce(logits, labels, probs_out=None, dlogits_out=None):
    """Fused softmax + cross-entropy. Returns the mean loss.

    labels must be int32 and contiguous. Pass probs_out and/or dlogits_out to get the
    probability matrix and the gradient in the same pass -- that fusion is the point.
    """
    rows, V = logits.shape
    lp = ptr(logits)
    lab = labels.ctypes.data_as(_i32)
    pp = ptr(probs_out) if probs_out is not None else None
    dp = ptr(dlogits_out) if dlogits_out is not None else None
    return float(_softmax_ce(lp, lab, ctypes.c_int(rows), ctypes.c_int(V), pp, dp))


def f32(shape):
    """Allocate a contiguous float32 array. NumPy is the allocator only."""
    return np.zeros(shape, dtype=np.float32)


def i32(shape):
    return np.zeros(shape, dtype=np.int32)
