"""lens/fusedgemv.py - OUR OWN fused float16 GEMV, loaded from C through ctypes (PLAN item (f2b)).

WHY: (f2) measured that int8/f16 weights are 4.0000x / 2.0000x fewer bytes per decoded token and that
every NUMPY form of using them is SLOWER (1.689x to 4.741x the float32 reference, lens/runs/
INT8-DECODE.txt), because every "widen, then multiply" form writes a 4-byte-wide copy of the weights and
reads it back. The byte win is collectable only by a kernel that widens INSIDE the multiply. That kernel
is lens/f16gemv.c; this module builds it, loads it and exposes it as a decode arm.

WHAT IT IS NOT: it is not wired into any default path. lens/decode.py, lens/kvcache.py, lens/serve.py and
lens/quant.py are untouched by this module - an arm here is chosen by name, and the float32 path stays
what it was, so the queued KV-DECODE regression still re-measures the stored numbers against the code
they came from.

NOT CLAIMED: nothing about accuracy, nothing about training. This is a decode-path kernel.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.quant import QuantWindowCache                        # noqa: E402

SRC = os.path.join(_ROOT, "lens", "f16gemv.c")
BUILD_DIR = os.path.join(_ROOT, "lens", "runs", "_tmp")
SO = os.path.join(BUILD_DIR, "f16gemv.so")
CFLAGS = ["-O3", "-march=native", "-fopenmp", "-funroll-loops", "-shared", "-fPIC"]


def build(force=False, verbose=True):
    """Compile lens/f16gemv.c to lens/runs/_tmp/f16gemv.so when the source is newer. Returns (so, cmd)."""
    cmd = ["gcc"] + CFLAGS + ["-o", SO, SRC, "-lm"]
    fresh = (os.path.exists(SO) and os.path.getmtime(SO) >= os.path.getmtime(SRC))
    if force or not fresh:
        os.makedirs(BUILD_DIR, exist_ok=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"gcc failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        if verbose:
            print(f"built {SO}\n  {' '.join(cmd)}")
            if r.stderr.strip():
                print(f"  gcc stderr: {r.stderr.strip()}")
    return SO, cmd


class FusedGemv:
    """The ctypes front door. One instance per process; the .so is built on first use."""

    def __init__(self, force_build=False, verbose=True):
        so, self.cmd = build(force=force_build, verbose=verbose)
        self.so = so
        self.lib = ctypes.CDLL(so)
        p_f, p_u16 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16)
        self.lib.f16_gemv_f32.argtypes = [p_u16, p_f, p_f, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int]
        self.lib.f16_gemv_f32.restype = ctypes.c_int
        self.lib.f16_gemv_f32_scalar.argtypes = [p_u16, p_f, p_f, ctypes.c_int, ctypes.c_int,
                                                 ctypes.c_int]
        self.lib.f16_gemv_f32_scalar.restype = ctypes.c_int
        try:
            self.lib.f16_gemv_f32_avx2.argtypes = [p_u16, p_f, p_f, ctypes.c_int, ctypes.c_int,
                                                    ctypes.c_int, ctypes.c_int]
            self.lib.f16_gemv_f32_avx2.restype = ctypes.c_int
            self.has_avx2 = True
        except AttributeError:
            self.has_avx2 = False
        for name in ("f16_decode_sw", "f16_decode_fast"):
            fn = getattr(self.lib, name)
            fn.argtypes = [p_u16, p_f, ctypes.c_size_t]
            fn.restype = None
        self.nt_max = self._max_threads()

    def _max_threads(self):
        try:
            import multiprocessing
            return int(multiprocessing.cpu_count())
        except Exception:                                       # pragma: no cover
            return 1

    # ---------------------------------------------------------------- conversion checks
    def decode(self, bits, which="fast"):
        bits = np.ascontiguousarray(bits, dtype=np.uint16).reshape(-1)
        out = np.empty(bits.shape[0], dtype=np.float32)
        fn = self.lib.f16_decode_fast if which == "fast" else self.lib.f16_decode_sw
        fn(bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_size_t(bits.shape[0]))
        return out

    # ---------------------------------------------------------------- the kernel
    def gemv(self, bits, x, nthreads=3, out=None, scalar=False, variant="avx2"):
        """y = x @ W, with W given as raw float16 BIT PATTERNS (uint16, row-major, shape (n_in, n_out)).

        bits must be C-contiguous uint16 of shape (n_in, n_out); x is (m, n_in) float32.
        variant: "avx2" (block-outer, register accumulators) or "generic" (input-major, the first version,
        kept because it is the one that measured 4.5-9 GB/s and the record needs both).
        """
        bits = np.ascontiguousarray(bits, dtype=np.uint16)
        n_in, n_out = bits.shape
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != n_in:
            raise ValueError(f"x has {x.shape[1]} inputs, W has {n_in}")
        m = x.shape[0]
        if out is None:
            out = np.empty((m, n_out), dtype=np.float32)
        pb = bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        px = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        py = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if scalar:
            rc = self.lib.f16_gemv_f32_scalar(pb, px, py, m, n_in, n_out)
        elif variant == "avx2" and self.has_avx2:
            rc = self.lib.f16_gemv_f32_avx2(pb, px, py, m, n_in, n_out, int(nthreads))
        elif variant in ("avx2", "generic"):
            rc = self.lib.f16_gemv_f32(pb, px, py, m, n_in, n_out, int(nthreads))
        else:
            raise ValueError(f"unknown variant {variant!r}")
        if rc != 0:
            raise RuntimeError(f"f16_gemv returned {rc}")
        return out


_GEMV = None


def gemv_singleton(force_build=False, verbose=True):
    global _GEMV
    if _GEMV is None:
        _GEMV = FusedGemv(force_build=force_build, verbose=verbose)
    return _GEMV


class FusedF16WindowCache(QuantWindowCache):
    """The decode step with f16 weights multiplied by OUR C kernel.

    Everything that is not a weight read is the parent's code (lens/quant.py, itself a subclass of the
    shipped lens/kvcache.py WindowCache): the same causal mask, the same sigmoid written the same way,
    the same window bookkeeping. Only the six matmuls differ, so a measured difference is attributable
    to the kernel.

    control=True is THE CONTROL: the same kernel, the same bytes, fed a column-REVERSED answer.W. If that
    does not change the tokens, the token-identity test is decoration and proves nothing.
    """

    def __init__(self, net, fold=True, nthreads=3, control=False, force_build=False):
        super().__init__(net, fold=fold, arm="f16")   # the parent already builds a fused WindowCache
        self.nthreads = int(nthreads)
        self.control = bool(control)
        self.g = gemv_singleton(force_build=force_build)
        # cache the uint16 views once: a view is free, but doing it per step is not the measurement.
        self._bits = {}
        for name in ("Wq", "Wk", "Wv", "Wo", "cW", "aW"):
            qw = getattr(self, "q" + name)
            b = np.ascontiguousarray(qw.q, dtype=np.float16).view(np.uint16)
            if self.control and name == "aW":
                b = np.ascontiguousarray(b[:, ::-1])          # wrong weights, same bytes
            self._bits[name] = b
        self.kernel_calls = 0

    def _mat(self, x, qw, key):
        b = self._bits[key]
        if b.shape[0] != qw.shape[0]:                         # guard: a view must match the matrix
            raise RuntimeError(f"{key}: bits {b.shape} vs weight {qw.shape}")
        self.kernel_calls += 1
        return self.g.gemv(b, x, nthreads=self.nthreads)

    def summary(self):
        return (super().summary() + f" | KERNEL lens/f16gemv.c, threads {self.nthreads}"
                f"{', CONTROL column-reversed answer.W' if self.control else ''}"
                f", kernel calls {self.kernel_calls}")
