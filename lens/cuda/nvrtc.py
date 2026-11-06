"""lens/cuda/nvrtc.py — our own runtime CUDA compiler wrapper.

NVRTC compiles CUDA C to PTX at runtime, in-process. It is the GPU equivalent of gcc: a
COMPILER, not a compute library. Nothing here borrows a tensor library, an operator, or a kernel.

The library is located by trying known paths rather than a full-disk search -- a recursive find
over / took over seven minutes and timed out. Paths are tried in order and the first that loads
and reports a version wins; the chosen path is printed so it is never a mystery which compiler
was used.

nvrtcGetPTX returns the PTX text, which we hand to cuModuleLoadData in driver.py.
"""
import ctypes
import os

CANDIDATES = [
    # CUDA 13 shipped inside the onnx venv -- verified working, nvrtcVersion 13.0
    "/run/media/yahya/work/onnx-venv/lib/python3.11/site-packages/nvidia/cu13/lib/libnvrtc.so.13",
    # CUDA 12.4 nvrtc from another venv
    "/run/media/yahya/work/1.5b-finetune/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    # a system toolkit, if one is ever installed
    "/usr/local/cuda/lib64/libnvrtc.so",
    "/usr/lib/x86_64-linux-gnu/libnvrtc.so",
]

_LOADED_FROM = None
_lib = None
_last_error = None

for _p in CANDIDATES:
    if not os.path.exists(_p):
        continue
    try:
        # NVRTC needs libnvrtc-builtins alongside it, and it looks it up through the dynamic
        # loader rather than next to itself. Without this preload the compile fails with
        # "failed to open libnvrtc-builtins.so.13.0". Loading it RTLD_GLOBAL puts it in the
        # global symbol scope so NVRTC can find it.
        _dir = os.path.dirname(_p)
        for _extra in sorted(os.listdir(_dir)):
            if _extra.startswith("libnvrtc-builtins"):
                try:
                    ctypes.CDLL(os.path.join(_dir, _extra), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        _cand = ctypes.CDLL(_p)
        _cand.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        _a, _b = ctypes.c_int(), ctypes.c_int()
        if _cand.nvrtcVersion(ctypes.byref(_a), ctypes.byref(_b)) == 0:
            _lib, _LOADED_FROM = _cand, _p
            NVRTC_VERSION = (_a.value, _b.value)
            break
    except Exception as e:                     # pragma: no cover
        _last_error = e

if _lib is None:
    raise RuntimeError(
        "no usable libnvrtc found. Tried:\n  " + "\n  ".join(CANDIDATES) +
        f"\nlast error: {_last_error}")


class NvrtcError(RuntimeError):
    pass


_prog_t = ctypes.c_void_p
_lib.nvrtcCreateProgram.argtypes = [ctypes.POINTER(_prog_t), ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
_lib.nvrtcCompileProgram.argtypes = [_prog_t, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
_lib.nvrtcGetPTXSize.argtypes = [_prog_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.nvrtcGetPTX.argtypes = [_prog_t, ctypes.c_char_p]
_lib.nvrtcGetProgramLogSize.argtypes = [_prog_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.nvrtcGetProgramLog.argtypes = [_prog_t, ctypes.c_char_p]
_lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(_prog_t)]


def _log(prog):
    n = ctypes.c_size_t()
    if _lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n)) != 0 or n.value <= 1:
        return ""
    buf = ctypes.create_string_buffer(n.value)
    _lib.nvrtcGetProgramLog(prog, buf)
    return buf.value.decode(errors="replace")


def compile_cuda(source: str, arch: str = "compute_75", extra_options=()):
    """Compile CUDA C source to PTX. Returns the PTX as bytes.

    arch defaults to compute_75 -- the GTX 1660 SUPER is Turing (sm_75). Pass a different arch
    for a different card; a mismatch shows up as CUDA_ERROR_NO_BINARY_FOR_GPU at load time,
    which is a clear error rather than a wrong answer.
    """
    prog = _prog_t()
    rc = _lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"lens_kernels.cu",
                                 0, None, None)
    if rc != 0:
        raise NvrtcError(f"nvrtcCreateProgram failed with code {rc}")

    opts = [f"--gpu-architecture={arch}".encode(),
            b"--std=c++14",
            b"-default-device"]
    opts += [o.encode() for o in extra_options]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = _lib.nvrtcCompileProgram(prog, len(opts), arr)
    if rc != 0:
        detail = _log(prog)
        _lib.nvrtcDestroyProgram(ctypes.byref(prog))
        raise NvrtcError(f"nvrtcCompileProgram failed (code {rc}):\n{detail}")

    n = ctypes.c_size_t()
    if _lib.nvrtcGetPTXSize(prog, ctypes.byref(n)) != 0:
        raise NvrtcError("nvrtcGetPTXSize failed")
    ptx = ctypes.create_string_buffer(n.value)
    if _lib.nvrtcGetPTX(prog, ptx) != 0:
        raise NvrtcError("nvrtcGetPTX failed")
    _lib.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value


def info():
    return {"loaded_from": _LOADED_FROM, "version": NVRTC_VERSION}
