"""lens/cuda/driver.py — our own CUDA driver-API bindings, via ctypes.

This is the whole GPU stack: the driver API, called directly. No cupy, no torch, no cuda-python.
The only external thing used anywhere in L2 is NVRTC, and NVRTC is a COMPILER (the CUDA
equivalent of gcc), not a compute library. Writing our own PTX assembler instead would be
absurd, exactly as hand-writing x86 for the CPU kernels would be.

Handles are opaque pointers, so every function returns an int status code and we check it.
Every status code is translated to a name, because "error 1" is useless and
"CUDA_ERROR_INVALID_VALUE" is not.

Driver API notes that matter:
  * cuInit must be called before anything else, and it is safe to call repeatedly
  * a context is required before allocating; we create one per process, lazily
  * cuMemAlloc takes a SIZE_T* output and returns a device pointer as an integer
  * kernel arguments are passed as an ARRAY OF POINTERS, not as values -- this is the single
    most common mistake with cuLaunchKernel, so it is wrapped carefully below
"""
import ctypes
import os

_LIBNAMES = ["libcuda.so.1", "libcuda.so"]

_STATUS = {
    0: "CUDA_SUCCESS", 1: "CUDA_ERROR_INVALID_VALUE", 2: "CUDA_ERROR_OUT_OF_MEMORY",
    3: "CUDA_ERROR_NOT_INITIALIZED", 4: "CUDA_ERROR_DEINITIALIZED", 100: "CUDA_ERROR_NO_DEVICE",
    101: "CUDA_ERROR_INVALID_DEVICE", 201: "CUDA_ERROR_INVALID_CONTEXT",
    400: "CUDA_ERROR_INVALID_HANDLE", 700: "CUDA_ERROR_ILLEGAL_ADDRESS",
    719: "CUDA_ERROR_LAUNCH_FAILED", 999: "CUDA_ERROR_UNKNOWN",
}


class CudaError(RuntimeError):
    pass


def _load():
    last = None
    for n in _LIBNAMES:
        try:
            return ctypes.CDLL(n)
        except OSError as e:                    # pragma: no cover
            last = e
    raise CudaError(f"cannot load the CUDA driver library ({last}). Is the NVIDIA driver "
                    "installed and is libcuda.so.1 on the loader path?")


_drv = _load()


def _fn(name, argtypes):
    f = getattr(_drv, name)
    f.argtypes = argtypes
    f.restype = ctypes.c_int
    return f


_cuInit = _fn("cuInit", [ctypes.c_uint])
_cuDeviceGet = _fn("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
_cuDeviceGetName = _fn("cuDeviceGetName", [ctypes.c_char_p, ctypes.c_int, ctypes.c_int])
_cuDeviceTotalMem = _fn("cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int])
_cuCtxCreate = _fn("cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int])
_cuCtxDestroy = _fn("cuCtxDestroy_v2", [ctypes.c_void_p])
_cuMemAlloc = _fn("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
_cuMemFree = _fn("cuMemFree_v2", [ctypes.c_void_p])
_cuMemcpyHtoD = _fn("cuMemcpyHtoD_v2",
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])
_cuMemcpyDtoH = _fn("cuMemcpyDtoH_v2",
                    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t])
_cuModuleLoadData = _fn("cuModuleLoadData", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p])
_cuModuleUnload = _fn("cuModuleUnload", [ctypes.c_void_p])
_cuModuleGetFunction = _fn("cuModuleGetFunction",
                           [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p])
_cuLaunchKernel = _fn("cuLaunchKernel",
                      [ctypes.c_void_p,               # f
                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # grid x y z
                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # block x y z
                       ctypes.c_uint,                 # shared mem bytes
                       ctypes.c_void_p,               # stream
                       ctypes.POINTER(ctypes.c_void_p),  # kernelParams
                       ctypes.POINTER(ctypes.c_void_p)])  # extra
_cuCtxSynchronize = _fn("cuCtxSynchronize", [])


def check(status, what):
    if status != 0:
        raise CudaError(f"{what} failed: {_STATUS.get(status, f'code {status}')}")
    return status


class Device:
    """A CUDA device plus a context. One per process is enough for this project."""

    def __init__(self, index=0):
        check(_cuInit(0), "cuInit")
        self.handle = ctypes.c_int()
        check(_cuDeviceGet(ctypes.byref(self.handle), index), "cuDeviceGet")
        buf = ctypes.create_string_buffer(128)
        check(_cuDeviceGetName(buf, 128, self.handle), "cuDeviceGetName")
        self.name = buf.value.decode()
        total = ctypes.c_size_t()
        check(_cuDeviceTotalMem(ctypes.byref(total), self.handle), "cuDeviceTotalMem")
        self.total_mem = total.value
        self.ctx = ctypes.c_void_p()
        check(_cuCtxCreate(ctypes.byref(self.ctx), 0, self.handle), "cuCtxCreate")

    def __repr__(self):
        return (f"Device({self.name!r}, {self.total_mem / 2**20:.0f} MiB, "
                f"ctx={self.ctx.value})")

    def sync(self):
        check(_cuCtxSynchronize(), "cuCtxSynchronize")

    def destroy(self):
        if self.ctx:
            _cuCtxDestroy(self.ctx)
            self.ctx = ctypes.c_void_p()


class Buffer:
    """A device allocation. Keeps its own byte size so copies cannot overrun."""

    def __init__(self, nbytes):
        self.nbytes = int(nbytes)
        self.ptr = ctypes.c_void_p()
        check(_cuMemAlloc(ctypes.byref(self.ptr), self.nbytes), "cuMemAlloc")
        if not self.ptr.value:
            raise CudaError(f"cuMemAlloc returned a null pointer for {self.nbytes} bytes")

    def from_host(self, arr):
        if arr.nbytes > self.nbytes:
            raise ValueError(f"host array is {arr.nbytes} bytes, buffer is {self.nbytes}")
        check(_cuMemcpyHtoD(self.ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes),
              "cuMemcpyHtoD")

    def to_host(self, arr):
        if arr.nbytes > self.nbytes:
            raise ValueError(f"host array is {arr.nbytes} bytes, buffer is {self.nbytes}")
        check(_cuMemcpyDtoH(ctypes.c_void_p(arr.ctypes.data), self.ptr, arr.nbytes),
              "cuMemcpyDtoH")

    def free(self):
        if self.ptr:
            _cuMemFree(self.ptr)
            self.ptr = ctypes.c_void_p()

    def __repr__(self):
        return f"Buffer({self.nbytes} bytes @ {self.ptr.value:#x})"


def load_module(ptx_or_cubin: bytes):
    """Load PTX or cubin produced by NVRTC. Returns an opaque module handle."""
    mod = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(ptx_or_cubin)
    check(_cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)),
          "cuModuleLoadData")
    return mod


def get_function(module, name):
    fn = ctypes.c_void_p()
    check(_cuModuleGetFunction(ctypes.byref(fn), module, name.encode()), "cuModuleGetFunction")
    return fn


def launch(fn, grid, block, args, shared=0):
    """Launch a kernel.

    `args` is a list of things to pass BY POINTER -- device Buffer objects or ctypes scalars.
    cuLaunchKernel wants an array of POINTERS to the argument values, which is why this wrapper
    exists: passing values directly is the classic mistake and it segfaults or silently computes
    nonsense.
    """
    holders = []
    for a in args:
        if isinstance(a, Buffer):
            holders.append(ctypes.c_void_p(a.ptr.value))
        elif isinstance(a, ctypes._SimpleCData):
            holders.append(a)
        elif isinstance(a, int):
            holders.append(ctypes.c_long(a))
        elif hasattr(a, "dtype"):              # numpy scalar
            # WIDTH MATTERS. A kernel taking `long n` needs 8 bytes and one taking `int m`
            # needs 4; passing the wrong width reads garbage from the argument block. So the
            # numpy dtype decides, not a blanket cast.
            holders.append(ctypes.c_int(int(a)) if a.dtype.itemsize == 4
                           else ctypes.c_long(int(a)))
        else:
            raise TypeError(f"cannot pass {type(a)} as a kernel argument "
                            "(pass a Buffer, a ctypes scalar, or an int/numpy integer)")
    ptr_array = (ctypes.c_void_p * len(holders))(
        *[ctypes.cast(ctypes.pointer(h), ctypes.c_void_p) for h in holders])
    check(_cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                          shared, None, ptr_array, None), "cuLaunchKernel")
    return holders
