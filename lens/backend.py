"""lens/backend.py — where the numbers actually live.

WHY THIS EXISTS
---------------
The first GPU attempt copied every operand to the card and back, once per operation. Measured, it
was SLOWER than the CPU (0.69x on a training run, 0.5x on a big matmul) because the transfers
cost more than the arithmetic. Measured again with the arrays STAYING on the card:

    matmul 512x784x256    CPU 0.432 ms   GPU resident 0.108 ms    4.0x
    add 5M elements       CPU 4.096 ms   GPU resident 0.211 ms   19.4x

So the fix is not a faster kernel, it is keeping the arrays where the work happens. `xp` is the
array module -- NumPy or CuPy -- and everything downstream uses `xp` instead of either one
directly.

ONE GLOBAL DEVICE, AND WHAT THAT COSTS
--------------------------------------
The device is PROCESS-GLOBAL, and a model's parameters live on whichever device was active when
it was built. So:
  * building a model after `use_gpu(True)` gives you a GPU model;
  * a CPU model must be evaluated with the device set back to "cpu";
  * mixing models from two devices in one process is NOT supported.
The alternative -- a device tag on every tensor, with dispatch per operation -- is a larger
change and was not needed yet. The limit is written here so it is a known constraint rather than
a surprise, and it is exactly what broke the first version of the CPU-vs-GPU gate.

WHAT IS OURS AND WHAT IS NOT (owner's rule, stated plainly)
-----------------------------------------------------------
The owner relaxed the from-scratch rule for the COMPUTE BACKEND only. So:
  * OURS: the model architecture (the named concept layer), the library (training API, explain()),
    the losses, the optimizer, the autograd, the training loop.
  * NOT OURS: the array arithmetic and the matmul on the GPU, which CuPy provides. The owner's
    original brief allowed this explicitly ("a raw GPU array backend like CuPy").
This is written here rather than left implicit, because a claim of "from scratch" that quietly
covers borrowed GPU math would be false.

The hand-written C kernels and CUDA kernels remain in the tree, still tested by the L0/L1/L2
gates. They are kept because they pass their gates and because the record of what was tried
should not be deleted -- but the tensor ops no longer route through them.
"""
import numpy as np

xp = np                     # the array module in use
_DEVICE = "cpu"


def use_gpu(flag=True):
    """Switch the array module. Returns the module actually in use."""
    global xp, _DEVICE
    if flag:
        import cupy                     # imported here so a CPU-only machine still works
        cupy.cuda.runtime.getDeviceCount()
        xp = cupy
        _DEVICE = "gpu"
    else:
        xp = np
        _DEVICE = "cpu"
    return xp


def device():
    return _DEVICE


def is_gpu():
    return _DEVICE == "gpu"


def to_dev(a):
    """Host array -> the array module's array. No copy if it is already there."""
    if is_gpu():
        import cupy
        return a if isinstance(a, cupy.ndarray) else cupy.asarray(a)
    return a if isinstance(a, np.ndarray) else np.asarray(a)


def to_cpu(a):
    """Array -> NumPy, for printing, logging, comparing, or handing to explain().

    NEVER mix a cupy array and a numpy array in one expression -- cupy refuses, correctly, and
    the error ("Unsupported type <class 'numpy.ndarray'>") is easy to misread as a cupy bug.
    Bring the result back with .get() first.
    """
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "requires_grad"):
        # a lens Tensor: unwrap its raw array (which may be cupy). `hasattr(a, "data")` alone is NOT
        # enough — a numpy scalar also has a `.data` memoryview, and .get() on that crashes.
        data = a.data
        return data.get() if hasattr(data, "get") else data
    if hasattr(a, "get"):               # a bare cupy array
        return a.get()
    return np.asarray(a)


def sync():
    """Wait for outstanding GPU work. Needed before timing anything."""
    if is_gpu():
        import cupy
        cupy.cuda.Stream.null.synchronize()


def info():
    if is_gpu():
        import cupy
        p = cupy.cuda.runtime.getDeviceProperties(0)
        name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
        free, total = cupy.cuda.runtime.memGetInfo()
        return {"device": name, "free_mib": free / 2**20, "total_mib": total / 2**20,
                "cupy": cupy.__version__}
    return {"device": "cpu", "cupy": None}
