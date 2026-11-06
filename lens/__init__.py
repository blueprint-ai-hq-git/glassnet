"""lens — a readable, fully transparent neural network library.

    import lens

    concepts = lens.concepts_from_words(train_text, [30, 31])
    net = lens.Net(vocab=40, concepts=concepts, d=64)
    net = lens.train(net, X, y, epochs=8)
    print(lens.explain(net, X[0]))      # why it answered, exactly
    lens.audit(net, X)                  # does each concept mean what its name says?

WHAT IS OURS
------------
The architecture (a middle layer of NAMED concepts, and an answer layer that sees only those
concepts), the library, the losses, the optimizer, the autograd engine, and the training loop.

WHAT IS NOT (owner's rule, 2026-10-06: from-scratch relaxed for the compute backend only)
----------------------------------------------------------------------------------------
The array arithmetic and the matmul, provided by NumPy on the CPU and CuPy on the GPU. The
owner's original brief allowed this explicitly. The hand-written C and CUDA kernels remain in
the tree and still pass their gates, but they are no longer on the tensor path -- they measured
3.1x slower than NumPy and 14.7x slower than CuPy, and the record of that is kept rather than
quietly deleted.

THE ONE IDEA
------------
A normal network's middle layer is a wall of numbers with no meaning. Here it is a list of
concepts a person can read, and the answer layer sees only those -- so every reason for every
answer is a named concept, and `explain()` can be EXACT: contributions plus bias must equal the
model's own score, and it says so with a residual.

THE ONE TRAP
------------
**100% coverage with dead concepts is possible.** The old project proved it: a model whose every
column was a named concept, with nearly all of them detecting nothing. Coverage is therefore
always reported next to `audit()`, and `audit()` is tested against a concept that cannot be
learned to prove it can actually fail.

Layout:
    lens/backend.py    where the numbers live      (NumPy or CuPy; one process-global device)
    lens/tensor.py     tensors + autograd          (L1)
    lens/model.py      the readable model          (the concept layer + the answer layer)
    lens/concepts.py   named concepts, explain()   (L3)
    lens/api.py        the library                 (L4)
    lens/kern.c        our own C kernels           (L0 -- kept, no longer on the tensor path)
    lens/cuda/         our own GPU layer           (L2 -- kept, no longer on the tensor path)
    lens/tests/        one gate per layer, each printing real numbers
"""
__version__ = "0.1.0"

from lens.api import (Net, Result, audit, concepts_from_words, explain, explain_text, load,
                      save, train)
from lens.concepts import Concept
from lens.model import LensNet
from lens import backend, tensor

__all__ = ["Net", "train", "explain", "explain_text", "audit", "save", "load",
           "concepts_from_words", "Result", "Concept", "LensNet", "backend", "tensor"]
