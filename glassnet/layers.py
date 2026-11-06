"""glassnet.layers — layer primitives, written by hand."""
import numpy as np

from .tensor import Tensor, matmul, to_backend


class Dense:
    """Fully-connected layer: y = x @ W + b. He-init weights, zero bias."""

    def __init__(self, in_features, out_features, rng=None, name=None, dtype=None):
        rng = rng if rng is not None else np.random.default_rng()
        dt = dtype if dtype is not None else np.float32  # training: float32 (project rule)
        scale = (2.0 / in_features) ** 0.5
        base = name if name else "layer"
        self.name = base
        # initialise on the host, then place on the ACTIVE backend so a GPU
        # run keeps every parameter on the GPU (to_backend is a no-op on CPU)
        self.W = Tensor(
            to_backend((rng.standard_normal((in_features, out_features)) * scale).astype(dt)),
            requires_grad=True, name=f"{base}.W")
        self.b = Tensor(
            to_backend(np.zeros(out_features, dtype=dt)),
            requires_grad=True, name=f"{base}.b")

    def __call__(self, x):
        return matmul(x, self.W) + self.b

    def parameters(self):
        return [self.W, self.b]


class Sequential:
    """A list of layers (or activation functions); call them in order."""

    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        """evaluator-compatible: returns {'logits': Tensor}."""
        return {"logits": self(x)}

    def save(self, path, **extra):
        import numpy as np
        out = {p.name: p.data for layer in self.layers
               if hasattr(layer, "parameters")
               for p in layer.parameters()}
        out.update(extra)
        np.savez(path, **out)

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def named_parameters(self):
        """(key, Tensor) pairs for checkpointing. p.name already carries the
        layer prefix (Dense names its params "<layer>.W"), so no re-prefix."""
        out = {}
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    out[p.name] = p
        return out