"""glassnet.optim — plain gradient descent with a manual learning rate.
No momentum, no weight decay, no scheduler: exactly what Phase 1 demands."""


class SGD:
    def __init__(self, parameters, lr):
        self.params = [p for p in parameters if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is None:
                raise RuntimeError(
                    f"no gradient for {p.name!r} — did you call backward()?")
            p.data = p.data - self.lr * p.grad