"""OUR OWN ARCHITECTURE vs THE REGULAR SHAPE - one file, one variable at a time.

WHY: the model so far was the inherited shape (tokens -> squeeze into concepts -> answer) with patches bolted
on. This tests the replacement described in lens/runs/ARCH-OURS.md:
  OUR WAY   : tokens + POSITION -> RECURRENT COMPUTE CORE (state carried over steps) -> named concepts ->
              answer read straight off the compute state (no funnel)
  REGULAR   : tokens + POSITION -> concepts -> answer from concepts only  (the inherited shape)
Both use the same data, same budget, same optimiser, same seed. The only difference is the compute core plus
the direct readout, exactly as promised: one change, one suspect.

Predictions were written before the run (ARCH-OURS.md):
  P1 our way beats the plain shape on held-out next-token accuracy
  P2 our way reaches a higher number than 24.10% is not required here (different budget/corpus slice);
     this run only answers "does our shape beat the regular shape at equal cost"
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import lens.tensor as T
from lens.tensor import softmax_ce

rng = np.random.default_rng(0)
W_, D, V = 32, 256, 12002
P_DIR = "data/chat/prepared_mix_ctrl_honest"

ids = np.load(os.path.join(P_DIR, "ids_v12000.npy"), mmap_mode="r")
N = len(ids) - W_ - 1
print("corpus tokens:", len(ids), flush=True)

def gather(emb, idx):
    """Embedding lookup through the ENGINE's own op, so gradients are the verified ones."""
    return T.embedding(emb, idx, 1, D)   # one position: window=1 so backward reshapes correctly

class Ours:
    """tokens + position -> recurrent compute core -> concepts -> answer from the state."""
    def __init__(self):
        s = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.emb = T.Tensor((rng.standard_normal((V, D)) * 0.02).astype(np.float32), requires_grad=True, name="emb")
        self.pos = [T.Tensor((rng.standard_normal(D) * 0.02).astype(np.float32), requires_grad=True, name=f"pos{t}") for t in range(W_)]
        self.Wx = T.Tensor(s(D, D), requires_grad=True, name="Wx")
        self.Wh = T.Tensor((rng.standard_normal((D, D)) * np.sqrt(1.0 / D)).astype(np.float32), requires_grad=True, name="Wh")
        self.b = T.Tensor(np.zeros(D, np.float32), requires_grad=True, name="b")
        self.Wc = T.Tensor(s(D, 64), requires_grad=True, name="Wc")
        self.bc = T.Tensor(np.zeros(64, np.float32), requires_grad=True, name="bc")
        self.Wa = T.Tensor(s(64, V), requires_grad=True, name="Wa")
        self.ba = T.Tensor(np.zeros(V, np.float32), requires_grad=True, name="ba")
    def params(self):
        return [self.emb] + self.pos + [self.Wx, self.Wh, self.b, self.Wc, self.bc, self.Wa, self.ba]
    def __call__(self, x):                      # x: (n, W) int
        n = x.shape[0]
        h = T.Tensor(np.zeros((n, D), np.float32))
        for t in range(W_):                                # recurrent compute core, state carried
            xt = T.add(gather(self.emb, np.ascontiguousarray(x[:, t:t+1])), self.pos[t])   # position ADDED
            h = T.relu(T.add(T.add(T.matmul(xt, self.Wx), T.matmul(h, self.Wh)), self.b))
        c = T.relu(T.add(T.matmul(h, self.Wc), self.bc))   # named concepts read the final state
        a = T.relu(T.add(T.matmul(c, self.Wa), self.ba))            # answer path through concepts
        direct = T.matmul(h, T.transpose(self.emb))                  # TIED read straight from the state
        return T.add(a, direct)

class Plain:
    """the inherited shape: tokens + position -> concepts -> answer from concepts only."""
    def __init__(self):
        s = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.emb = T.Tensor((rng.standard_normal((V, D)) * 0.02).astype(np.float32), requires_grad=True, name="emb")
        self.pos = [T.Tensor((rng.standard_normal(D) * 0.02).astype(np.float32), requires_grad=True, name=f"pos{t}") for t in range(W_)]
        self.Wc = T.Tensor(s(D, 64), requires_grad=True, name="Wc")
        self.bc = T.Tensor(np.zeros(64, np.float32), requires_grad=True, name="bc")
        self.Wo = T.Tensor(s(64, V), requires_grad=True, name="Wo")
        self.bo = T.Tensor(np.zeros(V, np.float32), requires_grad=True, name="bo")
    def params(self):
        return [self.emb] + self.pos + [self.Wc, self.bc, self.Wo, self.bo]
    def __call__(self, x):
        n = x.shape[0]
        acc = T.Tensor(np.zeros((n, D), np.float32))
        for t in range(W_):
            acc = T.add(acc, T.add(gather(self.emb, np.ascontiguousarray(x[:, t:t+1])), self.pos[t]))
        c = T.relu(T.add(T.matmul(acc, self.Wc), self.bc))
        return T.add(T.matmul(c, self.Wo), self.bo)

class SGD:
    def __init__(self, ps, lr): self.ps, self.lr = list(ps), lr
    def zero(self):
        for p in self.ps: p.grad = None
    def step(self):
        for p in self.ps:
            if p.grad is not None: p.data = p.data - self.lr * p.grad

def batch(n, offset):
    idx = np.arange(offset, offset + n) % (N - 2)
    x = np.stack([np.asarray(ids[i:i + W_]) for i in idx]).astype(np.int64)
    y = np.array([int(ids[i + W_]) for i in idx], np.int64)
    return x, y

def evaluate(model, x, y):
    z = model(x)
    return float((z.data.argmax(1) == y).mean())

x_va, y_va = batch(512, N - 2000)          # held-out slice, never trained on
res = {}
for name, make in (("REGULAR (concepts only)", Plain), ("OURS (compute core + direct)", Ours)):
    m = make(); opt = SGD(m.params(), lr=0.01); t0 = time.time()
    x_tr, y_tr = batch(256, 1000)
    for step in range(150):
        xb, yb = batch(256, 1000 + step * 256)
        opt.zero()
        loss = softmax_ce(m(xb), yb)
        loss.backward(); opt.step()
        if step % 50 == 49:
            print(json.dumps({"model": name, "step": step + 1, "loss": round(float(loss.data), 4),
                              "train_acc": round(evaluate(m, xb, yb), 4),
                              "HELD_OUT": round(evaluate(m, x_va, y_va), 4),
                              "sec": round(time.time() - t0, 1)}), flush=True)
    res[name] = evaluate(m, x_va, y_va)
print(json.dumps({"FINAL": res}), flush=True)
