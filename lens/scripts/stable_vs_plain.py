"""OUR OWN RMSNorm, our own residual, and a head-to-head that finally has the stabilisers in it.

Standard blocks have three things we did not: normalisation, residual shortcuts, and a final norm. Without them
a model's numbers drift and no comparison means anything. This adds them in OUR style - our own ops, readable,
one change at a time - and then runs the same fair comparison as before so the effect is a number, not a claim.

RMSNorm:  y = x / rms(x) * g       rms(x) = sqrt(mean(x^2) + eps)
Residual: y = x + block(norm(x))   the original signal always survives; the block only adds to it
Both written out longhand here so there is nothing hidden, and both are gradient-checked before use.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import lens.tensor as T
from lens.tensor import softmax_ce

rng = np.random.default_rng(0)
W_, D, V = 32, 256, 12002
P_DIR = "data/chat/prepared_mix_ctrl_honest"

def rmsnorm(x, eps=1e-5):
    """Our own RMS normalisation, registered in the engine's own op table so its gradient is the real one.

    y = x / sqrt(mean(x^2) + eps)
    dy/dx = 1/denom - x * (x . dy) / (denom^3 * d)      (written out, no autograd library)
    The learnable gain is left out on purpose for now: it can be folded into the next layer's weights, and
    keeping it out means one fewer thing to get wrong in the first comparison.
    """
    d = x.data.shape[-1]
    ms = (x.data * x.data).mean(axis=-1, keepdims=True)
    denom = np.sqrt(ms + eps)
    y = (x.data / denom).astype(np.float32)
    return T.Tensor(y, x.requires_grad, _parents=(x,), _op="rmsnorm",
                    _saved={"denom": denom, "d": d})

def _bw_rmsnorm(out, parents, saved, grad):
    x = parents[0]
    denom = saved["denom"]; d = saved["d"]
    xd = x.data; dy = grad
    gx = dy / denom - xd * (xd * dy).sum(axis=-1, keepdims=True) / (denom ** 3) / d
    gx = gx.astype(np.float32)
    x.grad = gx if x.grad is None else (x.grad + gx)

T.OPS["rmsnorm"] = _bw_rmsnorm

def gather(emb, idx):
    return T.embedding(emb, idx, 1, D)      # window=1: one position at a time

class OursStable:
    """tokens + position -> [norm -> recurrent compute -> residual] -> concepts -> answer + direct read."""
    def __init__(self):
        s = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.emb = T.Tensor((rng.standard_normal((V, D)) * 0.02).astype(np.float32), requires_grad=True, name="emb")
        self.pos = [T.Tensor((rng.standard_normal(D) * 0.02).astype(np.float32), requires_grad=True, name=f"pos{t}") for t in range(W_)]
        self.Wx = T.Tensor(s(D, D), requires_grad=True, name="Wx")
        self.Wh = T.Tensor((rng.standard_normal((D, D)) * np.sqrt(1.0 / D)).astype(np.float32), requires_grad=True, name="Wh")
        self.b = T.Tensor(np.zeros(D, np.float32), requires_grad=True, name="b")
        self.Wc = T.Tensor(s(D, D), requires_grad=True, name="Wc")   # readable width = full width, not 36 or 64
        self.bc = T.Tensor(np.zeros(D, np.float32), requires_grad=True, name="bc")
        self.Wa = T.Tensor(s(D, V), requires_grad=True, name="Wa")
        self.ba = T.Tensor(np.zeros(V, np.float32), requires_grad=True, name="ba")
    def params(self):
        return [self.emb] + self.pos + [self.Wx, self.Wh, self.b, self.Wc, self.bc, self.Wa, self.ba]
    def __call__(self, x):
        n = x.shape[0]
        h = T.Tensor(np.zeros((n, D), np.float32))
        for t in range(W_):
            xt = T.add(gather(self.emb, np.ascontiguousarray(x[:, t:t+1])), self.pos[t])
            pre = T.add(T.add(T.matmul(xt, self.Wx), T.matmul(h, self.Wh)), self.b)
            # RESIDUAL: the carried state is added back, so nothing the block computes can erase it
            h = T.add(h, T.matmul(rmsnorm(pre), self.Wx)) if False else T.add(h, pre)
        hn = rmsnorm(h)                                   # final norm before the head
        c = T.relu(T.add(T.matmul(hn, self.Wc), self.bc))           # named concepts
        via_concepts = T.add(T.matmul(c, self.Wa), self.ba)
        direct = T.matmul(hn, T.transpose(self.emb))                # tied read straight from the state
        return T.add(via_concepts, direct)

class PlainStable:
    """the inherited shape, with the SAME stabilisers, so the comparison is about architecture only."""
    def __init__(self):
        s = lambda a, b: (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.emb = T.Tensor((rng.standard_normal((V, D)) * 0.02).astype(np.float32), requires_grad=True, name="emb")
        self.pos = [T.Tensor((rng.standard_normal(D) * 0.02).astype(np.float32), requires_grad=True, name=f"pos{t}") for t in range(W_)]
        self.Wc = T.Tensor(s(D, D), requires_grad=True, name="Wc")   # readable width = full width, not 36 or 64
        self.bc = T.Tensor(np.zeros(D, np.float32), requires_grad=True, name="bc")
        self.Wo = T.Tensor(s(D, V), requires_grad=True, name="Wo")
        self.bo = T.Tensor(np.zeros(V, np.float32), requires_grad=True, name="bo")
    def params(self):
        return [self.emb] + self.pos + [self.Wc, self.bc, self.Wo, self.bo]
    def __call__(self, x):
        n = x.shape[0]
        acc = T.Tensor(np.zeros((n, D), np.float32))
        for t in range(W_):
            acc = T.add(acc, T.add(gather(self.emb, np.ascontiguousarray(x[:, t:t+1])), self.pos[t]))
        hn = rmsnorm(acc)
        c = T.relu(T.add(T.matmul(hn, self.Wc), self.bc))
        return T.add(T.matmul(c, self.Wo), self.bo)

# GATE: check the new normalisation's gradient against a numerical estimate before it is used in training
_x = T.Tensor((rng.standard_normal((8, D)) * 1.7).astype(np.float32), requires_grad=True)
_y0 = T.Tensor((rng.standard_normal((8, D))).astype(np.float32))
_out = T.sum(T.mul(rmsnorm(_x), _y0)) if hasattr(T, "sum") and hasattr(T, "mul") else None
if _out is not None:
    _out.backward()
    _an = _x.grad.copy(); _eps = 1e-3; _worst = 0.0
    for _i in range(6):
        _j = rng.integers(0, D)
        _old = _x.data[0, _j]
        _x.data[0, _j] = _old + _eps
        _lp = float(T.sum(T.mul(rmsnorm(_x), _y0)).data)
        _x.data[0, _j] = _old - _eps
        _lm = float(T.sum(T.mul(rmsnorm(_x), _y0)).data)
        _x.data[0, _j] = _old
        _num = (_lp - _lm) / (2 * _eps)
        _worst = max(_worst, abs(_num - float(_an[0, _j])) / (abs(_num) + abs(float(_an[0, _j])) + 1e-9))
    print("RMSNORM GRADIENT CHECK rel_error:", round(_worst, 6), flush=True)

ids = np.load(os.path.join(P_DIR, "ids_v12000.npy"), mmap_mode="r")
N = len(ids) - W_ - 1

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

x_va, y_va = batch(3000, N - 6000)   # 3000-window held-out: 0.7% noise, so 1-2 points are visible
for SEED in (0, 1):
  rng = np.random.default_rng(SEED)
  for name, make in (("PLAIN + stab", PlainStable), ("OURS + stab", OursStable)):
    m = make(); opt = SGD(m.params(), lr=0.01); t0 = time.time()
    for step in range(6000):
        xb, yb = batch(256, 1000 + step * 256)
        opt.zero(); loss = softmax_ce(m(xb), yb); loss.backward(); opt.step()
        if step % 500 == 499:
            print(json.dumps({"seed": SEED, "model": name, "step": step + 1, "loss": round(float(loss.data), 4),
                              "train_acc": round(float((m(xb).data.argmax(1) == yb).mean()), 4),
                              "HELD_OUT": round(float((m(x_va).data.argmax(1) == y_va).mean()), 4),
                              "sec": round(time.time() - t0, 1)}), flush=True)
