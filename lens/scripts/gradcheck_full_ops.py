"""gradcheck_full_ops.py — numeric gradient check for EVERY op in the engine, sweeping several input
ranges per op so a formula that is right only at one magic value cannot hide.

LESSON FROM 2026-10-10 (the w=1 bug): the old check built its concepts with support=0.5, so
pos_weight = (1-0.5)/0.5 = 1, and the WRONG backward formula `w*(h-t)/(h(1-h))` is EXACTLY right when
w == 1. The check certified a wrong formula for months while every concept with a real support
(0.02 -> w=49, 0.9 -> w=0.111) trained against a mis-scaled gradient. A gradient check that tests a
single configuration can certify a wrong formula, so each op here is tested at several inputs.

OPS ARE MODULE FUNCTIONS (T.add, T.mul, ...), not operator overloads - the engine deliberately has no
__add__ so that every op in the graph is explicit in the source.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.tensor as T                                   # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.vision import ImageConcept, VisionNet            # noqa: E402

T.set_device("cpu")
T.set_dtype("float64")
rng = np.random.default_rng(7)
EPS = 1e-6
results = {}


def numeric_grad(fn, x, eps=EPS):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        o = x[i]
        x[i] = o + eps
        lp = fn()
        x[i] = o - eps
        lm = fn()
        x[i] = o
        g[i] = (lp - lm) / (2 * eps)
    return g


def check(name, graph_fn, shapes, n_sets=3, scale=0.7):
    """graph_fn(list_of_tensors) -> scalar Tensor. Numeric via re-running graph_fn on raw arrays."""
    worst_abs = worst_rel = 0.0
    for _ in range(n_sets):
        arrs = [rng.standard_normal(sh).astype(np.float64) * scale for sh in shapes]
        ts = [T.Tensor(a, requires_grad=True) for a in arrs]
        T.sum_all(graph_fn(ts)).backward()
        ans = [np.asarray(t.grad, dtype=np.float64).copy() for t in ts]
        for k, a in enumerate(arrs):
            num = numeric_grad(
                lambda: float(T.sum_all(graph_fn([T.Tensor(x, requires_grad=True) for x in arrs])).data), a)
            d = np.abs(ans[k] - num)
            worst_abs = max(worst_abs, d.max())
            worst_rel = max(worst_rel, (d / (np.abs(ans[k]) + np.abs(num) + 1e-12)).max())
    ok = worst_abs < 1e-8
    results[name] = ok
    print(f"  {name:<32} worst abs {worst_abs:9.2e}  worst rel {worst_rel:9.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


print("=" * 96)
print("FULL OP-TABLE GRADIENT AUDIT — float64, several input ranges per op")
print("=" * 96)


def _attn_graph(ts):
    """The full causal-attention stack as a graph on our own ops: embed-like input (N, W*D),
    q/k/v projections, batched scores, causal mask, row softmax, weighted values, output
    projection, flattened back to the window shape."""
    from lens.model import CausalSelfAttention
    rng = np.random.default_rng(3)
    attn = CausalSelfAttention(d=4, window=3, rng=rng)
    # build the graph from the input tensor through the attention (params get their own analytic path)
    x = ts[0]
    N, WD = x.data.shape
    W, D = 3, 4
    xr = T.reshape(x, (N * W, D))
    q = T.reshape(T.matmul(xr, attn.Wq), (N, W, D))
    k = T.reshape(T.matmul(xr, attn.Wk), (N, W, D))
    v = T.reshape(T.matmul(xr, attn.Wv), (N, W, D))
    s = T.bmm3(q, k, trans_b=True)
    mask3 = T.Tensor(B.xp.broadcast_to(attn.mask_dev, (N, 3, 3)), requires_grad=False)
    s = T.add(s, mask3)
    p = T.softmax_rows(s)
    o = T.bmm3(p, v)
    o = T.reshape(o, (N * W, D))
    o = T.matmul(o, attn.Wo)
    return T.reshape(o, (N, WD))

# ---- elementwise / shape ops ----
check("add same shape", lambda t: T.add(t[0], t[1]), [(5, 4), (5, 4)])
check("add bias broadcast", lambda t: T.add(t[0], t[1]), [(5, 4), (4,)])
check("sub", lambda t: T.sub(t[0], t[1]), [(5, 4), (5, 4)])
check("mul same shape", lambda t: T.mul(t[0], t[1]), [(5, 4), (5, 4)])
check("matmul", lambda t: T.matmul(t[0], t[1]), [(5, 4), (4, 6)])
check("bmm3 plain", lambda t: T.bmm3(t[0], t[1]), [(3, 5, 4), (3, 4, 6)])
check("bmm3 trans", lambda t: T.bmm3(t[0], t[1], trans_b=True), [(3, 5, 4), (3, 6, 4)])
check("softmax_rows", lambda t: T.softmax_rows(t[0]), [(3, 5, 5)])
check("attn-chain", lambda t: _attn_graph(t), [(4, 12)])

# ---- full-net check: EVERY parameter of a small attention LensNet (the w=1 lesson: the check
# must cover what the optimizer actually updates, not just the op inputs) ----
print("\nfull attention-net parameter audit (numeric vs analytic on ALL params):")
from lens.model import LensNet                        # noqa: E402
from lens.concepts import Concept                      # noqa: E402

cpts = [Concept("c0", ids=[1, 2], support=0.4), Concept("c1", ids=[3], support=0.1)]
rng2 = np.random.default_rng(11)
net = LensNet(vocab_size=9, concepts=cpts, d=4, window=3, seed=11, lam_concept=0.3,
              lam_act=0.1, lam_w=0.0, attention=True)
ids_np = rng2.integers(0, 9, size=(2, 3))
y_np = rng2.integers(0, 9, size=(2,))
params = net.parameters()
# analytic grads
loss, _ = net.loss(ids_np, y_np)
loss.backward()
want = {id(p): np.asarray(p.grad, dtype=np.float64).copy() for p in params if p.grad is not None}
# numeric: perturb each param element, re-run loss
def loss_val():
    lo, _ = net.loss(ids_np, y_np)
    return float(lo.data)
worset = (0.0, None)
eps = 1e-6
for p in params:
    if p.grad is None:
        continue
    g0 = np.zeros_like(p.data, dtype=np.float64)
    for idx in np.ndindex(p.data.shape):
        orig = p.data[idx]
        p.data[idx] = orig + eps; lp = loss_val()
        p.data[idx] = orig - eps; lm = loss_val()
        p.data[idx] = orig
        g0[idx] = (lp - lm) / (2 * eps)
    d = np.abs(want[id(p)] - g0)
    wa = d.max()
    wr = (d / (np.abs(want[id(p)]) + np.abs(g0) + 1e-12)).max()
    # gate on ABSOLUTE error like every other check in this file (rel explodes on near-zero
    # gradient elements, e.g. saturated sigmoid paths — that is a metric artifact, not a bug)
    if wr > worset[0]:
        worset = (wr, p.name)
    print(f"  {p.name:<14} worst abs {wa:9.2e}  worst rel {wr:9.2e}  "
          f"{'PASS' if wa < 1e-8 else 'FAIL'}")
    results["attn-net:" + p.name] = wa < 1e-8
print(f"  NET worst rel {worset[0]:.2e} on {worset[1]} (rel is informational only; the ABS gate is the verdict)")
results["attn-net-all"] = all(results.get(f"attn-net:{p.name}", False) for p in params)
check("reshape", lambda t: T.reshape(t[0], (2, 12)), [(4, 6)])
check("concat", lambda t: T.concat([t[0], t[1]], axis=1), [(5, 3), (5, 4)])
check("relu", lambda t: T.relu(t[0]), [(5, 6)])
check("sigmoid", lambda t: T.sigmoid(t[0]), [(5, 6)])

# ---- fused softmax + CE, plain and weighted ----
lab = np.asarray([0, 1, 2, 3], dtype=np.int64)
check("softmax_ce", lambda t: T.softmax_ce(t[0], lab), [(4, 5)])
check("softmax_ce + class_weights",
      lambda t: T.softmax_ce(t[0], lab, class_weights=np.asarray([1.0, 2.0, 0.5, 3.0])), [(4, 5)])
check("softmax_ce + sample_weights",
      lambda t: T.softmax_ce(t[0], lab, sample_weights=np.asarray([1.0, 0.3, 2.0, 0.7])), [(4, 5)])

# ---- embedding ----
def emb_graph(t):
    # embedding(table, ids, window, d): gather rows then flatten the window
    ids = np.asarray([[1, 3, 0, 5]], dtype=np.int64)
    E = T.embedding(t[0], ids, 4, 4)
    return T.reshape(E, (1, 4 * 4))

check("embedding (gather)", emb_graph, [(6, 4)])

print()
print("  ---- concept-loss gradient vs the positive weight (the discovered bug) ----")
for sup in (0.02, 0.2, 0.5, 0.9):
    w = (1.0 - sup) / sup

    def _c(nm, fn, s):
        return ImageConcept(nm, fn, support=s, formula="test")

    imgs = rng.random((6, 784))
    yy = np.arange(6)
    cv = [_c("a", lambda X: (np.asarray(X).mean(axis=1) > 0.2).astype(np.float64), sup),
          _c("b", lambda X: (np.asarray(X).mean(axis=1) > 0.4).astype(np.float64), sup),
          _c("c", lambda X: (np.asarray(X).std(axis=1) > 0.2).astype(np.float64), sup)]
    net = VisionNet(cv, n_classes=6, d=6, patch=7, seed=0, lam_concept=0.3)
    for p in net.parameters():
        p.data = np.asarray(B.to_cpu(p.data), dtype=np.float64)
    net.loss(imgs, yy)[0].backward()

    def nloss():
        return float(net.loss(imgs, yy)[0].data)

    worst = 0.0
    for p in net.parameters():
        base = np.asarray(B.to_cpu(p.data), dtype=np.float64).copy()
        an = np.asarray(B.to_cpu(p.grad), dtype=np.float64).copy().reshape(-1)
        flat = base.reshape(-1)
        sel = np.linspace(0, flat.size - 1, min(15, flat.size)).astype(int)
        for j in sel:
            o = flat[j]
            flat[j] = o + EPS
            p.data = base.copy()
            lp = nloss()
            flat[j] = o - EPS
            p.data = base.copy()
            lm = nloss()
            flat[j] = o
            worst = max(worst, abs(an[j] - (lp - lm) / (2 * EPS)))
        p.data = base
        break                      # one param tensor is enough to detect a w-scaling error
    ok = worst < 1e-8
    results[f"concept_loss w={w:.4g}"] = ok
    print(f"  concept_loss @ support {sup:<5} (w={w:<7.4g})  worst abs {worst:9.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

print()
fails = [k for k, v in results.items() if not v]
print(f"  {len(results)} op/configurations checked: {len(results)-len(fails)} PASS, {len(fails)} FAIL")
for f in fails:
    print(f"    FAIL: {f}")
print(f"\n  OVERALL: {'ALL OPS GRADIENT-CORRECT' if not fails else 'BUGS REMAIN'}")
sys.exit(0 if not fails else 2)
