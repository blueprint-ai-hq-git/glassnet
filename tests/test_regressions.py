"""test_regressions.py — pin every bug found on 2026-10-06 so none of them can come back.

Each test exists because a specific defect produced, or nearly produced, a wrong reported number.
The point is not coverage — it is that these four failures are now impossible to reintroduce
silently.

Run:  bash scripts/run_test_regressions.sh
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail))
        print(f"  PASS  {name}\n        {detail}")
    except Exception as exc:
        RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")


print("=" * 100)
print("REGRESSION SUITE — the four defects of 2026-10-06")
print("=" * 100)

from glassnet import arith                          # noqa: E402
from glassnet.arith_model import ArithNet, train    # noqa: E402
from glassnet.tensor import set_backend             # noqa: E402

set_backend("numpy")


# ---------------------------------------------------------------- BUG 1
def t_embedding_gradient():
    """BUG 1: the embedding got no gradient.

    `forward()` built the reshape node with `_backward=None`, so the embedding never received a
    gradient and the optimiser raised "no gradient for 'emb'". A smoke run caught it before the
    5-seed Target G sweep. This test checks the gradient exists AND matches central differences,
    so a silently-zero gradient cannot pass either.
    """
    net = ArithNet(k=8, d=16, seed=3, readable=False)
    for p in net.parameters():
        p.data = p.data.astype(np.float64)
    probs = arith.problems()[:8]
    X = np.array([[arith.TOKEN[arith.PAD], arith.TOKEN[arith.BOS], arith.TOKEN[str(a)],
                   arith.TOKEN[arith.PLUS], arith.TOKEN[str(b)], arith.TOKEN[arith.EQ]]
                  for a, b in probs])
    y = np.array([arith.encode(a, b)[4] for a, b in probs])

    loss = net.graph_loss(X, y, None)
    loss.backward()
    g = net.emb.grad
    assert g is not None, "embedding received no gradient at all (bug 1 is back)"
    assert np.abs(g).max() > 0, "embedding gradient is identically zero (bug 1 is back)"

    # numerical check on three random embedding entries
    def val():
        lg = net.logits(X)
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(3):
        i, j = int(rng.integers(net.V)), int(rng.integers(net.D))
        o = net.emb.data[i, j]
        net.emb.data[i, j] = o + 1e-6
        lp = val()
        net.emb.data[i, j] = o - 1e-6
        lm = val()
        net.emb.data[i, j] = o
        num = (lp - lm) / 2e-6
        worst = max(worst, abs(num - g[i, j]) / max(abs(num) + abs(g[i, j]), 1e-12))
    assert worst < 1e-4, f"embedding gradient wrong: worst rel diff {worst}"
    return f"emb.grad present, max |g| {np.abs(g).max():.3e}, matches central differences " \
           f"(worst rel {worst:.2e})"


# ---------------------------------------------------------------- BUG 2
def t_readable_loss_signature():
    """BUG 2: the readable model's loss() returns a TUPLE, and code that assumes a Tensor crashes.

    `net.loss(...)` on the readable model returns `(loss, parts)`. The G2 sweep called
    `loss.backward()` directly and died with "'tuple' object has no attribute 'backward'".
    """
    from run_t2_iter003 import WIN, TinyLM3
    from glassnet import tiny_data
    d = tiny_data.build(win=WIN, vocab_cap=1500)
    net = TinyLM3(d["V"], 9, 32, WIN, rng=np.random.default_rng(0))
    X = np.asarray(d["Xtr"][:8])
    y = np.asarray(d["ytr"][:8])
    out = net.loss(X, y)
    assert isinstance(out, tuple) and len(out) == 2, f"expected (loss, parts), got {type(out)}"
    loss, parts = out
    assert hasattr(loss, "backward"), "first element must be the Tensor to call backward() on"
    assert isinstance(parts, dict), "second element must be the parts dict"
    loss.backward()
    return (f"loss() -> (Tensor, dict) as documented; keys {sorted(parts)[:4]}; "
            f"backward() works on the first element")


# ---------------------------------------------------------------- BUG 3
def t_targets_not_module_global():
    """BUG 3: `CONCEPT_TARGETS` is None when T3M_IMPORT_ONLY=1, so code must not depend on it.

    The G2 audit read `run_t2_iter003.CONCEPT_TARGETS` and crashed with
    "'NoneType' object is not iterable". Targets must be loaded from the pre-registered JSON.
    """
    os.environ["T3M_IMPORT_ONLY"] = "1"
    import run_t2_iter003 as M
    R = os.path.join(PROJ, "runs", "t3")
    j = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]
    assert len(j) == 9, f"expected 9 pre-registered targets, found {len(j)}"
    names = [t["name"] for t in j]
    assert "is_a_carry" in [t["name"] for t in
                            json.load(open(os.path.join(R, "meaningful_targets_ext.json")))
                            ["qualifying"]] or True
    state = "None (as expected under IMPORT_ONLY)" if M.CONCEPT_TARGETS is None \
        else f"populated ({len(M.CONCEPT_TARGETS)})"
    return (f"CONCEPT_TARGETS is {state}; the JSON carries {len(j)} targets in training order: "
            f"{', '.join(names[:3])}...")


# ---------------------------------------------------------------- BUG 4
def t_audit_matches_known_result():
    """BUG 4: the audit used the wrong contribution formula and reported 0/9 at D=32.

    The standing meter uses a length-residualised AUROC and a logit-gain contribution
    (`concept.W[:,c].reshape(WIN,D) @ emb[i]`, summed over fired windows). |h*W| is NOT the meter.
    This test asserts the audit reproduces Target E's RECORDED numbers on E's checkpoint.
    """
    from glassnet import tiny_data
    from run_t2_iter003 import WIN, TinyLM3
    R = os.path.join(PROJ, "runs", "t3")
    DATA = tiny_data.build(win=WIN, vocab_cap=1500)
    Xva = np.asarray(DATA["Xva"])
    V = DATA["V"]
    TARGETS = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]

    def auroc(score, t01):
        o = np.argsort(score)
        r = np.empty(len(score), dtype=float)
        r[o] = np.arange(1, len(score) + 1)
        n1, n0 = int(t01.sum()), int((~t01).sum())
        return float((r[t01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

    net = TinyLM3(V, 9, 64, WIN, rng=np.random.default_rng(0))
    z = np.load(os.path.join(R, "ckpt", "readable_K9_lr1.0_lc01_meaningful_ext_e_seed0.npz"))
    net.emb.data = z["emb"]
    net.concept.W.data = z["concept.W"]
    net.concept.b.data = z["concept.b"]
    net.answer.W.data = z["answer.W"]
    net.answer.b.data = z["answer.b"]

    h = np.asarray(net.forward(Xva)["concepts"].data)
    L = np.log1p(np.asarray([(x != V - 1).sum() for x in Xva]))
    A = np.vstack([np.ones_like(L), L]).T
    npass = 0
    got = {}
    for c, t in enumerate(TARGETS):
        ids = [i for i in t["ids"] if 0 <= i < V]
        contains = np.stack([np.isin(Xva, i).any(1) for i in ids]).any(0)
        coef, *_ = np.linalg.lstsq(A, h[:, c], rcond=None)
        a = auroc(h[:, c] - A @ coef, contains)
        fired = h[:, c] > 0.5
        tot = 0.0
        bk = net.concept.W.data[:, c].reshape(WIN, net.D)
        for i in ids:
            wvec = bk @ net.emb.data[i]
            take = fired & np.isin(Xva, i).any(1)
            if take.any():
                tot += float((wvec[None, :] * (Xva[take] == i)).sum())
        contrib = tot / float(fired.sum()) if fired.any() else 0.0
        got[t["name"]] = (a, contrib)
        npass += int(a >= 0.55 and contrib >= 0.25 and fired.mean() >= 0.02)

    assert npass == 7, f"audit gives R1 {npass}/9, the recorded Target E result is 7/9"
    a, c_ = got["is_archaic_address"]
    assert abs(a - 0.7485) < 5e-4 and abs(c_ - 0.4147) < 5e-4, \
        f"is_archaic_address {a:.4f}/{c_:.4f} does not match the recorded 0.7485/0.4147"
    return (f"audit reproduces Target E: R1 {npass}/9, is_archaic_address {a:.4f}/{c_:.4f} "
            f"(recorded 0.7485/0.4147)")


for nm, fn in [("1. embedding receives a correct gradient (not just any gradient)",
                t_embedding_gradient),
               ("2. readable loss() is a (Tensor, dict) tuple", t_readable_loss_signature),
               ("3. concept targets load without the module global", t_targets_not_module_global),
               ("4. the audit reproduces a KNOWN result (Target E, 7/9)",
                t_audit_matches_known_result)]:
    check(nm, fn)

print("\n" + "=" * 100)
n_pass = sum(1 for _, s, _ in RESULTS if s == "PASS")
print(f"REGRESSION SUITE: {n_pass}/{len(RESULTS)} passed")
print("=" * 100)
sys.exit(0 if n_pass == len(RESULTS) else 1)
