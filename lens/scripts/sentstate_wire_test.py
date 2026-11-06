#!/usr/bin/env python3
"""lens/scripts/sentstate_wire_test.py -- STRUCTURAL test of the sentstate graft. No corpus, no GPU, no epoch.

Predictions: lens/runs/SENTSTATE-WIRE-PREDICTIONS-P114.txt (written before this file existed).

What it proves, and nothing more:
  P114a-ish  the base parameters are IDENTICAL with the graft off and on (emb byte-equal, same seed), i.e.
             the graft does not disturb the original initialisation or its RNG draw order;
  P114b      the parameter delta equals D*K + K*ds + ds*ds + ds + ds*V + V to the digit;
  P114c      every new tensor carries a NONZERO gradient, and a BASE parameter's gradient differs from its
             flag-off value on the same batch (the lever is not detached and not inert);
  P114d      the first-step loss differs from flag-off on the same batch and seed;
  P114f      answer.W keeps shape (K, V) and sread.W is (ds, V).

It does NOT prove the lever helps, does not train anything, and makes no quality claim.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lens import backend                                    # noqa: E402
from lens import tensor as T                                # noqa: E402
from lens.concepts import Concept                           # noqa: E402
from lens.model import LensNet                              # noqa: E402
import lens.sentstate_wire as W                             # noqa: E402

V, D, WIN, K, DS, N = 64, 64, 32, 4, 64, 8
FAILS = []


def chk(name, cond, detail):
    print(f"  {name}: {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILS.append(name)


def build(enable):
    concepts = [Concept(f"c{k}", [1 + 3 * k, 2 + 3 * k]) for k in range(K)]
    W.disable()
    if enable:
        W.enable(ds=DS)
    net = LensNet(V, concepts, d=D, window=WIN, seed=0, lam_concept=0.3)
    W.disable()
    return net


def main():
    T.set_device("cpu")
    T.set_dtype(np.float32)
    installed = W.install(ds=DS)
    print("=" * 100)
    print("SENTSTATE GRAFT -- STRUCTURAL TEST (CPU, d=64, window=32, V=64, K=4, N=8 batch)")
    print(f"  install() returned {installed} (True = freshly patched; False = already patched)")
    print("=" * 100)

    off = build(False)
    on = build(True)

    # ---- P114a-ish: base parameters untouched by the graft
    emb_off = np.asarray(backend.to_cpu(off.emb.data))
    emb_on = np.asarray(backend.to_cpu(on.emb.data))
    chk("BASE-INIT-UNTOUCHED", np.array_equal(emb_off, emb_on),
        f"emb identical to the byte: {np.array_equal(emb_off, emb_on)} (same seed 0, graft draws from its own stream)")
    n_off = sum(p.data.size for p in off.parameters())
    n_on = sum(p.data.size for p in on.parameters())

    # ---- P114b: exact parameter delta
    _, expect = W.ship_config_cost(D=D, K=K, ds=DS, V=V)
    parts, _ = W.ship_config_cost(D=D, K=K, ds=DS, V=V)
    delta = n_on - n_off
    chk("P114b PARAM-DELTA", delta == expect,
        f"measured +{delta}, formula +{expect}  {parts}")
    _, ship = W.ship_config_cost()
    chk("P114b SHIP-ARITHMETIC", ship == 795810,
        f"ship config (D=256,K=36,ds=64,V=12002) = +{ship} on 5,908,686 = "
        f"+{100.0 * ship / 5908686:.2f}%")

    # ---- P114f: shapes
    chk("P114f ANSWER-SHAPE", tuple(on.answer._inner.W.data.shape) == (K, V),
        f"answer.W {tuple(on.answer._inner.W.data.shape)} (unchanged, K x V)")
    chk("P114f SREAD-SHAPE", tuple(on.sread.W.data.shape) == (DS, V),
        f"sread.W {tuple(on.sread.W.data.shape)} (ds x V)")

    # ---- one batch, both ways
    rng = np.random.default_rng(3)
    X = rng.integers(0, V, size=(N, WIN)).astype(np.int64)
    Y = rng.integers(0, V, size=(N,)).astype(np.int64)

    def loss_and_grads(net):
        for p in net.parameters():
            p.zero_grad()
        total, parts = net.loss(X, Y)
        total.backward()
        return float(backend.to_cpu(total.data)), parts, net

    L_off, _, _ = loss_and_grads(off)
    L_on, parts_on, net_on = loss_and_grads(on)

    # ---- P114d: the state reaches the logits
    chk("P114d LOSS-DIFFERS", abs(L_on - L_off) > 0.0,
        f"loss off {L_off:.6f} vs on {L_on:.6f}  delta {L_on - L_off:+.6f}")

    # ---- P114c: gradients
    new_t = {"Wc2": net_on.pconcept.Wc2, "Wc": net_on.state.Wc, "Ur": net_on.state.Ur,
             "b": net_on.state.b, "sread.W": net_on.sread.W, "sread.b": net_on.sread.b}
    for k, t in new_t.items():
        g = None if t.grad is None else float(np.abs(np.asarray(backend.to_cpu(t.grad))).sum())
        chk(f"P114c GRAD {k}", g is not None and g > 0.0, f"sum|grad| = {g}")
    # a BASE parameter's gradient must move too (else the state is a detached add-on)
    g_off = float(np.abs(np.asarray(backend.to_cpu(off.concept.W.grad))).sum())
    g_on = float(np.abs(np.asarray(backend.to_cpu(net_on.concept._inner.W.grad))).sum())
    chk("P114c BASE-GRAD-MOVES", abs(g_on - g_off) > 1e-9,
        f"concept.W sum|grad| off {g_off:.6f} vs on {g_on:.6f}")
    # ... AND the state's own path must not be dead for the answer layer either
    ga = float(np.abs(np.asarray(backend.to_cpu(net_on.answer._inner.W.grad))).sum())
    chk("P114c ANSWER-GRAD", ga > 0.0, f"answer.W sum|grad| = {ga:.6f}")

    # ---- contractivity sanity of the carry as wired (the P111e lesson, re-checked at the graft's init)
    st = net_on.state
    s = np.full((N, DS), 0.9, dtype=np.float32)
    hp = np.zeros((N, WIN, K), dtype=np.float32)
    seq = [float(np.abs(np.asarray(backend.to_cpu(s))).max())]
    for _ in range(8):
        s_t = W.sstate_scan(T.Tensor(backend.to_dev(hp), requires_grad=False), st.Wc, st.Ur, st.b,
                            s_init=T.Tensor(backend.to_dev(s), requires_grad=False))
        s = np.asarray(backend.to_cpu(s_t.data))
        seq.append(float(np.abs(s).max()))
    chk("CARRY-CONTRACTIVE", seq[-1] < seq[0],
        "zero-input carry, 8 windows: " + " -> ".join(f"{v:.4f}" for v in seq))

    print("=" * 100)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL ' + ', '.join(FAILS)}")

    # ---- SAVE / RELOAD (the silent-wrong-model guard, added 18:0x after reading api.load's loop)
    import tempfile
    from lens import backend as B
    from lens.api import save as api_save, load as api_load
    tmp = os.path.join(tempfile.mkdtemp(prefix="sswire_"), "grafted.npz")
    on.device_name = "cpu"      # api.Net() sets this; a bare LensNet does not (api.save reads it)
    api_save(on, tmp)
    z = np.load(tmp, allow_pickle=False)
    keys = set(z.files) - {"__meta__"}
    print("\n  SAVE/RELOAD: the grafted checkpoint carries", len(keys), "tensors; graft tensors present:",
          sorted(k for k in W.GRAFT_PARAMS if k in keys))
    # 1) the un-grafted loader IGNORES the graft tensors silently -> that is why load_grafted exists
    plain = api_load(tmp, device="cpu")
    plain_names = {p.name for p in plain.parameters()}
    ignored = sorted(keys - plain_names)
    chk("SILENT-IGNORE-CONFIRMED", len(ignored) == len(W.GRAFT_PARAMS),
        f"api.load() rebuilt a model WITHOUT {ignored} -- no error, a different model (the APILOAD-DIRECT class)")
    # 2) the guarded loader reproduces the model bit-exactly
    g = W.load_grafted(tmp, device="cpu")
    Xa = rng.integers(0, V, size=(3, WIN)).astype(np.int64)
    lo = np.asarray(B.to_cpu(on.forward(Xa)["logits"].data))
    lg = np.asarray(B.to_cpu(g.forward(Xa)["logits"].data))
    dmax = float(np.abs(lg - lo).max())
    chk("RELOAD-BIT-EXACT", dmax == 0.0, f"max |logit difference| reloaded vs original = {dmax}")
    dmax2 = float(np.abs(np.asarray(B.to_cpu(plain.forward(Xa)["logits"].data)) - lo).max())
    chk("PLAIN-LOAD-DIFFERS", dmax2 > 1e-3,
        f"the un-grafted load differs from the trained model by {dmax2:.4f} of logit -- measured, not assumed")

    print("  This is a WIRING result only: no training, no quality claim, no epoch, nothing shipped.")
    print("=" * 100)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
