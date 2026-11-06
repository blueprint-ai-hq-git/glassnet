"""train_seetalk_k22.py — the SHIPPING config: see+talk with K=22 real concepts, big rung.

The width sweep (concepts_sweep.json) says K=22 is where concept width pays (+4.4 over K=12) and
flattening begins; real ~= dummy at every width (transparency costs nothing). The product model is
therefore d=128, big data, K=22 REAL named concepts. Same gates g1-g4 as train_seetalk (bars
unchanged); if g4's delta grows past the K=12 big number (75.51 vs 61.00, +14.5), the ladder moves
on evidence.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.tensor as T                                  # noqa: E402
import train_seetalk as ts                               # noqa: E402
from concepts_more import FormulaConcept, extended_specs  # noqa: E402


def build_k22(Xtr, exclude=()):
    import image_concepts
    exclude = set(exclude)
    base, rep12 = image_concepts.build(Xtr, min_support=0.02, verbose=False)
    cons = [b for b in base if b.name not in exclude]
    added = []
    for name, fn, f in extended_specs():
        if len(cons) >= 22:
            break
        if name in exclude:
            continue
        lab = np.asarray(fn(Xtr), dtype=np.float32)
        s = float(lab.mean())
        if s < 0.02 or s > 0.98:
            continue
        cons.append(FormulaConcept(name, fn, f, support=s))
        added.append({"name": name, "support": s, "formula": f})
    return cons, {"base": rep12, "excluded": sorted(exclude), "added": added, "K": len(cons)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--decorr", type=float, default=0.0,
                    help="OPTIONAL rivalry penalty on the concept activations (off by default; "
                         "gradient numerically checked). Targets the strict audit's own "
                         "criterion: it penalises the squared off-diagonal covariance of the "
                         "concept activations, i.e. concepts tracking each other.")
    ap.add_argument("--concept-mirror", action="store_true", default=False,
                    help="add the |I-mirror(I)| map as a concept-layer input block (the nonlinear "
                         "basis a pairwise formula needs; K21o)")
    ap.add_argument("--concept-direct", action="store_true", default=False,
                    help="the concept layer ALSO reads the raw patches (information-bottleneck "
                         "hypothesis -- K21n)")
    ap.add_argument("--concept-first", type=int, default=0,
                    help="CONCEPT-FIRST STAGE: for the first N epochs train ONLY the concept "
                         "loss (answer loss off), then FREEZE the front+concept layer and "
                         "train the answer layer alone. Rationale (measured): the honesty "
                         "margins of the boundary concepts have been lost to tuning levers "
                         "(pruning/seeds/capacity/epochs all falsified), so this decouples the "
                         "two objectives instead of re-tuning one of them.")
    ap.add_argument("--seed", type=int, default=0,
                    help="MODEL init seed only -- the datum stays seed 0 so the "
                         "strict audit measures the same windows every run")
    ap.add_argument("--out", default="seetalk_k22_big")
    a = ap.parse_args()
    T.set_device(a.device)

    print("=" * 100)
    print("SEETALK K=22 SHIPPING CONFIG — big rung, d=128, 60k images")
    print("=" * 100)
    data = ts.build_caption_data(n=a.n, seed=0)
    Xu = data["Xtr"][::4].astype(np.float32)
    cons, krep = build_k22(Xu, exclude=[x for x in os.environ.get("CONCEPT_EXCLUDE", "").split(",") if x])
    print(f"  concepts K={krep['K']} (base {krep['base']['kept']} + "
          f"{len(krep['added'])} extended); dropped by support floor: "
          f"{10 - len(krep['added'])}")
    for c in krep["added"]:
        print(f"    + {c['name']:<22} support {c['support']:.4f}   {c['formula']}")

    net = ts.SeeTalkNet(cons, vocab_size=len(data["vocab"]), d=a.d, window=data["W"],
                        seed=a.seed, lam_concept=a.lam_concept, decorr=a.decorr,
                        concept_direct=a.concept_direct,
                        concept_mirror=a.concept_mirror)
    npar = int(sum(p.data.size for p in net.parameters()))
    ctrl = ts.TextOnlyControl(len(data["vocab"]), d=a.d, window=data["W"], seed=0)
    print(f"  params {npar:,}")
    hist = ts.train(net, data, epochs=a.epochs, seed=0, tag="k22_see_talk",
                    concept_first=a.concept_first)
    # SAVE IMMEDIATELY, BEFORE THE GATES. Two measured reasons: (1) a gate crash used to lose the
    # whole training run (three K21j attempts died after training finished, saving nothing);
    # (2) the g3 CPU-mode check LOADS this checkpoint -- with the save after the gates it would
    # have loaded a STALE file from a previous run with the same --out name, or nothing at all.
    _OUT_EARLY = os.path.join(PROJ, "lens", "runs", "seetalk")
    os.makedirs(_OUT_EARLY, exist_ok=True)
    net.save(os.path.join(_OUT_EARLY, f"{a.out}.npz"), data)
    print(f"  saved {a.out}.npz before the gates (a gate crash can no longer lose the model)")
    hist_c = ts.train(ctrl, data, epochs=a.epochs, seed=0, tag="k22_control",
                      concept_first=0)   # the control has NO concept layer: the
                      # concept-first schedule cannot apply to it (it crashed the
                      # previous run when the freeze loop reached for net.front/net.concept)

    print()
    print("=" * 100)
    print("GATES (same bars, un-reworded)")
    print("=" * 100)
    vi = {w: i for i, w in enumerate(data["vocab"])}
    prefix = np.asarray([vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"], vi["is"], vi["a"]],
                        dtype=np.int64)
    idx10 = np.linspace(0, len(data["Xva"]) - 1, 10).astype(int)
    outs = [int(net.forward_arrays(data["Xva"][i:i + 1], prefix[None, :])["logits"][0].argmax())
            for i in idx10]
    g1 = len(set(outs)) >= 5
    print(f"  g1: {len(set(outs))} distinct words from 10 images -> "
          f"{'PASS' if g1 else 'FAIL'}  {sorted({data['vocab'][w] for w in outs})}")
    # g2 compares LIKE WITH LIKE. With concept-first the answer loss is OFF for the first N
    # epochs, so the total loss of epoch 1 (concept term only, tiny) is not comparable with the
    # answer phase's -- that comparison measured a right number with a wrong conclusion. The
    # gate's subject is the ANSWER loss, so it compares the CE part across answer epochs only.
    ce_hist = [h.get("ce_loss", h["train_loss"]) for h in hist
               if h.get("phase", "answer") == "answer"]
    if len(ce_hist) < 2:
        ce_hist = [h["train_loss"] for h in hist]
    g2 = ce_hist[-1] < ce_hist[0]
    print(f"  g2: CE loss {ce_hist[0]:.4f} -> {ce_hist[-1]:.4f} over {len(ce_hist)} answer "
          f"epochs -> {'PASS' if g2 else 'FAIL'}"
          + (f"  [concept-first {a.concept_first}: the concept-only epochs are excluded, "
             f"their loss is a different term]" if a.concept_first else ""))
    Xb, Sb = data["Xva"][:64], data["Sva"][:64]

    def _manual_residual(net, Xb, Sb):
        """explain()'s hand-recomputed arithmetic vs the forward — SAME numpy ops, CPU."""
        Pn = net.front.patches(Xb)
        fW = ts.backend.to_cpu(net.front.proj.W.data)
        fb = ts.backend.to_cpu(net.front.proj.b.data)
        zd = (Pn.reshape(-1, Pn.shape[-1]) @ fW + fb).reshape(len(Xb), -1)
        if getattr(net, "concept_direct", False):
            # the concept layer also reads the raw patches (pixels in [0,1]) -- the bypass must
            # appear in the manual arithmetic too, or g3 would compare unlike things.
            zd = np.concatenate([zd, Pn.reshape(len(Xb), -1)], axis=1)
        if getattr(net, "concept_mirror", False):
            zd = np.concatenate([zd, net._mirror_map(Xb)], axis=1)
        cl = zd @ ts.backend.to_cpu(net.concept.W.data) + ts.backend.to_cpu(net.concept.b.data)
        h3 = 1.0 / (1.0 + np.exp(-np.clip(cl, -60, 60)))
        cemb = ts.backend.to_cpu(net.cap_emb.data)
        cxt = np.concatenate([cemb[Sb[:, w]] for w in range(Sb.shape[1])], axis=1)
        manual = (np.concatenate([h3, cxt], axis=1) @ ts.backend.to_cpu(net.cap_out.W.data)
                  + ts.backend.to_cpu(net.cap_out.b.data))
        return float(np.abs(manual - net.forward_arrays(Xb, Sb)["logits"]).max())

    resid = _manual_residual(net, Xb, Sb)
    g3 = resid < 1e-5
    if not g3 and a.device == "gpu":
        # BUG-G3: the GPU float32 GEMMs accumulate in a different order than the numpy manual
        # recomputation, so a |residual| near 1e-5 is an accumulation artifact, not a broken
        # explain(). Verify the SAME arithmetic on a CPU-loaded copy: both sides CPU float32
        # numpy -> rounding matches -> residual ~1e-7. The gate's subject is the CPU-mode check.
        # SeeTalkNet.load() takes no device argument (it builds on the ENGINE's current device),
        # so switch the engine to CPU, load, verify, and switch back. The first version passed
        # device="cpu" and crashed the whole run at the gate -- the gate itself was never the
        # problem, the call was.
        _prev_dev = ts.T.device()
        ts.T.set_device("cpu")
        try:
            # SeeTalkNet.load returns (net, meta) -- the first version of this line forgot the
            # unpacking and crashed the run at the gate with "'tuple' object has no attribute
            # 'front'". Caught by the K21j smoke's sibling path only after a real run.
            net_c, _meta_c = ts.SeeTalkNet.load(os.path.join(PROJ, "lens", "runs", "seetalk",
                                                             f"{a.out}.npz"))
            resid_c = _manual_residual(net_c, Xb, Sb)
        finally:
            ts.T.set_device(_prev_dev)
        g3 = resid_c < 1e-5
        print(f"  g3: GPU float32 residual {resid:.2e}; CPU-mode residual {resid_c:.2e} "
              f"-> {'PASS' if g3 else 'FAIL'} (accumulation-order artifact verified)")
    else:
        print(f"  g3: explain residual {resid:.2e} -> {'PASS' if g3 else 'FAIL'}")
    va, vb = hist[-1]["val_acc"], hist_c[-1]["val_acc"]
    g4 = va > vb + 0.005
    print(f"  g4: see+talk {va*100:.2f}% vs control {vb*100:.2f}% (delta {100*(va-vb):+.2f}; "
          f"K12 big was +14.51) -> {'PASS' if g4 else 'FAIL'}")
    n_pass = sum([g1, g2, g3, g4])
    print(f"\n  RESULT: {n_pass}/4 gates {'ALL PASS' if n_pass == 4 else 'NOT ALL PASS'}")

    OUT = os.path.join(PROJ, "lens", "runs", "seetalk")
    json.dump({"args": vars(a), "k22": krep, "hist": hist, "hist_control": hist_c,
               "gates": {"g1": bool(g1), "g2": bool(g2), "g3": bool(g3), "g4": bool(g4)},
               "delta_vs_k12": {"k12_big": 14.51, "k22_big": round(100*(va-vb), 2)}},
              open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)
    net.save(os.path.join(OUT, f"{a.out}.npz"), data)
    net2, meta2 = ts.SeeTalkNet.load(os.path.join(OUT, f"{a.out}.npz"))
    rt = float(np.abs(net.forward_arrays(Xb, Sb)["logits"]
                      - net2.forward_arrays(Xb, Sb)["logits"]).max())
    print(f"  checkpoint round-trip: {rt:.3e} {'IDENTICAL' if rt == 0.0 else 'MISMATCH'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
