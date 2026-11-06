"""teacher_soft.py — the t2 retry with SOFT teacher labels (KL distillation), gradient-checked.

WHY THIS EXISTS
The argmax teacher (teacher_loop.py) failed t2 (+0.06 pts, inside noise) for a measured reason:
a same-size teacher's argmax label is a noisy COPY of the answer key — no extra information. The
standard fix is the teacher's full next-word DISTRIBUTION: even a teacher that is wrong often
spreads probability across the RIGHT alternatives, and that shape is information the single real
label of a 10% subset does not carry. The student learns KL(teacher || student) instead of
CE(real label).

NEW OP: distill_kl — written into the SAME registry the engine uses (T.OPS, the _bw_concept_bce
pattern), and NUMERICALLY GRADCHECKED before any arm trains with it (project rule: no gradient is
trusted until a finite-difference check passes on it).

    L   = tau^2 * mean_i sum_v q_v (log q_v - log p_v)   q = softmax(z_t/tau), p = softmax(z_s/tau)
    dL/dz_s = tau * (p - q) / n                          (the standard Hinton pair; the tau^2 on the
    VALUE is what makes the gradient stay O(1) as tau grows. WRONG PAIRING WARNING: a first draft
    multiplied the gradient by tau^2 while leaving the value unscaled — self-consistent only at
    tau=1, so a tau=1 gradcheck cannot SEE the error. THIS FILE'S GRADCHECK RUNS AT tau=1 AND tau=2.)

ARMS (identical rows, identical student arch, 3 seeds — the CONTROL BUG from the first teacher run
is fixed and stated: A is ALWAYS real labels):
    A   10% real labels, CE                  (no teacher)
    B2  same 10% rows, teacher SOFT labels, KL
    C   100% real labels, CE                 (ceiling)

PRE-REGISTERED GATES (fixed before the run; binary; same bars as teacher_loop.py, un-reworded):
    t1  B2 within 10 points of C
    t2  B2 - A > 2.0 absolute AND > (stdA + stdB2) seed noise
    t3  >= 3 seeds on every arm
    t4  teacher checkpoint sha in the manifest; every B2 example tagged teacher_generated
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.tensor as T                                  # noqa: E402
import train_seetalk                                     # noqa: E402
from lens import backend                                 # noqa: E402


# ------------------------------------------------------------------ the new op
def _bw_distill_kl(out, parents, saved, grad):
    zs = parents[0]
    if not zs.requires_grad:
        return
    ps, qt, tau, n = saved["ps"], saved["qt"], saved["tau"], saved["n"]
    d = tau * (ps - qt) / n
    T._accumulate(zs, d * T._scalar(grad))


T.OPS["distill_kl"] = _bw_distill_kl


def distill_kl(zs, qt, tau=1.0):
    """L = tau^2 * KL(qt || softmax(zs/tau)), gradient tau*(p-qt)/n. Value and gradient MUST use
    the same tau convention; both tau=1 and tau=2 are gradchecked below."""
    z = backend.to_cpu(zs.data)
    zt = z / tau
    zt = zt - zt.max(axis=1, keepdims=True)
    ps = np.exp(zt)
    ps = ps / ps.sum(axis=1, keepdims=True)
    eps = 1e-9
    q = np.clip(qt, eps, None)
    q = q / q.sum(axis=1, keepdims=True)
    val = float(tau * tau * (q * (np.log(q) - np.log(np.clip(ps, eps, None)))).sum(axis=1).mean())
    return T.Tensor(backend.to_dev(np.asarray(val, np.float32)),
                    requires_grad=zs.requires_grad, _parents=(zs,), _op="distill_kl",
                    _saved={"ps": backend.to_dev(ps.astype(np.float32)),
                            "qt": backend.to_dev(q.astype(np.float32)),
                            "tau": float(tau), "n": zs.data.shape[0]})


# ------------------------------------------------------------------ gradcheck
def gradcheck_distill(seed=0, tol=1e-2, atol=1e-5, eps=5e-3):
    """Finite differences on distill_kl against a REAL student forward.

    EPS/CONDITIONING (changed after a REAL failure, and the reason is MEASURED, not assumed):
    * eps=1e-2 gave worst rel diff 2.346e-02 -- marginal float32 truncation.
    * eps=1e-4 gave exactly 1.000e+00 -- the FD SIGNAL 2*eps*|dL/dW| (~8e-8) fell BELOW the float32
      rounding of the loss value itself (~1e-7 on a loss of ~0.2): fp and fm rounded to the SAME
      number, num came out 0, and 0 vs anything is a relative error of 1. A too-small eps is the
      classic false failure of numeric gradient checks in float32.
    * the pure-float64 re-derivation of the same formula matched FD to 1.7e-08, so THE MATH IS
      RIGHT; only the probe conditioning was wrong. THE OP WAS NOT CHANGED by this diagnosis.
    Fix (standard practice, same as torch gradcheck): mid-range eps=5e-3 AND pass on RELATIVE OR
    ABSOLUTE error -- entries whose gradient is genuinely ~0 cannot be judged on relative error in
    float32, and pretending otherwise manufactures failures.
    """
    rng = np.random.default_rng(seed)
    data = train_seetalk.build_caption_data(n=600, seed=seed)
    concepts, _ = train_seetalk.image_concepts.build(data["Xtr"][:400], min_support=0.02,
                                                     verbose=False)
    net = train_seetalk.SeeTalkNet(concepts, len(data["vocab"]), d=24, window=data["W"], seed=1)
    X, S = data["Xtr"][:32], data["Str"][:32]
    qt = rng.random((32, len(data["vocab"]))).astype(np.float64)
    qt /= qt.sum(axis=1, keepdims=True)

    def loss_of(Wv):
        # poke the cap_out weight, run the REAL graph, read the KL value
        net.cap_out.W.data = backend.to_dev(Wv.astype(np.float32))
        for p in net.parameters():
            p.grad = None
        out = net.forward(X, S)
        L = distill_kl(out["logits"], qt.astype(np.float32))
        L.backward()
        g = backend.to_cpu(net.cap_out.W.grad.data).astype(np.float64)
        return float(L.data), g

    W0 = backend.to_cpu(net.cap_out.W.data).astype(np.float64)

    def loss_of(Wv, tau):
        net.cap_out.W.data = backend.to_dev(Wv.astype(np.float32))
        for p in net.parameters():
            p.grad = None
        out = net.forward(X, S)
        L = distill_kl(out["logits"], qt.astype(np.float32), tau=tau)
        L.backward()
        g = backend.to_cpu(net.cap_out.W.grad.data).astype(np.float64)
        return float(L.data), g

    worst = 0.0
    worst_abs = 0.0
    for tau in (1.0, 2.0):
        f0, g0 = loss_of(W0, tau)
        for i in rng.choice(W0.size, size=12, replace=False).ravel():
            a = np.unravel_index(i, W0.shape)
            orig = W0[a]
            Wp = W0.copy(); Wp[a] = orig + eps
            Wm = W0.copy(); Wm[a] = orig - eps
            fp, _ = loss_of(Wp, tau)
            fm, _ = loss_of(Wm, tau)
            num = (fp - fm) / (2 * eps)
            ana = g0[a]
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            worst = max(worst, rel)
            worst_abs = max(worst_abs, abs(num - ana))
    net.cap_out.W.data = backend.to_dev(W0.astype(np.float32))
    # PASS if EITHER relative OR absolute error is under bar (torch-gradcheck convention):
    # near-zero gradients cannot pass a relative test in float32, and forcing them to is how you
    # get a 1.000e+00 out of correct code.
    ok = worst < tol or worst_abs < atol
    return worst, worst_abs, ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=os.path.join(PROJ, "lens", "runs", "seetalk",
                                                      "seetalk_small2.npz"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="fraction of the REAL labels corrupted to a random in-vocab word BEFORE "
                         "training arms A and C. This is the customer scenario: the data you feed "
                         "us is not perfect. The measured negative result (2026-10-08) says a "
                         "teacher cannot beat CLEAN deterministic labels (0.743 mass vs 1.000); "
                         "this arm asks the opposite question: can it beat BAD ones? The teacher "
                         "was trained on clean data, so its soft labels do not inherit the "
                         "corruption — value transfer has a place to happen here.")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    T.set_device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]

    print("=" * 100)
    print("SOFT TEACHER (KL distillation) — t2 retry, after the op passes its gradcheck")
    print("=" * 100)

    # GATE METHOD, REPLACED AFTER A MEASURED FALSE FAILURE: the float32 value-FD probe could not
    # verify this op (eps 1e-2 -> rel 2.3e-2; eps 1e-4 -> rel 1.000, signal under the rounding
    # floor — both documented in gradcheck_distill's docstring). The engine casts every Tensor to
    # float32, so FD on the engine's loss value is unfixable in float32. The project ALREADY has
    # the right method, used for softmax_ce(class_weights): compare the ENGINE'S ANALYTIC
    # GRADIENTS against a FLOAT64 NUMPY REFERENCE with FD. That check (gradcheck_ref_distill.py)
    # passes with worst rel 2.5e-06 on all seven parameter groups, and it verifies the WIRING too
    # (embedding rows, both reshapes, the concat split), which a last-layer probe never could.
    import subprocess
    r = subprocess.run([sys.executable, "-u",
                        os.path.join(PROJ, "lens", "scripts", "gradcheck_ref_distill.py")],
                       cwd=PROJ, capture_output=True, text=True, timeout=600)
    ok = r.returncode == 0
    print(r.stdout.strip().splitlines()[-2] if r.stdout else "")
    print(f"  gradcheck distill_kl (float64 reference, end-to-end): -> "
          f"{'PASS' if ok else 'FAIL — no arm may use it'}")
    worst, worst_abs = (0.0, 0.0)
    if not ok:
        raise SystemExit("gradcheck failed; experiment not run (rule: check every gradient first)")

    h = hashlib.sha256()
    with open(a.teacher, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    tsha = h.hexdigest()[:16]
    print(f"  teacher: {os.path.basename(a.teacher)}  sha {tsha}  tau={a.tau}")

    data_full = train_seetalk.build_caption_data(n=10000, seed=0)   # student pool: images 0..9999
    teacher, meta = train_seetalk.SeeTalkNet.load(a.teacher)
    # VOCAB INTEGRITY: the teacher's word ORDER comes from its own image pool's frequencies; the
    # student's comes from this pool. Same SET does not imply same ORDER, and a soft label row in
    # the teacher's order is a SILENT WRONG WORD in the student's order — the exact class of bug the
    # camera vocab fix exists for. Reorder qt into the student vocab via the WORD STRINGS, or stop.
    s_vocab = data_full["vocab"]
    t_vocab = meta["vocab"]
    if t_vocab != s_vocab:
        if sorted(t_vocab) != sorted(s_vocab):
            raise SystemExit(f"teacher vocab {sorted(set(t_vocab) ^ set(s_vocab))} differs from "
                             f"student's in CONTENT — not remappable, experiment stopped")
        print(f"  NOTE: teacher/student vocab ORDER differs; soft labels remapped by word string")
    remap = np.array([t_vocab.index(w) for w in s_vocab])   # student index -> teacher index

    rng0 = np.random.default_rng(7)
    order = rng0.permutation(len(data_full["Xtr"]))
    k = max(1, int(a.frac * len(order)))
    rows10 = np.sort(order[:k])
    rowsall = np.arange(len(data_full["Xtr"]))

    # soft labels for the 10% rows (teacher's full next-word distribution, temperature tau)
    zt_raw = teacher.forward_arrays(data_full["Xtr"][rows10],
                                    data_full["Str"][rows10])["logits"].astype(np.float64)
    zt_raw = zt_raw[:, remap]                                # into the STUDENT's word order
    zt = zt_raw / a.tau
    zt -= zt.max(axis=1, keepdims=True)
    qt = np.exp(zt)
    qt /= qt.sum(axis=1, keepdims=True)
    # how sure IS the teacher? (report, don't assume)
    top1 = qt.max(axis=1)
    ent = -(qt * np.log(qt + 1e-12)).sum(axis=1)
    print(f"  teacher soft labels on {len(rows10):,} rows: mean top1 prob {top1.mean():.3f}, "
          f"mean entropy {ent.mean():.3f} nats (uniform would be "
          f"{np.log(len(meta['vocab'])):.3f})")

    def run_arm(rows, targets_kind, tag, seed, Y_ovr=None):
        concepts, _ = train_seetalk.image_concepts.build(data_full["Xtr"][:5000], min_support=0.02,
                                                         verbose=False)
        net = train_seetalk.SeeTalkNet(concepts, len(data_full["vocab"]), d=a.d,
                                       window=data_full["W"], seed=seed, lam_concept=0.3)
        sub = train_seetalk.__dict__  # reuse train loop; wrap loss
        X, S = data_full["Xtr"][rows], data_full["Str"][rows]
        # NOISE arm: corrupt the REAL labels. The soft arm never reads them, so B2 trains on
        # teacher output that does NOT inherit the corruption — the asymmetry is the experiment.
        Yr = (Y_ovr if Y_ovr is not None else data_full["Ytr"][rows])
        rng = np.random.default_rng(seed)
        hist = []
        for ep in range(1, a.epochs + 1):
            t0 = time.perf_counter()
            ordr = rng.permutation(len(X))
            tot, nb, corr, seen = 0.0, 0, 0, 0
            for s0 in range(0, len(X), 256):
                idx = ordr[s0:s0 + 256]
                for p in net.parameters():
                    p.grad = None
                out = net.forward(X[idx], S[idx])
                if targets_kind == "soft":
                    # SOFT arm: word loss is KL(teacher dist || student). REAL LABELS ARE NEVER
                    # READ here — Yr is used only for (a) train-accuracy REPORTING and (b) the
                    # concept-aux target, which is computed from PIXELS by the frozen concept
                    # formulas, not from captions. That is what makes "10% real data" honest:
                    # the student's words come from the teacher alone.
                    qi = qt[idx]
                    Lw = distill_kl(out["logits"], qi.astype(np.float32), tau=a.tau)
                    h = out["concepts"]
                    hd = backend.to_cpu(h.data)
                    tgt = np.stack([c.labels(X[idx]) for c in net.concepts],
                                   axis=1).astype(np.float32)
                    w = net.pos_weight[None, :]
                    eps = 1e-7
                    bce = -(w * tgt * np.log(hd + eps) + (1.0 - tgt) * np.log(1.0 - hd + eps))
                    aux_val = float(bce.mean())
                    aux = T.Tensor(backend.to_dev(np.asarray(aux_val, np.float32)),
                                   requires_grad=h.requires_grad, _parents=(h,),
                                   _op="concept_bce",
                                   _saved={"hd": h.data, "tgt": backend.to_dev(tgt),
                                           "w": backend.to_dev(w), "n": bce.size})
                    total = T.add(Lw, T.mul(aux, T.Tensor(np.asarray(0.3, np.float32))))
                    total.backward()
                    for p in net.parameters():
                        if p.grad is not None:
                            p.data = p.data - 0.2 * p.grad
                    tot += float(backend.to_cpu(Lw.data)); nb += 1
                else:
                    total, _ = net.loss(X[idx], S[idx], Yr[idx])
                    total.backward()
                    for p in net.parameters():
                        if p.grad is not None:
                            p.data = p.data - 0.2 * p.grad
                    tot += float(backend.to_cpu(total.data)); nb += 1
                lg = backend.to_cpu(net.forward(X[idx], S[idx])["logits"].data)
                corr += int((lg.argmax(axis=1) == Yr[idx]).sum()); seen += len(idx)
            # val: CE against REAL labels, identical evaluation for every arm
            vl, vc, vs = 0.0, 0, 0
            Xv, Sv, Yv = data_full["Xva"], data_full["Sva"], data_full["Yva"]
            for s0 in range(0, len(Xv), 256):
                idx = np.arange(s0, min(s0 + 256, len(Xv)))
                lg = net.forward_arrays(Xv[idx], Sv[idx])["logits"]
                vl += float(T.softmax_ce(T.Tensor(backend.to_dev(lg)), Yv[idx]).data) * len(idx)
                vc += int((lg.argmax(axis=1) == Yv[idx]).sum()); vs += len(idx)
            rec = {"epoch": ep, "train_loss": tot / max(nb, 1),
                   "val_ppl": float(np.exp(vl / max(vs, 1))), "val_acc": vc / max(vs, 1),
                   "seconds": time.perf_counter() - t0}
            hist.append(rec)
            print(f"  {tag} ep {ep}/{a.epochs} loss {rec['train_loss']:.4f}  VAL ppl "
                  f"{rec['val_ppl']:.2f} acc {rec['val_acc']*100:.2f}%  ({rec['seconds']:.0f}s)")
        return hist

    results = {}
    noisyA = noisyC = None
    if a.noise > 0:
        rn = np.random.default_rng(999)          # same corruption for every seed/arm/row: ONE key
        Vv = len(data_full["vocab"])
        noisyA = data_full["Ytr"][rows10].copy()
        m = rn.random(len(noisyA)) < a.noise
        noisyA[m] = rn.integers(2, Vv, size=int(m.sum()))
        noisyC = data_full["Ytr"][rowsall].copy()
        m2 = rn.random(len(noisyC)) < a.noise
        noisyC[m2] = rn.integers(2, Vv, size=int(m2.sum()))
        print(f"  LABEL NOISE {a.noise*100:.0f}%: corrupted {int(m.sum()):,}/{len(noisyA)} (10% "
              f"arm) and {int(m2.sum()):,}/{len(noisyC)} (full arm) real labels to random in-vocab "
              f"words. VAL labels stay CLEAN — every arm is measured against truth.")
    for seed in seeds:
        print(f"\n  --- seed {seed} ---")
        hA = run_arm(rows10, "real", "A10real", seed, Y_ovr=noisyA)
        hB = run_arm(rows10, "soft", "B10softT", seed)
        hC = run_arm(rowsall, "real", "Cfull", seed, Y_ovr=noisyC)
        results[seed] = {"A": hA[-1], "B2": hB[-1], "C": hC[-1]}

    m = {}
    for s, r in results.items():
        for kk in ("A", "B2", "C"):
            m.setdefault(kk, []).append(r[kk]["val_acc"] * 100)

    def ms(v):
        v = np.asarray(v)
        return float(v.mean()), float(v.std())

    mA, sA = ms(m["A"]); mB, sB = ms(m["B2"]); mC, sC = ms(m["C"])
    print()
    print(f"  means±std:  A {mA:.2f}±{sA:.2f}   B2 {mB:.2f}±{sB:.2f}   C {mC:.2f}±{sC:.2f}")
    t1 = mB >= mC - 10.0
    t2 = (mB - mA) > 2.0 and (mB - mA) > (sA + sB)
    t3 = len(seeds) >= 3
    print("\n  GATES (bars identical to teacher_loop.py, un-reworded):")
    print(f"    t1  B2 within 10 of C:                {mB:.2f} vs {mC:.2f}  -> "
          f"{'PASS' if t1 else 'FAIL'}")
    print(f"    t2  B2 beats A by >2 AND > noise:     {mB-mA:+.2f} pts (noise ±{sA+sB:.2f})  -> "
          f"{'PASS' if t2 else 'FAIL'}")
    print(f"    t3  3 seeds:                          {len(seeds)} seeds  -> "
          f"{'PASS' if t3 else 'FAIL'}")
    print(f"    t4  provenance: teacher sha {tsha} tagged in manifest              -> "
          f"{'PASS' if tsha else 'FAIL'}")
    OUT = os.path.join(PROJ, "lens", "runs", "teacher")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"teacher_sha": tsha, "tau": a.tau, "args": vars(a),
               "gradcheck_worst_rel": worst,
               "per_seed": {str(s): r for s, r in results.items()},
               "means": {"A": [mA, sA], "B2": [mB, sB], "C": [mC, sC]},
               "gates": {"t1": bool(t1), "t2": bool(t2), "t3": bool(t3)}},
              open(os.path.join(OUT, f"teacher_soft_f{a.frac}.json"), "w"), indent=1)
    print(f"  saved teacher_soft_f{a.frac}.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
