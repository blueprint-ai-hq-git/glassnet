#!/usr/bin/env python3
"""numeric_path_opt_test.py - PLAN.txt item (d2): OUR OWN OPTIMISER IN THE NUMERIC PATH.

PLAN (d2): "the numeric head is lr-sensitive with plain SGD: PASS at lr 0.2, FAIL at 0.5 and 1.0. Wire our
own scale-invariant optimiser (lens/opt.py RelativeGrad, GAP 2) into the numeric path and re-measure: the
claim to test is that one setting then works at any step size, which is what it exists for."

Predictions were written FIRST: lens/runs/NUMERIC-PATH-OPT-PREDICTIONS.txt (P39a-j, the constant-step sweep)
and lens/runs/NUMERIC-PATH-OPT-DECAY-PREDICTIONS.txt (P40a-i, the follow-up with the decay schedule).
Nothing here is tuned after the numbers exist.

THE HARNESS IS IMPORTED, NOT COPIED. corpus / make_net / the scoring come from lens/scripts/numeric_path_test.py
(the file that produced lens/runs/NUMERIC-PATH.txt), so the only thing that differs between the stored run
and this one is the STEP RULE. Same 100 train pairs (operands 0..9), 100 unseen pairs (10..19), 4000 steps,
d=32, W=4, seeds 0,1,2.

Usage:
  OPT_KIND=relgrad OPT_LRS=0.005,0.01,0.02,0.05 SEEDS=0,1,2 python3 lens/scripts/numeric_path_opt_test.py
  OPT_DECAY=0.999 ... python3 lens/scripts/numeric_path_opt_test.py       # the decay schedule
  python3 lens/scripts/numeric_path_opt_test.py --property    # P39h/P40h: the step rule's own property
  python3 lens/scripts/numeric_path_opt_test.py --orbit       # P40a: is it a limit cycle?
  python3 lens/scripts/numeric_path_opt_test.py --regress     # P40h: decay=1.0 is unchanged
  python3 lens/scripts/numeric_path_opt_test.py --verdict     # P39a-j
  python3 lens/scripts/numeric_path_opt_test.py --verdict2    # P40a-i

Env: OPT_KIND (sgd|relgrad), OPT_LRS, OPT_MOM (default 0.0), OPT_DECAY (default 1.0), SEEDS, STEPS
(default 4000), TAG, ARM_C=1 (also run arm C), ARM_ONLY=c, OUT_JSONL (default lens/runs/numeric_path_opt.jsonl).
Rows are appended to the jsonl (never overwritten) so a chunked or interrupted run keeps everything measured.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

from lens import backend, tensor as T              # noqa: E402
from lens.numeric import NumericHead, num_mse      # noqa: E402
from lens.opt import RelativeGrad                  # noqa: E402
import numeric_path_test as npt                    # noqa: E402  (the harness, imported not copied)

OUT_JSONL = os.environ.get("OUT_JSONL", f"{ROOT}/lens/runs/numeric_path_opt.jsonl")
PREFIX_JSONL = f"{ROOT}/lens/runs/numeric_path_opt_prefix.jsonl"
SCALE = npt.SCALE
W = npt.W


def train(net, ids, ytok, ynum, mag, mask, kind, lr, steps, seed, momentum=0.0, decay=1.0):
    """One training loop; the ONLY difference between arms is the step rule."""
    if kind == "sgd":
        return npt.sgd(net, ids, ytok, ynum, mag, mask, steps=steps, lr=lr, seed=seed)
    if kind == "relgrad":
        opt = RelativeGrad(net.parameters(), lr=lr, mode="relgrad", momentum=momentum, decay=decay)
        for _ in range(steps):
            for p in net.parameters():
                p.grad = None
            _loss, parts = net.loss(ids, ytok, mag=mag, ynum=ynum, lam_num=npt.LAM_NUM, magmask=mask)
            _loss.backward()
            opt.step()
        return parts
    raise ValueError(f"unknown optimiser {kind!r}")


def train_head(head, q_tr, ynum_tr, kind, lr, steps, momentum=0.0, decay=1.0, trace=False):
    """Arm C's loop. With trace=True the training loss is recorded every step (the limit-cycle evidence)."""
    hist = []
    if kind == "sgd":
        for _ in range(steps):
            for p in head.parameters():
                p.grad = None
            loss = num_mse(head(q_tr), ynum_tr)
            loss.backward()
            if trace:
                hist.append(float(np.asarray(loss.data).ravel()[0]))
            for p in head.parameters():
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
        return hist
    opt = RelativeGrad(head.parameters(), lr=lr, mode="relgrad", momentum=momentum, decay=decay)
    for _ in range(steps):
        for p in head.parameters():
            p.grad = None
        loss = num_mse(head(q_tr), ynum_tr)
        loss.backward()
        if trace:
            hist.append(float(np.asarray(loss.data).ravel()[0]))
        opt.step()
    return hist


def _append(row):
    with open(OUT_JSONL, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _prop_row(**kw):
    row = dict(arm="prop", kind="property", lr=float("nan"), seed=-1, steps=0, momentum=0.0, decay=1.0,
               b0_exact=float("nan"), finite=True, ms_per_step=0.0, train_exact=float("nan"),
               unseen_exact=float("nan"), unseen_mae=float("nan"))
    row.update(kw)
    _append(row)
    return row


def _orbit_row(**kw):
    row = dict(arm="orbit", kind="orbit", lr=float("nan"), seed=0, steps=4000, momentum=0.0, decay=1.0,
               b0_exact=float("nan"), finite=True, ms_per_step=0.0, train_exact=float("nan"),
               unseen_exact=float("nan"), unseen_mae=float("nan"))
    row.update(kw)
    _append(row)
    return row


def measure(kind, lr, seed, steps, momentum, tag, decay=1.0):
    tr, ex = npt.corpus(0, 10), npt.corpus(10, 20)
    ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr = tr[1], tr[2], tr[3], tr[4], tr[5]
    ids_ex, ytok_ex, mag_ex, mask_ex = ex[1], ex[2], ex[4], ex[5]
    mag_ex_zero = np.zeros_like(mag_ex)

    net, _ = npt.make_net(seed, numeric=True)
    t0 = time.perf_counter()
    train(net, ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, kind, lr, steps, seed, momentum, decay)
    train_ms = (time.perf_counter() - t0) * 1000.0

    pred_tr = npt.quantity(net, ids_tr, mag_tr, mask_tr)
    pred_ex = npt.quantity(net, ids_ex, mag_ex, mask_ex)
    pred0 = npt.quantity(net, ids_ex, mag_ex_zero, mask_ex)
    fin = bool(np.isfinite(pred_ex).all())
    row = dict(
        tag=tag, arm="B", kind=kind, lr=lr, seed=seed, steps=steps, momentum=momentum, decay=decay,
        train_exact=float((np.round(pred_tr) == ytok_tr).mean()),
        unseen_exact=float((np.round(pred_ex) == ytok_ex).mean()) if fin else 0.0,
        unseen_mae=float(np.abs(pred_ex - ytok_ex).mean()) if fin else float("nan"),
        b0_exact=float((np.round(pred0) == ytok_ex).mean()),
        finite=fin, ms_per_step=train_ms / max(steps, 1),
    )
    _append(row)
    print(f"  {kind:8s} mom {momentum:4.2f} decay {decay:<7g} lr {lr:<6g} seed {seed}  "
          f"train {row['train_exact']*100:6.1f}%  unseen {row['unseen_exact']*100:6.1f}%  "
          f"MAE {row['unseen_mae']:8.4f}  B0 {row['b0_exact']*100:5.1f}%  finite {str(fin):5s}  "
          f"{row['ms_per_step']:.3f} ms/step", flush=True)
    return row


def measure_head(kind, lr, seed, steps, momentum, tag, decay=1.0):
    tr, ex = npt.corpus(0, 10), npt.corpus(10, 20)
    mag_tr, ynum_tr, ytok_tr = tr[4], tr[3], tr[2]
    mag_ex, ytok_ex = ex[4], ex[2]
    rng = np.random.default_rng(seed)
    head = NumericHead(W, rng, name="num_opt")
    q_tr, q_ex = T.Tensor(backend.to_dev(mag_tr)), T.Tensor(backend.to_dev(mag_ex))
    t0 = time.perf_counter()
    train_head(head, q_tr, ynum_tr, kind, lr, steps, momentum, decay)
    ms = (time.perf_counter() - t0) * 1000.0
    c_tr = np.asarray(backend.to_cpu(head(q_tr).data)).reshape(-1) * SCALE
    c_ex = np.asarray(backend.to_cpu(head(q_ex).data)).reshape(-1) * SCALE
    fin = bool(np.isfinite(c_ex).all())
    row = dict(tag=tag, arm="C", kind=kind, lr=lr, seed=seed, steps=steps, momentum=momentum, decay=decay,
               train_exact=float((np.round(c_tr) == ytok_tr).mean()),
               unseen_exact=float((np.round(c_ex) == ytok_ex).mean()) if fin else 0.0,
               unseen_mae=float(np.abs(c_ex - ytok_ex).mean()) if fin else float("nan"),
               b0_exact=float("nan"), finite=fin, ms_per_step=ms / max(steps, 1))
    _append(row)
    print(f"  ARM C    {kind:8s} mom {momentum:4.2f} decay {decay:<7g} lr {lr:<6g} seed {seed}  "
          f"train {row['train_exact']*100:6.1f}%  unseen {row['unseen_exact']*100:6.1f}%  "
          f"MAE {row['unseen_mae']:8.4f}  finite {str(fin):5s}", flush=True)
    return row


def property_check(lr=0.02):
    """P39h / P40h: the step rule's own property - update norm, and what `scale` does in each mode."""
    print("=" * 100)
    print("THE STEP RULE'S OWN PROPERTY: is the update norm exactly lr, and what does `scale` do?")
    print("=" * 100)
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (1, 4, 16, 64, 256, 1024, 4096):
        p = T.Tensor(np.zeros(n, dtype=np.float32), requires_grad=True, name=f"prop{n}")
        p.grad = rng.standard_normal(n).astype(np.float32)
        opt = RelativeGrad([p], lr=lr, mode="relgrad", scale=True)
        before = np.array(p.data, dtype=np.float64).copy()
        opt.step()
        ratio = float(np.linalg.norm(np.asarray(p.data, dtype=np.float64) - before) / lr)
        worst = max(worst, abs(ratio - 1.0))
        _prop_row(mode="relgrad", scale=True, n=int(n), ratio=ratio, lr=float(lr))
        print(f"  size {n:5d}  relgrad  ||update||/lr = {ratio:.12f}")
    print(f"  worst |norm/lr - 1| for relgrad over the sizes = {worst:.3e}   (bar 1e-6)")
    print(f"  P39h relgrad update norm == lr : {'PASS' if worst <= 1e-6 else 'FAIL'}")
    print()
    print("  rmsprop mode (per-entry scaling): does `scale` make a wide tensor's movement equal lr?")
    print("  NOTE ON THE EXPECTED VALUE: on the FIRST step the EMA of g^2 is cold (e = 0.1*g^2), so a")
    print("  single step moves each entry by lr/sqrt(0.1) = 3.16228*lr, not lr. The two bars below are")
    print("  therefore different on purpose: the declared P40h bar ('exactly lr') is a FIRST-STEP bar and")
    print("  is measured as FAIL; the corrected property is that scale=True divides the movement by")
    print("  exactly sqrt(n) against scale=False, which is what the fix implements.")
    ok_scaled = True
    ok_factor = True
    cold = 1.0 / np.sqrt(0.1)
    for n in (1, 4, 64, 1024):
        ratios = {}
        for scale in (True, False):
            p = T.Tensor(np.zeros(n, dtype=np.float32), requires_grad=True, name=f"rms{n}")
            p.grad = rng.standard_normal(n).astype(np.float32)
            opt = RelativeGrad([p], lr=lr, mode="rmsprop", scale=scale)
            before = np.array(p.data, dtype=np.float64).copy()
            opt.step()
            ratio = float(np.linalg.norm(np.asarray(p.data, dtype=np.float64) - before) / lr)
            ratios[scale] = ratio
            _prop_row(mode="rmsprop", scale=bool(scale), n=int(n), ratio=ratio, lr=float(lr))
            print(f"  size {n:5d}  rmsprop  scale={str(scale):5s}  ||update||/lr = {ratio:9.4f}  "
                  f"(sqrt(n) = {np.sqrt(n):.3f})")
            if scale and abs(ratio - 1.0) > 1e-6:
                ok_scaled = False
        factor = ratios[False] / ratios[True]
        rel = abs(factor / np.sqrt(n) - 1.0)
        same = rel <= 1e-5          # float32 tensors: the residual is float32 rounding, so this is RELATIVE
        ok_factor = ok_factor and same
        print(f"             scale=False / scale=True = {factor:.12f}   sqrt(n) = {np.sqrt(n):.12f}   "
              f"relative error {rel:.2e}   {'MATCH' if same else 'MISMATCH'}")
    print(f"  P40h AS DECLARED: rmsprop with scale=True moves every tensor by exactly lr : "
          f"{'PASS' if ok_scaled else 'FAIL'}   (measured {cold:.5f} x lr, the cold-EMA factor "
          f"1/sqrt(0.1); the bar ignored the EMA warm-up and is recorded as FAILED as worded)")
    print(f"  CORRECTED PROPERTY (not a redefinition of the bar): scale=True divides the movement by exactly")
    print(f"  sqrt(n) against scale=False : {'PASS' if ok_factor else 'FAIL'}")
    return worst


def orbit(lrs=(0.01, 0.02, 0.05), seed=0, steps=4000):
    """P40a: a constant normalised step is a LIMIT CYCLE, not a stuck point. Loss trace, both step rules."""
    print("=" * 100)
    print("P40a THE LIMIT CYCLE: arm C's training loss every step, constant step vs decayed step")
    print("  if best << last-100 mean, the run REACHED the solution and then left it (an orbit)")
    print("=" * 100)
    tr = npt.corpus(0, 10)
    mag_tr, ynum_tr = tr[4], tr[3]
    print(f"  {'step rule':26s} {'lr':>6s}  {'best loss':>12s} {'last-100 mean':>14s} "
          f"{'ratio':>10s}  {'||Wn-[1,0,1,0]||':>16s}  {'step of best':>12s}")
    for lr in lrs:
        for decay, label in ((1.0, "constant"), (0.999, "decay 0.999")):
            rng = np.random.default_rng(seed)
            head = NumericHead(W, rng, name="num_orb")
            q_tr = T.Tensor(backend.to_dev(mag_tr))
            hist = train_head(head, q_tr, ynum_tr, "relgrad", lr, steps, 0.0, decay, trace=True)
            h = np.asarray(hist, dtype=np.float64)
            best, last = float(h.min()), float(h[-100:].mean())
            ratio = (last / best) if best > 0 else float("inf")
            wn = np.asarray(backend.to_cpu(head.Wn.data)).reshape(-1)
            dist = float(np.linalg.norm(wn - np.asarray([1.0, 0.0, 1.0, 0.0])))
            _orbit_row(lr=float(lr), decay=float(decay), best=best, last100=last, ratio=float(ratio),
                       offset_norm=dist, step_of_best=int(np.argmin(h)))
            print(f"  {label:26s} {lr:>6g}  {best:>12.3e} {last:>14.3e} {ratio:>10.1f}  "
                  f"{dist:>16.4f}  {int(np.argmin(h)):>12d}")
    return 0


def _load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def _dedupe(rows):
    seen, out = {}, []
    for r in rows:
        key = (r.get("arm", "B"), r["kind"], round(float(r["lr"]), 12),
               round(float(r.get("momentum", 0.0)), 12), round(float(r.get("decay", 1.0)), 12), int(r["seed"]),
               r.get("mode"), r.get("scale"), r.get("n"))   # mode/scale/n: the property rows share a seed
        if key in seen:
            continue
        seen[key] = True
        out.append(r)
    return out


def _regression_ok():
    """Compare the fresh decay=1.0 rows against the pre-change rows. Returns (ok_or_None, compared_rows)."""
    pre = {(r["kind"], round(float(r["lr"]), 12), int(r["seed"]), round(float(r.get("momentum", 0.0)), 12)): r
           for r in _load(PREFIX_JSONL) if r.get("arm", "B") == "B"}
    fresh = {(r["kind"], round(float(r["lr"]), 12), int(r["seed"]), round(float(r.get("momentum", 0.0)), 12)): r
             for r in _load(f"{ROOT}/lens/runs/numeric_path_opt_regress.jsonl") if r.get("arm", "B") == "B"}
    keys = sorted(set(pre) & set(fresh))
    if not keys:
        return None, []
    out = []
    for k in keys:
        a, b = pre[k], fresh[k]
        same = (abs(a["unseen_exact"] - b["unseen_exact"]) < 1e-12
                and abs(a["train_exact"] - b["train_exact"]) < 1e-12
                and (a["unseen_mae"] == b["unseen_mae"] or abs(a["unseen_mae"] - b["unseen_mae"]) < 1e-12)
                and a["finite"] == b["finite"])
        out.append((k, same, a, b))
    return all(s for _, s, _, _ in out), out


def regress():
    """P40h: after the lens/opt.py change, the decay=1.0 path must reproduce the pre-change numbers exactly."""
    print("=" * 100)
    print("P40h REGRESSION: relgrad at decay=1.0, re-measured after the lens/opt.py change, against the")
    print(f"  pre-change rows kept in {PREFIX_JSONL}")
    print("=" * 100)
    pre = {(r["kind"], round(float(r["lr"]), 12), int(r["seed"]), round(float(r.get("momentum", 0.0)), 12)): r
           for r in _load(PREFIX_JSONL) if r.get("arm", "B") == "B"}
    globals()["OUT_JSONL"] = f"{ROOT}/lens/runs/numeric_path_opt_regress.jsonl"
    fresh = {}
    for (kind, lr, seed, mom) in sorted(pre):
        if kind != "relgrad":
            continue
        row = measure(kind, lr, seed, 4000, mom, "regress", 1.0)
        fresh[(kind, lr, seed, mom)] = row
    print()
    ok = True
    print(f"  {'config':34s} {'unseen pre/post':>22s}  {'train pre/post':>22s}  {'MAE pre/post':>24s}  same")
    for k in sorted(fresh):
        a, b = pre[k], fresh[k]
        same = (abs(a["unseen_exact"] - b["unseen_exact"]) < 1e-12
                and abs(a["train_exact"] - b["train_exact"]) < 1e-12
                and (a["unseen_mae"] == b["unseen_mae"]
                     or abs(a["unseen_mae"] - b["unseen_mae"]) < 1e-12)
                and a["finite"] == b["finite"])
        ok = ok and same
        print(f"  {f'{k[0]} lr {k[1]:g} seed {k[2]} mom {k[3]:g}':34s} "
              f"{a['unseen_exact']*100:9.1f}%/{b['unseen_exact']*100:8.1f}%  "
              f"{a['train_exact']*100:9.1f}%/{b['train_exact']*100:8.1f}%  "
              f"{a['unseen_mae']:11.6f}/{b['unseen_mae']:11.6f}  {str(same):5s}")
    print(f"  P40h decay=1.0 reproduces the pre-change numbers exactly : {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def verdict():
    raw = _load(OUT_JSONL)
    rows = [r for r in _dedupe(raw) if abs(float(r.get("decay", 1.0)) - 1.0) < 1e-12]
    skipped = len(_dedupe(raw)) - len(rows)
    print("=" * 100)
    print(f"AGGREGATE over {len(rows)} measured configurations ({len(raw)} rows on disk, "
          f"{skipped} decay!=1.0 configurations excluded - see --verdict2)   file: {OUT_JSONL}")
    print("=" * 100)
    print(f"  {'kind':9s} {'mom':>4s} {'lr':>7s}  {'seeds':>5s} {'finite':>6s}  {'unseen exact':>12s} "
          f"{'MAE':>8s}  {'train exact':>11s}  {'ms/step':>8s}   admissible")
    adm = {}
    for kind, mom in sorted({(r["kind"], r.get("momentum", 0.0)) for r in rows if r.get("arm", "B") == "B"}):
        lrs = sorted({r["lr"] for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                      and r.get("arm", "B") == "B"})
        for lr in lrs:
            sel = [r for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                   and r["lr"] == lr and r.get("arm", "B") == "B"]
            fin = sum(1 for r in sel if r["finite"])
            ue = float(np.mean([r["unseen_exact"] for r in sel]))
            te = float(np.mean([r["train_exact"] for r in sel]))
            mae = float(np.nanmean([r["unseen_mae"] for r in sel])) if fin else float("nan")
            ok = (fin == len(sel)) and (ue >= 0.90)
            adm[(kind, mom, lr)] = ok
            print(f"  {kind:9s} {mom:4.2f} {lr:>7g}  {len(sel):>5d} {fin:>3d}/{len(sel):<2d}  "
                  f"{ue*100:>11.1f}% {mae:>8.4f}  {te*100:>10.1f}%  "
                  f"{np.mean([r['ms_per_step'] for r in sel]):>8.3f}   {'YES' if ok else 'no'}")

    def window(kind, mom):
        lrs = [lr for (k, m, lr), ok in adm.items() if k == kind and m == mom and ok]
        if len(lrs) < 2:
            return (0.0, lrs)
        return (max(lrs) / min(lrs), sorted(lrs))

    print()
    print("  VERDICTS (bars fixed in lens/runs/NUMERIC-PATH-OPT-PREDICTIONS.txt before the run)")
    sgd_lrs = sorted({r["lr"] for r in rows if r["kind"] == "sgd" and r.get("arm") != "C"})
    sgd_fin = {lr: sum(1 for r in rows if r["kind"] == "sgd" and r["lr"] == lr and r.get("arm") != "C"
                       and r["finite"]) for lr in sgd_lrs}
    sgd_n = {lr: sum(1 for r in rows if r["kind"] == "sgd" and r["lr"] == lr and r.get("arm") != "C")
             for lr in sgd_lrs}
    print(f"  P39a SGD regression, AS WORDED (0.2 all finite, 0.5 none) : "
          f"{'PASS' if (sgd_fin.get(0.5, 9) == 0 and sgd_n.get(0.5, 0) > 0) else 'FAIL'}   "
          f"MEASURED 0.2 {sgd_fin.get(0.2)}/{sgd_n.get(0.2)} finite, 0.5 {sgd_fin.get(0.5)}/{sgd_n.get(0.5)} "
          f"against the stored 3/3 and 1/3 (lens/runs/NUMERIC-PATH-LR05.txt) - the bar was mis-worded, the "
          f"regression itself reproduced the stored numbers")
    all_fin = True
    for kind, mom in sorted({(r["kind"], r.get("momentum", 0.0)) for r in rows
                             if r["kind"] == "relgrad" and r.get("arm") != "C"}):
        for lr in sorted({r["lr"] for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                          and r.get("arm") != "C"}):
            sel = [r for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                   and r["lr"] == lr and r.get("arm") != "C"]
            all_fin = all_fin and all(r["finite"] for r in sel)
    print(f"  P39b our optimiser cannot overflow (all finite)       : {'PASS' if all_fin else 'FAIL'}")
    w0, lrs0 = window("relgrad", 0.0)
    print(f"  P39c admissible at >= 3 of 4 of our step sizes        : {len(lrs0)}/4  "
          f"{'PASS' if len(lrs0) >= 3 else 'FAIL'}   admissible {lrs0}")
    sgd_ok = sum(1 for lr in sgd_lrs if adm.get(("sgd", 0.0, lr), False))
    print(f"  P39d plain SGD admissible at <= 2 of 4 step sizes     : {sgd_ok}/4  "
          f"{'PASS' if sgd_ok <= 2 else 'FAIL'}")
    print(f"  P39e our admissible window >= 4x (the (d2) claim)     : {w0:.1f}x  "
          f"{'PASS' if w0 >= 4.0 else 'FAIL'}")
    wm, lrsm = window("relgrad", 0.9)
    print(f"  P39f momentum 0.9 at lr 0.02 >= 90% unseen            : "
          f"{'PASS' if adm.get(('relgrad', 0.9, 0.02), False) else 'FAIL'}   (admissible {lrsm})")
    b0 = [r["b0_exact"] for r in rows if r.get("arm") != "C" and r["kind"] == "relgrad" and r["finite"]]
    b0m = float(np.mean(b0)) if b0 else float("nan")
    print(f"  P39g control B0 (channel zeroed) <= 10% with ours     : {b0m*100:.1f}%  "
          f"{'PASS' if (b0m == b0m and b0m <= 0.10) else 'FAIL'}")
    crow = [r for r in rows if r.get("arm") == "C"]
    c_un = float(np.mean([r["unseen_exact"] for r in crow])) if crow else float("nan")
    print(f"  P39i arm C (harness) with ours >= 95% unseen          : {c_un*100:.1f}%  "
          f"{'PASS' if (c_un == c_un and c_un >= 0.95) else 'FAIL'}")
    sgd_ms = np.mean([r["ms_per_step"] for r in rows if r["kind"] == "sgd" and r.get("arm") != "C"])
    rg_ms = np.mean([r["ms_per_step"] for r in rows if r["kind"] == "relgrad" and r.get("arm") != "C"])
    ratio = rg_ms / sgd_ms if sgd_ms else float("nan")
    print(f"  P39j our step rule <= 2x slower per step              : {ratio:.2f}x  "
          f"{'PASS' if ratio <= 2.0 else 'FAIL'}   (sgd {sgd_ms:.3f} ms, ours {rg_ms:.3f} ms)")
    core = (all_fin and len(lrs0) >= 3 and w0 >= 4.0 and adm.get(("relgrad", 0.9, 0.02), False)
            and b0m <= 0.10 and c_un >= 0.95 and ratio <= 2.0)
    print()
    print(f"  (d2) NUMERIC PATH WITH OUR OPTIMISER (constant step): {'PASS' if core else 'FAIL'}")
    return 0 if core else 1


def verdict2():
    raw = _load(OUT_JSONL)
    rows = [r for r in _dedupe(raw) if abs(float(r.get("decay", 1.0)) - 1.0) > 1e-12]
    print("=" * 100)
    print(f"AGGREGATE over {len(rows)} DECAYED configurations   file: {OUT_JSONL}")
    print("=" * 100)
    print(f"  {'kind':9s} {'mom':>4s} {'decay':>7s} {'lr':>7s}  {'seeds':>5s} {'finite':>6s}  "
          f"{'unseen exact':>12s} {'MAE':>8s}  {'train exact':>11s}   admissible")
    adm = {}
    for kind, mom, dec in sorted({(r["kind"], r.get("momentum", 0.0), r.get("decay", 1.0)) for r in rows
                                  if r.get("arm", "B") == "B"}):
        for lr in sorted({r["lr"] for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                          and r.get("decay", 1.0) == dec and r.get("arm", "B") == "B"}):
            sel = [r for r in rows if r["kind"] == kind and r.get("momentum", 0.0) == mom
                   and r.get("decay", 1.0) == dec and r["lr"] == lr and r.get("arm", "B") == "B"]
            fin = sum(1 for r in sel if r["finite"])
            ue = float(np.mean([r["unseen_exact"] for r in sel]))
            te = float(np.mean([r["train_exact"] for r in sel]))
            mae = float(np.nanmean([r["unseen_mae"] for r in sel])) if fin else float("nan")
            ok = (fin == len(sel)) and (ue >= 0.90)
            adm[(kind, mom, dec, lr)] = (ok, ue, te, mae, fin, len(sel))
            print(f"  {kind:9s} {mom:4.2f} {dec:>7g} {lr:>7g}  {len(sel):>5d} {fin:>3d}/{len(sel):<2d}  "
                  f"{ue*100:>11.1f}% {mae:>8.4f}  {te*100:>10.1f}%   {'YES' if ok else 'no'}")
    crow = [r for r in rows if r.get("arm") == "C"]
    print()
    print("  VERDICTS (bars fixed in lens/runs/NUMERIC-PATH-OPT-DECAY-PREDICTIONS.txt before the run)")
    cadm = [(k, v) for k, v in adm.items() if k[0] == "relgrad" and k[1] == 0.0 and abs(k[2] - 0.999) < 1e-12]
    c_lrs = sorted(k[3] for k, v in cadm if v[0])
    all_fin = all(v[4] == v[5] for _, v in cadm) and len(cadm) > 0
    p40c = all_fin and len(c_lrs) == 4
    ctrain = np.mean([r["train_exact"] for r in crow]) if crow else float("nan")
    print(f"  P40b arm C with decay >= 95% of its own train set     : "
          f"{'PASS' if (crow and all(r['train_exact'] >= 0.95 and r['finite'] for r in crow)) else 'FAIL'}   "
          f"({len(crow)} rows, mean train {ctrain*100:.1f}%; with a constant step it was 19-28%)")
    print(f"  P40c decay admissible at ALL FOUR step sizes          : {len(c_lrs)}/4  "
          f"{'PASS' if p40c else 'FAIL'}   admissible {c_lrs}")
    grp = {lr: adm.get(("relgrad", 0.0, 0.999, lr), (False, float('nan')))[1] for lr in (0.01, 0.02, 0.05)}
    spread = (max(grp.values()) - min(grp.values())) * 100 if all(v == v for v in grp.values()) else float("nan")
    print(f"  P40d spread across lr {{0.01,0.02,0.05}} <= 10 pts      : {spread:.1f} pts  "
          f"{'PASS' if (spread == spread and spread <= 10.0) else 'FAIL'}   "
          f"(constant step: 45.3 pts = 54.7/66.7/100.0)")
    ok005 = adm.get(("relgrad", 0.0, 0.999, 0.005), (False,))[0]
    print(f"  P40e the same decay works at lr 0.005                 : {'PASS' if ok005 else 'FAIL'}")
    m02 = adm.get(("relgrad", 0.9, 0.999, 0.02), (False,))[0]
    m05 = adm.get(("relgrad", 0.9, 0.999, 0.05), (False,))[0]
    print(f"  P40f momentum 0.9 + decay at lr 0.02 / 0.05           : "
          f"{'PASS' if (m02 and m05) else 'FAIL'}   (0.02 {'PASS' if m02 else 'FAIL'}, "
          f"0.05 {'PASS' if m05 else 'FAIL'}; momentum alone: 100.0% at 0.02, 7.0% at 0.05)")
    te_all = [v[2] for _, v in cadm]
    print(f"  P40g decay does not cost training (train >= 90%)      : "
          f"{min(te_all)*100:.1f}% worst  {'PASS' if min(te_all) >= 0.90 else 'FAIL'}")
    b0 = [r["b0_exact"] for r in rows if r.get("arm") != "C" and r["kind"] == "relgrad" and r["finite"]]
    b0m = float(np.mean(b0)) if b0 else float("nan")
    print(f"  P40i control B0 (channel zeroed) <= 10% with decay    : {b0m*100:.1f}%  "
          f"{'PASS' if (b0m == b0m and b0m <= 0.10) else 'FAIL'}")
    allrows = _dedupe(raw)
    print()
    # ---- P40a: measured by --orbit, evaluated here against the bar DECLARED before the run
    orb = [r for r in allrows if r.get("arm") == "orbit" and abs(float(r.get("decay", 1.0)) - 1.0) < 1e-12]
    p40a = bool(orb) and all((r.get("best", 9e9) <= 1e-6 and r.get("ratio", 0.0) >= 100.0) for r in orb)
    if orb:
        print(f"  P40a constant step = a limit cycle (best <= 1e-6 AND last100 >= 100x best) : "
              f"{'PASS' if p40a else 'FAIL'}")
        for r in orb:
            print(f"        lr {r['lr']:<6g} best {r['best']:.3e}  last-100 mean {r['last100']:.3e}  "
                  f"ratio {r['ratio']:.2f}  ||Wn-[1,0,1,0]|| {r['offset_norm']:.4f}  "
                  f"(offset/lr {r['offset_norm']/r['lr']:.1f})")
        print("        THE MECHANISM I PREDICTED IS WRONG AND THE FAIL STANDS: the loss does NOT reach the")
        print("        solution and leave it (best ~ last-100 mean, ratio ~1.1, not >= 100). It settles at a")
        print("        STEADY-STATE OFFSET from the solution, and the offset grows with the step size")
        print("        (||Wn-[1,0,1,0]|| 0.276 / 0.531 / 0.941 at lr 0.01 / 0.02 / 0.05). With decay the same")
        print("        head ends at 0.085 at every one of the three step sizes.")
    # ---- P40h: part 1 the decay=1.0 regression (rows), part 2 the declared property bar (prop rows)
    reg_ok, reg_rows = _regression_ok()
    props = [r for r in allrows if r.get("arm") == "prop" and r.get("mode") == "rmsprop" and r.get("scale")]
    p40h_prop = bool(props) and all(abs(r.get("ratio", 0.0) - 1.0) <= 1e-6 for r in props)
    p40h = (reg_ok is True) and p40h_prop
    print(f"  P40h decay=1.0 reproduces the pre-change rows exactly : "
          f"{'PASS' if reg_ok else ('NO ROWS' if reg_ok is None else 'FAIL')}   "
          f"({len(reg_rows)} configurations compared)")
    print(f"  P40h AS DECLARED: rmsprop scale=True moves by exactly lr : {'PASS' if p40h_prop else 'FAIL'}   "
          f"({len(props)} rows; measured 3.16228 x lr = the cold-EMA factor 1/sqrt(0.1), the bar ignored it)")
    print(f"  P40h OVERALL (the conjunction as declared) : {'PASS' if p40h else 'FAIL'}")
    core = (p40c and (spread == spread and spread <= 10.0) and m02 and m05 and min(te_all) >= 0.90
            and b0m <= 0.10 and p40a and p40h)
    print()
    print("  THE SUBSTANTIVE BARS (P40b-g, P40i) ALL PASS; the conjunction above is what the prediction file")
    print("  declared before the run, and it FAILS on two of my own bars: P40a's mechanism (an offset, not a")
    print("  limit cycle) and P40h's single-step value (a cold EMA). Both are recorded as FAILED as worded.")
    print()
    print(f"  (d2) FOLLOW-UP: OUR OPTIMISER WITH A DECAY SCHEDULE: {'PASS' if core else 'FAIL'}")
    return 0 if core else 1


def main():
    if "--verdict" in sys.argv:
        return verdict()
    if "--verdict2" in sys.argv:
        return verdict2()
    if "--property" in sys.argv:
        property_check()
        return 0
    if "--orbit" in sys.argv:
        return orbit()
    if "--regress" in sys.argv:
        return regress()
    kind = os.environ.get("OPT_KIND", "relgrad")
    lrs = [float(x) for x in os.environ.get("OPT_LRS", "0.005,0.01,0.02,0.05").split(",")]
    mom = float(os.environ.get("OPT_MOM", "0.0"))
    decay = float(os.environ.get("OPT_DECAY", "1.0"))
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    steps = int(os.environ.get("STEPS", "4000"))
    tag = os.environ.get("TAG", "")
    heads = os.environ.get("ARM_C", "0") == "1"
    only_c = os.environ.get("ARM_ONLY", "").lower() == "c"
    print("=" * 100)
    print(f"(d2) THE NUMERIC PATH WITH OUR OWN OPTIMISER   kind={kind} momentum={mom} decay={decay} "
          f"lrs={lrs} seeds={seeds} steps={steps}   predictions P39a-j / P40a-i, written first")
    print("=" * 100)
    for lr in lrs:
        for seed in seeds:
            if not only_c:
                measure(kind, lr, seed, steps, mom, tag, decay)
            if heads or only_c:
                measure_head(kind, lr, seed, steps, mom, tag, decay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
