#!/usr/bin/env python3
"""numeric_bounded_lr_sweep.py - DIAGNOSTIC for PLAN item (d3b): how much smaller is the bounded form's
stable step size, and why.

The gate result is already on the record (lens/runs/NUMERIC-BOUNDED.txt): the bounded form is exactly as
expressive (arm C 100% unseen exact, MAE 0.0000) but inside the full architecture it went NON-FINITE on 2 of
3 seeds at the gate's lr 0.2, against the shipped unbounded form's 3/3 finite -> P44c FAIL. This diagnostic
measures the margin and the mechanism; it does NOT redefine that FAIL. Predictions D1/D2 written first in
lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt ADDENDUM 1. Output: lens/runs/NUMERIC-BOUNDED-LR.txt.

Both representations run in the SAME process, on the SAME pairs, with the SAME harness (imported unchanged):
only the representation and the step size differ.

Usage: python3 lens/scripts/numeric_bounded_lr_sweep.py
"""
import datetime
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.chdir(ROOT)

import numeric_path_test as npt                                        # noqa: E402
import numeric_bounded_test as nbt                                     # noqa: E402

LRS = [float(x) for x in os.environ.get("BOUNDED_LRS", "0.005,0.01,0.02,0.05,0.1,0.2").split(",")]
SEEDS = [int(x) for x in os.environ.get("BOUNDED_SEEDS", "0,1,2").split(",")]
STEPS = int(os.environ.get("NUMERIC_STEPS", "4000"))
OUT = os.path.join(ROOT, "lens", "runs", "NUMERIC-BOUNDED-LR.txt")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def run_arm_b(mode, seed, lr):
    """One arm-B run. Returns (unseen exact, MAE in original units, finite, train exact)."""
    if mode == "bounded":
        tr = nbt.corpus_bounded(0, 10)
        ex = nbt.corpus_bounded(10, 20)
        ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, d_tr = tr[1], tr[2], tr[3], tr[4], tr[5], tr[6]
        ids_ex, ytok_ex, _, mag_ex, mask_ex, d_ex = ex[1], ex[2], ex[3], ex[4], ex[5], ex[6]
        net, _ = npt.make_net(seed, numeric=True)
        npt.sgd(net, ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, steps=STEPS, lr=lr, seed=seed)
        pred_tr = nbt.quantity(net, ids_tr, mag_tr, mask_tr, d_tr)
        pred_ex = nbt.quantity(net, ids_ex, mag_ex, mask_ex, d_ex)
    else:
        tr = npt.corpus(0, 10)
        ex = npt.corpus(10, 20)
        ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr = tr[1], tr[2], tr[3], tr[4], tr[5]
        ids_ex, ytok_ex, _, mag_ex, mask_ex = ex[1], ex[2], ex[3], ex[4], ex[5]
        net, _ = npt.make_net(seed, numeric=True)
        npt.sgd(net, ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, steps=STEPS, lr=lr, seed=seed)
        pred_tr = nbt.quantity(net, ids_tr, mag_tr, mask_tr, np.full((len(ytok_tr), 1), npt.SCALE))
        pred_ex = nbt.quantity(net, ids_ex, mag_ex, mask_ex, np.full((len(ytok_ex), 1), npt.SCALE))
    finite = bool(np.isfinite(pred_ex).all())
    ex_exact = float((np.round(pred_ex) == ytok_ex).mean()) if finite else 0.0
    tr_exact = float((np.round(pred_tr) == ytok_tr).mean()) if bool(np.isfinite(pred_tr).all()) else 0.0
    mae = float(np.abs(pred_ex - ytok_ex).mean()) if finite else float("nan")
    return ex_exact, mae, finite, tr_exact


def grad_scale(mode, seed=0):
    """||g|| per tensor after ONE real batch, so the mechanism is measured and not guessed.

    NOTE ON A BUG OF MINE, FIXED HERE: the first version of this helper called
    `net.loss(...).backward()`, but lens.Net.loss() returns (loss, parts) - a tuple has no .backward(). The
    sweep table was already written to stdout when it died, so the table is real; only this function is new.
    """
    if mode == "bounded":
        tr = nbt.corpus_bounded(0, 10)
    else:
        tr = npt.corpus(0, 10)
    ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr = tr[1], tr[2], tr[3], tr[4], tr[5]
    net, _ = npt.make_net(seed, numeric=True)
    for p in net.parameters():
        p.grad = None
    loss, _parts = net.loss(ids_tr, ytok_tr, mag=mag_tr, ynum=ynum_tr, lam_num=npt.LAM_NUM, magmask=mask_tr)
    loss.backward()
    from lens import backend
    out = {}
    for p in net.parameters():
        if p.grad is None:
            continue
        g = np.asarray(backend.to_cpu(p.grad), dtype=np.float64)
        out[p.name] = float(np.sqrt((g * g).sum()))
    return out


def main():
    say("=" * 100)
    say("DIAGNOSTIC (d3b): STEP-SIZE MARGIN AND MECHANISM OF THE BOUNDED REPRESENTATION")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    say("predictions D1/D2 written first: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt ADDENDUM 1")
    say("the gate FAIL (P44c, lr 0.2) is NOT redefined here; this measures how far the failure extends")
    say("=" * 100)
    say(f"arm B only (concepts + magnitudes + numeric head), {STEPS} steps, seeds {SEEDS}")
    say("")

    results = {}
    for mode in ("unbounded", "bounded"):
        for lr in LRS:
            rows = [run_arm_b(mode, s, lr) for s in SEEDS]
            exs = [r[0] for r in rows]
            fin = sum(int(r[2]) for r in rows)
            maes = [r[1] for r in rows if np.isfinite(r[1])]
            results[(mode, lr)] = {"exact": exs, "finite": fin, "mae": maes, "train": [r[3] for r in rows]}
            say(f"  {mode:10s} lr {lr:<6g} unseen exact per seed {[round(100*x, 1) for x in exs]}   "
                f"finite {fin}/{len(SEEDS)}   train exact {[round(100*x, 1) for x in [r[3] for r in rows]]}   "
                f"MAE {('%.4f' % np.mean(maes)) if maes else 'nan'}")

    say("")
    say("  D1 STEP-SIZE MARGIN, read off the table above")
    for mode in ("unbounded", "bounded"):
        ok = [lr for lr in LRS
              if results[(mode, lr)]["finite"] == len(SEEDS)
              and min(results[(mode, lr)]["exact"]) >= 0.90]
        bad = [lr for lr in LRS if results[(mode, lr)]["finite"] < len(SEEDS)]
        say(f"    {mode:10s} 3/3 finite AND >= 90% exact at lr {ok if ok else 'NONE'};   "
            f"non-finite on >= 1 seed at lr {bad if bad else 'NONE'}")
    b_ok = [lr for lr in LRS if results[("bounded", lr)]["finite"] == len(SEEDS)
            and min(results[("bounded", lr)]["exact"]) >= 0.90]
    d1 = any(lr <= 0.05 for lr in b_ok) and results[("bounded", 0.2)]["finite"] < len(SEEDS)
    say(f"    D1 as declared (bounded holds at lr <= 0.05 and fails at 0.2): "
        f"{'PASS' if d1 else 'FAIL'}")

    say("")
    say("  D2 MECHANISM: ||g|| per tensor after ONE real batch (seed 0), both representations")
    gu = grad_scale("unbounded")
    gb = grad_scale("bounded")
    keys = [k for k in gu if k in gb]
    say(f"    {'tensor':16s} {'unbounded':>12s} {'bounded':>12s} {'ratio':>8s}")
    for k in sorted(keys):
        ratio = gb[k] / gu[k] if gu[k] else float("nan")
        say(f"    {k:16s} {gu[k]:>12.5f} {gb[k]:>12.5f} {ratio:>8.3f}")
    emb_u = gu.get("emb", float("nan"))
    emb_b = gb.get("emb", float("nan"))
    say(f"    (embedding gradient ratio {emb_b/emb_u if emb_u else float('nan'):.3f}; the declared D2 range "
        f"was 2-6x)")
    d2 = 2.0 <= (emb_b / emb_u if emb_u else 0.0) <= 6.0
    say(f"    D2 as declared: {'PASS' if d2 else 'FAIL'}")
    say("")
    say("  NOT A CLAIM: nothing here says anything about chat accuracy, no default changed, and the gate")
    say("  verdict P44c FAIL stands exactly as worded.")
    say("=== DONE " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ===")

    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
