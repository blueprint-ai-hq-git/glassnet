"""orderblind_test.py -- IS THE ANSWER PATH ORDER-BLIND INSIDE ITS WINDOW?  (MEASURED, not asserted)

PREDICTIONS P117a-g WERE WRITTEN BEFORE THIS FILE EXISTED:
    lens/runs/ORDER-BLIND-PREDICTIONS-P117.txt

WHAT IT DOES, exactly
  For each checkpoint, take the trainer's OWN held-out windows -- chat_data.make_windows(ids, bounds,
  split, W=32, which=1, return_index=True) -- and keep the windows whose ONE scored target (the token at
  corpus position S+W, i.e. the token that follows the window; this is the "assistant by position" rule
  the stored column uses, dialogue_mixed_eval.asst_flags, verified to reproduce that column to |diff|
  0.0000 by P108a) is assistant text.  Then five conditions over the SAME windows and the SAME targets:
     A   natural       window as the corpus has it
     A2  natural again  (the instrument's own determinism control, P117f)
     B   permuted      positions 0..30 permuted by a seeded per-window permutation; position 31 unchanged
     B2  reversed      positions 0..30 reversed; position 31 unchanged
     C   constant-ctx  positions 0..30 set to token id 0 (<unk>); position 31 unchanged
  Condition A must reproduce each checkpoint's STORED val_asst_acc to 0.0000 points -- that is the hard
  control, because the stored column is exactly this quantity (one target per window, 58,191 of them).

WHAT IT NEVER DOES: no training, no generation, no GPU, no checkpoint written, no stored file edited.
CPU only. Read-only over every input.
"""
import json
import os
import sys
import time

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("EVAL_DEVICE", "cpu")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

import chat_data                                              # noqa: E402
import dialogue_mixed_eval as D                               # noqa: E402  (load() and asst_flags(): the
                                                              # P108a-verified walk and loader; nothing else
                                                              # in that module runs at import)

W = 32
BATCH = int(os.environ.get("OB_BATCH", "128"))
CAP = int(os.environ.get("OB_MAXWIN", "60000"))
DRIFT_FLOOR = 0.0739          # points by position, lens/runs/DRIFT-NUMBERS.txt
STORED_N_ASST = 58191         # every stored epoch row's val_asst_n

_DEFAULT = [
    ("TWIN ct_d256_full169_fix3 (3 ep, stored 35.2632%)",
     "lens/runs/chat/ct_d256_full169_fix3_best.npz"),
    ("H2L  h2l_noconcept_169_6ep (6 ep, stored 37.2480%)",
     "lens/runs/chat/h2l_noconcept_169_6ep_best.npz"),
]


def log(s=""):
    print(s, flush=True)


def ckpt_list():
    spec = os.environ.get("CKPT_LIST", "").strip()
    if not spec:
        return [(a, os.path.join(P, b)) for a, b in _DEFAULT]
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.basename(label), label
        if not os.path.isabs(path):
            path = os.path.join(P, path)
        out.append((label.strip(), path))
    return out


def stored_best(ckpt):
    """The stored val_asst_acc of the row this best checkpoint was written from (read-only).

    REFUSES to guess when the checkpoint is not named *_best.npz. This is the second appearance of
    [BUG-EP-NAME-HISTORY-MAPPING] (BUGS.md; first at dialogue_mixed_eval.py, GRADE-P109 section 9(a)):
    basename.replace("_best.npz", "_history.jsonl") is a NO-OP for a file named *_ep1.npz, so the
    history path becomes the .npz itself and the open() raises UnicodeDecodeError after printing two
    rows. The fix here is a NAME CHECK, not a fallback: an instrument that cannot find the stored row
    must say so, never invent one.
    """
    base = os.path.basename(ckpt)
    if not base.endswith("_best.npz"):
        return None, None, f"(no history mapping: {base} does not end in _best.npz -- refusing to guess)"
    hp = os.path.join(os.path.dirname(ckpt), base.replace("_best.npz", "_history.jsonl"))
    if not os.path.exists(hp) or not hp.endswith(".jsonl"):
        return None, None, hp
    rows = [json.loads(l) for l in open(hp) if l.strip()]
    if not rows:
        return None, None, hp
    b = max(rows, key=lambda r: float(r.get("val_asst_acc", 0.0)))
    return float(b["val_asst_acc"]), int(b.get("epoch", -1)), hp


def last_position_acc(net, X, y, idx):
    """Accuracy of the window's ONE scored target (Y[i] = the token at corpus position S+W).

    MEASURED SHAPE FACT (cost me one probe run): net.forward_arrays returns logits of shape (B, V), not
    (B, W, V) -- this architecture predicts ONE token per window, the token that FOLLOWS it, and
    chat_data.make_windows returns Y as 1-D (chat_data.py line 221: Y = ids_arr[starts + window]). So the
    'last position' of the window is the model's only position.
    """
    correct = 0
    top1 = []
    for s in range(0, len(idx), BATCH):
        sl = idx[s:s + BATCH]
        lg = net.forward_arrays(np.ascontiguousarray(X[sl]))["logits"]
        if lg.ndim != 2:
            raise SystemExit(f"REFUSING: logits came back {lg.shape}, expected (B, V); the instrument's "
                             f"target indexing would be silently wrong")
        pred = lg.argmax(axis=1)
        correct += int((pred == y[sl]).sum())
        top1.append(pred)
    return correct / max(len(idx), 1), np.concatenate(top1) if top1 else np.zeros(0, dtype=np.int64)


def main():
    t0 = time.time()
    log("=" * 108)
    log(f"ORDER-BLIND TEST (P117a-g)  {time.strftime('%F %H:%M')}   device=cpu batch={BATCH} cap={CAP}")
    log("predictions: lens/runs/ORDER-BLIND-PREDICTIONS-P117.txt (written before this file existed)")
    log(f"drift floor (by position) = {DRIFT_FLOOR} pts [lens/runs/DRIFT-NUMBERS.txt]; 3.0 pts = "
        f"{3.0 / DRIFT_FLOOR:.0f}x it")
    log("=" * 108)

    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    bnd = np.asarray(bounds, dtype=np.int64)
    Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)
    log(f"val windows (the trainer's own call, which=1): {len(Xva):,}")

    conv_of = np.searchsorted(bnd, Sva, side="right") - 1
    if not (np.all(bnd[conv_of] <= Sva) and np.all(Sva < bnd[conv_of + 1])):
        log("REFUSING: window->conversation mapping failed a boundary check")
        return 2

    mask = np.zeros(len(Sva), dtype=bool)
    for j in np.unique(conv_of):
        start = int(np.searchsorted(conv_of, j, side="left"))
        end = int(np.searchsorted(conv_of, j, side="right"))
        if end <= start:
            continue
        a0, b0 = int(bnd[j]), int(bnd[j + 1])
        toks = [vocab[int(t)] for t in np.asarray(ids[a0:b0])]
        fa = D.asst_flags(toks)
        mask[start:end] = fa[(Sva[start:end] - a0) + W]
    idx = np.flatnonzero(mask)
    if len(idx) > CAP:
        idx = idx[np.linspace(0, len(idx) - 1, CAP).astype(np.int64)]     # evenly spaced, not a prefix
    n = len(idx)
    log(f"assistant-by-position last-position targets: {int(mask.sum()):,}   scored here: {n:,}   "
        f"(stored val_asst_n = {STORED_N_ASST:,})")
    log(f"CONTROL n: scored set must be <= the stored column -> "
        f"{'PASS' if n <= STORED_N_ASST else 'FAIL (larger than the stored column)'}"
        f"{'  (FULL SET: n == stored n)' if n == STORED_N_ASST else ''}")
    log("")

    report = []
    for label, ckpt in ckpt_list():
        if not os.path.exists(ckpt):
            log(f"SKIPPED (no such file): {ckpt}")
            continue
        net, meta, z = D.load(ckpt)
        sacc, sep, hp = stored_best(ckpt)
        log("-" * 108)
        log(f"CHECKPOINT {label}")
        log(f"  file {os.path.relpath(ckpt, P)}  params {sum(int(np.prod(p.data.shape)) for p in net.parameters()):,}"
            f"  direct={bool(meta.get('direct'))}/{meta.get('direct_mode')}  attention={bool(meta.get('attention'))}")
        log(f"  stored val_asst_acc of the row this best file was written from: "
            f"{sacc if sacc is None else round(sacc * 100, 4)}% at epoch {sep}  [{os.path.relpath(hp, P)}]")

        X = np.asarray(Xva[idx])
        y = np.asarray(Yva[idx]).reshape(-1)
        if y.shape[0] != n:
            log(f"REFUSING: target vector is {y.shape}, expected ({n},)")
            return 2
        rng = np.random.default_rng(0)

        # ---- condition A and its determinism repeat A2
        accA, topA = last_position_acc(net, X, y, np.arange(n))
        accA2, _ = last_position_acc(net, X, y, np.arange(n))
        dA2 = abs(accA - accA2) * 100.0

        # ---- condition B: permute positions 0..30, keep position 31 (the token right before the target)
        XB = X.copy()
        for r in range(n):
            XB[r, :W - 1] = X[r, rng.permutation(W - 1)]
        accB, topB = last_position_acc(net, XB, y, np.arange(n))

        # ---- condition B2: reverse positions 0..30 (a structured permutation, no sampling noise)
        XR = X.copy()
        XR[:, :W - 1] = X[:, :W - 1][:, ::-1]
        accR, _ = last_position_acc(net, XR, y, np.arange(n))

        # ---- condition C: destroy the content of the context, keep the token right before the target
        XC = X.copy()
        XC[:, :W - 1] = 0
        accC, _ = last_position_acc(net, XC, y, np.arange(n))

        agree = float((topA == topB).mean()) * 100.0
        order_eff = (accA - accB) * 100.0
        rev_eff = (accA - accR) * 100.0
        content_eff = (accA - accC) * 100.0
        ratio = order_eff / content_eff if abs(content_eff) > 1e-12 else float("nan")

        log("")
        log(f"  A   natural        acc {accA*100:.4f}%   n={n:,}")
        log(f"  A2  natural again  acc {accA2*100:.4f}%   A-vs-A2 |diff| {dA2:.4f} pts   "
            f"-> determinism control {'PASS' if dA2 == 0.0 else 'FAIL'}")
        log(f"  B   permuted 0..30 acc {accB*100:.4f}%   order effect  A-B  = {order_eff:+.4f} pts "
            f"({abs(order_eff)/DRIFT_FLOOR:.1f}x floor)")
        log(f"  B2  reversed 0..30 acc {accR*100:.4f}%   order effect  A-B2 = {rev_eff:+.4f} pts "
            f"({abs(rev_eff)/DRIFT_FLOOR:.1f}x floor)")
        log(f"  C   constant ctx   acc {accC*100:.4f}%   content effect A-C = {content_eff:+.4f} pts "
            f"({abs(content_eff)/DRIFT_FLOOR:.1f}x floor)")
        log(f"  top-1 agreement A vs B: {agree:.2f}%")
        log(f"  RATIO order/content: random {ratio:.4f}   reversed {rev_eff/content_eff if abs(content_eff)>1e-12 else float('nan'):.4f}")

        ctl = "N/A (no stored row found)"
        if sacc is not None and n == STORED_N_ASST:
            d_ctl = abs(accA - sacc) * 100.0
            ctl = (f"PASS  stored {sacc*100:.4f}%  measured {accA*100:.4f}%  |diff| {d_ctl:.4f} pts  "
                   f"(n={n:,} both)" if d_ctl == 0.0 else
                   f"FAIL  stored {sacc*100:.4f}%  measured {accA*100:.4f}%  |diff| {d_ctl:.4f} pts")
        log(f"  CONTROL A vs the STORED column: {ctl}")

        report.append(dict(label=label, ckpt=os.path.relpath(ckpt, P), n=n, accA=accA, accA2=accA2,
                           accB=accB, accR=accR, accC=accC, agree=agree, order_eff=order_eff,
                           rev_eff=rev_eff, content_eff=content_eff, ratio=ratio,
                           stored=sacc, stored_ep=sep, dA2=dA2))

    log("")
    log("=" * 108)
    log("VERDICTS")
    log("=" * 108)
    if report:
        r0, r1 = report[0], report[-1]

        def _band(name, key, lo, hi, fmt):
            vals = [r[key] for r in report]
            ok = all((lo is None or lo <= v) and (hi is None or v <= hi) for v in vals)
            band = f"[{lo}, {hi}]" if lo is not None else f"<= {hi}"
            print(f"{name}: " + ", ".join(f"{r['label'].split()[0]}={fmt % v}" for r, v in zip(report, vals))
                  + f"   band {band} -> {'PASS' if ok else 'MISS/FAIL (outside the pre-written band)'}",
                  flush=True)
            return ok

        # NOTE: order_eff / rev_eff / content_eff are ALREADY in points (x100 in main) and agree is
        # already a percentage -- the first version of this block multiplied them by 100 AGAIN and printed
        # +1093.75 pts beside a +10.9375 line. That is the right-number-wrong-verdict class this project
        # keeps finding; the values below are the same objects the per-checkpoint table printed.
        _band("P117a order effect (random perm), pts     ", "order_eff", -0.5, 1.0, "%+.4f")
        _band("P117b content effect (constant ctx), pts   ", "content_eff", 5.0, 25.0, "%+.4f")
        _band("P117c ratio order/content                  ", "ratio", None, 0.15, "%.4f")
        _band("P117d top-1 agreement A vs B, %            ", "agree", 55.0, 90.0, "%.2f")
        d_ep = abs(r0["order_eff"] - r1["order_eff"])
        log(f"P117e order effect agrees across checkpoints within 2.0 pts: {r0['order_eff']:+.4f} vs "
            f"{r1['order_eff']:+.4f} -> |diff| {d_ep:.4f} pts -> {'PASS' if d_ep < 2.0 else 'FAIL'}")
        log(f"P117f determinism control on A: {[r['dA2'] for r in report]} pts -> "
            f"{'PASS' if all(r['dA2'] == 0.0 for r in report) else 'FAIL'}")
    log("")
    log("WHAT THIS IS AND IS NOT: a diagnostic on stored checkpoints (no training, no generation, no ship). "
        "Its numbers are NOT the 35.26%/37.25% column unless condition A reproduces it exactly, and nothing "
        "here is talk progress.")
    log(f"wall {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
