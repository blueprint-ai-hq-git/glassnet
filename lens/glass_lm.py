#!/usr/bin/env python3
"""glass_lm.py - train OUR OWN block on real text, end to end, with nothing hidden.

The pieces, all ours:
  * token embedding + LEARNED position embedding  (tied output: logits = h @ E.T, so no second table)
  * N GlassBlocks (our attention + MLP + LayerNorm, gradients certified in lens/glass_grad.py)
  * our own momentum SGD, written here
  * a held-out score AND a matched constant baseline, because an accuracy with no baseline is not a result

GATE (written before the run): held-out accuracy must beat the matched constant by more than 0.
Falsifier: if held-out accuracy is at or below the constant, the block does not learn and the cause must be found before any
architecture work continues - the numbers are printed for every checkpointed step, and the constant is printed beside them.

Usage:
  PYTHONPATH=. .venv-gpu/bin/python3 -u lens/glass_lm.py --steps 400 --d 64 --layers 2 --T 32 --windows 20000
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", ".")
sys.path.insert(0, ROOT)
# chat_data lives in lens/scripts; ROOT/lens itself is deliberately NOT added because it shadows the real `cuda` package
# (measured problem in this project), so only the scripts subdirectory goes on the path.
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.environ.setdefault("CHAT_DATA_DIR", "${GLASSNET_ROOT:-.}/data/chat/prepared_mix_ctrl_honest")

from lens.glass_block import layernorm
from lens.glass_grad import forward_record, backward_record, ln_backward


class Momentum:
    """Our own optimizer: momentum SGD, or the SAME rule with a per-tensor adaptive scale.

    WHY THE ADAPTIVE FORM EXISTS (2026-11-20 17:25, his "when we feed the data it automatically aligns itself"): in the plain form
    ONE hand-set learning rate multiplies every tensor, so a value that suits the blocks wrecks the embedding and vice versa. In the
    adaptive form each tensor divides its own gradient by its own measured RMS scale:

        s_k <- b*s_k + (1-b)*sqrt(mean(g_k^2))        (the tensor's own gradient scale, measured from the data)
        v_k <- m*v_k - lr * g_k / (s_k + eps)
        p_k <- p_k + v_k

    The global `lr` then acts as a dimensionless scale rather than a physical step size, which is the property being tested: a knob
    that stops mattering. Nothing is borrowed - this is our own rule, written out.
    """

    def __init__(self, params, lr=0.01, momentum=0.9, adaptive=False, beta=0.9, eps=1e-8):
        self.params, self.lr, self.momentum = params, lr, momentum
        self.adaptive, self.beta, self.eps = adaptive, beta, eps
        self.vel = {k: np.zeros_like(v) for k, v in params.items()}
        self.scale = {k: (None if adaptive == "elem" else 0.0) for k in params}

    def step(self, grads):
        for k, v in self.params.items():
            g = grads.get(k)
            if g is None:
                continue
            if self.adaptive == "elem":
                # PER-ELEMENT scale (the pre-registered one retry after the per-tensor scalar failed at lr 0.50).
                if self.scale[k] is None or getattr(self.scale[k], "shape", None) != g.shape:
                    self.scale[k] = np.zeros_like(g)
                self.scale[k] = self.beta * self.scale[k] + (1.0 - self.beta) * np.abs(g)
                denom = self.scale[k] + self.eps
                self.vel[k] = self.momentum * self.vel[k] - self.lr * g / denom
            elif self.adaptive:
                rms = float(np.sqrt(np.mean(np.square(g))))
                self.scale[k] = self.beta * self.scale[k] + (1.0 - self.beta) * rms
                denom = self.scale[k] + self.eps
                self.vel[k] = self.momentum * self.vel[k] - self.lr * g / denom
            else:
                self.vel[k] = self.momentum * self.vel[k] - self.lr * g
            v += self.vel[k]


class GlassLM:
    def __init__(self, vocab, d=64, layers=2, n_heads=4, d_mlp=128, T=32, seed=0, emb_rank=0):
        """emb_rank > 0 replaces the token table with a FACTORED embedding E = A @ B (rank r).

        WHY (2026-11-20 16:10): the first real run showed E is 768,128 of 836,352 parameters = 92%, the same lookup-table
        dominance the anatomy scan found in the old architecture. A 92%-table model can only memorise. With V=12,002, d=64:
        full table 768,128 params; rank 16 gives A 192,032 + B 1,024 = 193,056, i.e. 3.98x fewer embedding parameters for a
        representation that must generalise across words instead of storing each row free.
        """
        self.emb_rank = int(emb_rank)
        self.V, self.d, self.L, self.H, self.d_mlp, self.T = vocab, d, layers, n_heads, d_mlp, T
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(d)
        self.p = {"P": (rng.standard_normal((T, d)) * 0.01).astype(np.float64)}
        if self.emb_rank > 0:
            r = self.emb_rank
            self.p["A"] = (rng.standard_normal((vocab, r)) * 0.02).astype(np.float64)
            self.p["B"] = (rng.standard_normal((r, d)) * (1.0 / math.sqrt(r))).astype(np.float64)
        else:
            self.p["E"] = (rng.standard_normal((vocab, d)) * 0.02).astype(np.float64)
        for l in range(layers):
            for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "ln1_scale", "ln1_shift", "ln2_scale", "ln2_shift"):
                if name in ("W1",):
                    arr = (rng.standard_normal((d, d_mlp)) * s)
                elif name in ("W2",):
                    arr = (rng.standard_normal((d_mlp, d)) * s)
                elif name in ("ln1_scale", "ln2_scale"):
                    arr = np.ones(d)
                elif name in ("ln1_shift", "ln2_shift"):
                    arr = np.zeros(d)
                else:
                    arr = (rng.standard_normal((d, d)) * s)
                self.p["%s_%d" % (name, l)] = arr.astype(np.float64)
        # BUG FIXED (2026-11-20 15:20): these used to live outside self.p, so the optimizer never stepped them - two parameter
        # vectors that looked trainable and were not. They are now in self.p like everything else, which also makes them visible
        # to n_params() and to the anatomy inventory.
        self.p["lnf_scale"] = np.ones(d)
        self.p["lnf_shift"] = np.zeros(d)

    def block_params(self, l):
        return {n: self.p["%s_%d" % (n, l)] for n in ("Wq", "Wk", "Wv", "Wo", "W1", "W2",
                                                      "ln1_scale", "ln1_shift", "ln2_scale", "ln2_shift")}

    def n_params(self):
        return sum(int(v.size) for k, v in self.p.items() if k != "__n")

    def forward(self, toks, target_mask=None):
        """toks: (B, T) int. Returns logits (B, T, V) and a record for backward."""
        B, T = toks.shape
        if self.emb_rank > 0:
            x = (self.p["A"][toks] @ self.p["B"]) + self.p["P"][:T][None, :, :]   # factored lookup
        else:
            x = self.p["E"][toks] + self.p["P"][:T][None, :, :]                  # (B,T,d)
        recs = []
        h = x.reshape(B * T, self.d)
        for l in range(self.L):
            h, rec = forward_record(h, self.block_params(l), self.H, norm=True)
            recs.append(rec)
        hn = layernorm(h, self.p["lnf_scale"], self.p["lnf_shift"])
        if self.emb_rank > 0:
            u = hn @ self.p["B"].T                                   # (BT, r) intermediate, kept for backward
            logits = u @ self.p["A"].T                               # tied through the factored form
        else:
            u = None
            logits = hn @ self.p["E"].T                              # tied output: same table, no extra matrix
        return logits.reshape(B, T, self.V), (toks, recs, hn, h, u)

    def eval_loss(self, toks, tgts, unk=-1):
        """Forward only: the loss right now. Used by the gain rule to measure how the loss RESPONDS to a step."""
        logits, _ = self.forward(toks)
        V = self.V
        lg = logits.reshape(-1, V)
        tg = tgts.reshape(-1)
        mask = tg != unk
        if int(mask.sum()) == 0:
            return None
        z = lg - lg.max(axis=1, keepdims=True)
        ez = np.exp(z)
        pr = ez / ez.sum(axis=1, keepdims=True)
        return float(-np.log(pr[np.arange(len(tg)), tg] + 1e-12)[mask].mean())

    def loss_and_grads(self, toks, tgts, unk=-1):
        """Cross entropy with <unk> targets excluded, plus gradients for every named tensor."""
        B, T = toks.shape
        logits, (toks_r, recs, hn, h_flat, u) = self.forward(toks)
        V = self.V
        lg = logits.reshape(-1, V)
        tg = tgts.reshape(-1)
        mask = (tg != unk)
        n = int(mask.sum())
        if n == 0:
            return None, None, 0
        z = lg - lg.max(axis=1, keepdims=True)
        ez = np.exp(z)
        pr = ez / ez.sum(axis=1, keepdims=True)
        loss = float(-np.log(pr[np.arange(len(tg)), tg] + 1e-12)[mask].mean())
        acc = float((pr.argmax(axis=1)[mask] == tg[mask]).mean())

        dlg = pr.copy()
        dlg[mask, tg[mask]] -= 1.0
        dlg[~mask] = 0.0
        dlg /= max(n, 1)

        grads = {}
        if self.emb_rank > 0:
            # logits = (hn @ B.T) @ A.T   ->   dA = dlg.T @ u ;  du = dlg @ A ;  dB = du.T @ hn ;  dhn = du @ B
            grads["A"] = dlg.T @ u
            du = dlg @ self.p["A"]
            grads["B"] = du.T @ hn
            dhn = du @ self.p["B"]
        else:
            # tied output: logits = hn @ E.T
            grads["E"] = dlg.T @ hn                                   # (V,d)
            dhn = dlg @ self.p["E"]                                   # (BT,d)
        # final layernorm
        dh, g_ln_f_scale, g_ln_f_shift = ln_backward(dhn, h_flat, self.p["lnf_scale"])
        grads["lnf_scale"] = g_ln_f_scale
        grads["lnf_shift"] = g_ln_f_shift

        # blocks in reverse
        for l in reversed(range(self.L)):
            dh, g = backward_record(dh, recs[l], self.block_params(l), self.H)
            for k, v in g.items():
                grads["%s_%d" % (k, l)] = v

        # embeddings: the lookup path adds to the same tensors the output path used
        d_emb = dh.reshape(-1, self.d)
        if self.emb_rank > 0:
            gA = np.zeros_like(self.p["A"])
            np.add.at(gA, toks.reshape(-1), d_emb @ self.p["B"].T)    # (BT,r) scattered per token row
            grads["A"] = grads["A"] + gA
            grads["B"] = grads["B"] + self.p["A"][toks.reshape(-1)].T @ d_emb   # (r,d), B is shared by every token
        else:
            dE_lookup = np.zeros_like(self.p["E"])
            np.add.at(dE_lookup, toks.reshape(-1), d_emb)
            grads["E"] = grads["E"] + dE_lookup
        dP = np.zeros_like(self.p["P"])
        dP[:T] = dh.reshape(B, T, self.d).sum(axis=0)
        grads["P"] = dP
        return loss, acc, grads

    def apply(self, grads, opt):
        # every gradient key must correspond to a real tensor in self.p - refuse to silently drop one
        missing = [k for k in grads if k not in self.p]
        if missing:
            raise SystemExit("gradients with no matching parameter: %s" % missing)
        opt.step(grads)


def gradcheck_factored(eps=1e-6, n_sample=12, seed=5):
    """Numerical gradient check for the FACTORED embedding path (A, B) plus P, the final LayerNorm and one block matrix.

    A factored table has a different backward path from a plain table (two matrices, one scatter and one shared-matrix sum), so it
    gets its own finite-difference check rather than inheriting trust from the full-table version.
    """
    rng = np.random.default_rng(seed)
    V, d, T, r = 40, 8, 3, 3
    lm = GlassLM(V, d=d, layers=1, n_heads=2, d_mlp=12, T=T, seed=seed, emb_rank=r)
    toks = rng.integers(0, V, size=(2, T))
    tgts = rng.integers(0, V, size=(2, T))
    loss, acc, g = lm.loss_and_grads(toks, tgts, unk=-1)
    print("=" * 96)
    print("FACTORED-EMBEDDING GRADIENT CHECK (V=%d, d=%d, rank=%d, T=%d, float64)" % (V, d, r, T))
    print("  loss %.12f" % loss)
    worst = 0.0
    for name in ("A", "B", "P", "lnf_scale", "Wq_0", "W1_0"):
        arr = lm.p[name]
        flat_idx = rng.choice(arr.size, size=min(n_sample, arr.size), replace=False)
        num, ana = [], []
        for fi in flat_idx:
            idx = np.unravel_index(int(fi), arr.shape)
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _, _ = lm.loss_and_grads(toks, tgts, unk=-1)
            arr[idx] = orig - eps
            lm2, _, _ = lm.loss_and_grads(toks, tgts, unk=-1)
            arr[idx] = orig
            num.append((lp - lm2) / (2 * eps))
            ana.append(g[name][idx])
        num, ana = np.array(num), np.array(ana)
        abs_err = np.abs(ana - num)
        # A RELATIVE-ONLY TEST IS MISLEADING HERE (2026-11-20 16:20): near-zero gradient entries divide by ~0 and report large
        # "relative" error on an absolute error of 1e-9, which is finite-difference truncation, not a wrong derivative. Both are
        # printed, and the verdict uses the ABSOLUTE error with a floor - the same standard the block's own check in
        # lens/glass_grad.py used (absolute errors were ~1e-9 there).
        den = np.maximum(np.abs(ana) + np.abs(num), 1e-8)
        rel = float(np.max(abs_err / den))
        ab = float(np.max(abs_err))
        worst = max(worst, ab)
        flag = "" if ab < 1e-7 else "   <<< OVER TOLERANCE"
        print("  %-10s %2d entries   max abs %.3e   max rel (floored denom) %.3e   |grad| typ %.2e%s"
              % (name, len(flat_idx), ab, rel, float(np.median(np.abs(ana))), flag))
    ok = worst < 1e-7
    print("  WORST ABSOLUTE %.3e  (tolerance 1e-7) -> %s" % (worst, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def train_gain(lm, a, Xtr, Xva, UNK, V, const_acc):
    """A GAIN RULE: the step size is not hand-set, it is chosen by measuring what the step did to the loss.

    THE DIAGNOSIS THIS ANSWERS (2026-11-20 18:00): both normalisation variants failed because dividing a gradient by its own
    magnitude discards HOW MUCH the loss cares about each coordinate - at per-element grain every coordinate gets the same size step
    and the model sinks (loss ~27.6 at every lr tested). This rule keeps one global direction and one scalar step, and picks that
    scalar by trial: take the step, measure the loss, keep it if the loss fell, otherwise halve and retry. The knob therefore stops
    being a physical step size and becomes only an initial guess.
    """
    rng = np.random.default_rng(a.seed + 1)
    step = float(a.lr)
    t0 = time.time()
    for it in range(1, a.steps + 1):
        idx = rng.integers(0, len(Xtr), size=a.batch)
        chunk = Xtr[idx]
        toks, tgts = chunk[:, :a.T], chunk[:, 1:]
        loss, acc, grads = lm.loss_and_grads(toks, tgts, unk=UNK)
        if grads is None or not np.isfinite(loss):
            print("ARM DIVERGED at step %d" % it); print("OUTCOME: DIVERGED"); return 3
        gnorm = float(np.sqrt(sum(float(np.sum(np.square(g))) for g in grads.values())))
        if not np.isfinite(gnorm) or gnorm <= 0:
            print("ARM DIVERGED at step %d (zero/non-finite gradient norm)" % it); print("OUTCOME: DIVERGED"); return 3
        accepted_at = -1
        trial_step = step
        for trial in range(4):
            for k, p_ in lm.p.items():
                g = grads.get(k)
                if g is not None:
                    p_ -= trial_step * g / gnorm          # global direction, scalar step
            new_loss = lm.eval_loss(toks, tgts, unk=UNK)
            if new_loss is not None and np.isfinite(new_loss) and new_loss < loss:
                accepted_at = trial
                break
            for k, p_ in lm.p.items():                    # revert and try a smaller step
                g = grads.get(k)
                if g is not None:
                    p_ += trial_step * g / gnorm
            trial_step *= 0.5
        if accepted_at < 0:
            for k, p_ in lm.p.items():
                g = grads.get(k)
                if g is not None:
                    p_ += step * g / gnorm
            step *= 0.5
        elif accepted_at == 0:
            step *= 1.3
        else:
            step *= 0.7
        if it % a.log_every == 0 or it == 1:
            print("step %4d/%d  loss %.4f  train acc %.2f%%  step_size %.5f  accept@%d  %.1fs"
                  % (it, a.steps, loss, 100 * acc, step, accepted_at, time.time() - t0), flush=True)
    losses, accs = [], []
    for i in range(0, len(Xva), 32):
        c = Xva[i:i + 32]
        l, ac, _ = lm.loss_and_grads(c[:, :a.T], c[:, 1:], unk=UNK)
        if l is not None and np.isfinite(l):
            losses.append(l); accs.append(ac)
    hl, ha = float(np.mean(losses)), float(np.mean(accs))
    print("\nHELD-OUT  loss %.4f  acc %.2f%%   |  matched constant %.2f%%   delta %+.2f  -> %s"
          % (hl, 100 * ha, 100 * const_acc, 100 * (ha - const_acc),
             "BEATS constant" if ha > const_acc else "AT OR BELOW constant - GATE FAILED"))
    print("OUTCOME: %s" % ("BEATS_CONSTANT" if ha > const_acc else "BELOW_CONSTANT"))
    return 0 if ha > const_acc else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-mlp", type=int, default=128)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--emb-rank", type=int, default=0,
                    help="0 = full token table; >0 = FACTORED embedding E = A @ B of that rank")
    ap.add_argument("--opt", choices=("momentum", "adaptive", "adaptive_elem", "gain"), default="momentum",
                    help="momentum = one hand-set lr for every tensor; adaptive = each tensor divides its own gradient by its own "
                         "measured RMS scale, so the lr becomes a dimensionless knob instead of a physical step size")
    ap.add_argument("--gradcheck", action="store_true",
                    help="numerically check the FACTORED embedding gradients on a tiny model, then exit")
    a = ap.parse_args()

    if a.gradcheck:
        return gradcheck_factored()

    import chat_data
    t0 = time.time()
    dat = chat_data.build()
    vocab, ids, bounds, split = dat["vocab"], dat["ids"], dat["bounds"], dat["split"]
    UNK = vocab.get("<unk>", 0) if isinstance(vocab, dict) else 0
    Xtr = chat_data.make_windows(ids, bounds, split, a.T + 1, which=0, max_windows=a.windows, seed=a.seed)[0]
    Xva = chat_data.make_windows(ids, bounds, split, a.T + 1, which=1, max_windows=max(a.windows // 5, 200), seed=99)[0]
    V = len(vocab) if not isinstance(vocab, int) else vocab
    print("data: train %s windows, held-out %s, vocab %d, load %.1fs"
          % (format(len(Xtr), ","), format(len(Xva), ","), V, time.time() - t0))

    # MATCHED CONSTANT BASELINE. BUG FIXED (2026-11-20 15:25): the first version counted EVERY target including <unk> and picked
    # token 0, which never occurs as a scored target - so the baseline came out 0.00% and "BEATS constant by +4.91" was worthless,
    # because a constant that scores zero can be beaten by any model that predicts at all. The baseline is now built the way the
    # rest of this project builds it: the most frequent target ON TRAIN AFTER EXCLUDING <unk>, scored on the same <unk>-excluded
    # held-out targets the model is scored on. If the two denominators differ, the comparison is not matched.
    tr_tg = Xtr[:, 1:].reshape(-1)
    tr_tg = tr_tg[tr_tg != UNK]
    const_tok = int(np.bincount(tr_tg, minlength=V).argmax())
    va_tg = Xva[:, 1:].reshape(-1)
    va_mask = va_tg != UNK
    const_acc = float((va_tg[va_mask] == const_tok).mean())
    print("baseline construction: %s scored targets on train (after <unk>), most frequent = token %d, %s held-out scored targets"
          % (format(len(tr_tg), ","), const_tok, format(int(va_mask.sum()), ",")))
    print("matched constant baseline (always predict token %d): held-out %.2f%%" % (const_tok, 100 * const_acc))

    lm = GlassLM(V, d=a.d, layers=a.layers, n_heads=a.heads, d_mlp=a.d_mlp, T=a.T, seed=a.seed, emb_rank=a.emb_rank)
    if a.opt == "gain":
        print("optimiser: gain rule  (direction normalised once; step size chosen from the MEASURED change in loss)")
        print("  initial step %.4f, halving up to 4 times per step, growing 1.3x after a first-try accept" % a.lr)
        return train_gain(lm, a, Xtr, Xva, UNK, V, const_acc)

    opt = Momentum(lm.p, lr=a.lr, momentum=0.9, adaptive=(None if a.opt == "momentum" else ("elem" if a.opt == "adaptive_elem" else True)))
    _labels = {"momentum": "  (one lr for every tensor)",
               "adaptive": "  (per-tensor scale, measured from the data)",
               "adaptive_elem": "  (per-ELEMENT scale, measured from the data)"}
    print("optimiser: %s%s" % (a.opt, _labels.get(a.opt, "")))
    emb_desc = ("factored A %s + B %s = %s" % (format(lm.p["A"].size, ","), format(lm.p["B"].size, ","),
                                                format(lm.p["A"].size + lm.p["B"].size, ","))
                if a.emb_rank > 0 else ("full table E %s" % format(lm.p["E"].size, ",")))
    emb_params = (lm.p["A"].size + lm.p["B"].size) if a.emb_rank > 0 else lm.p["E"].size
    print("model: %s parameters in %d named tensors (%s + pos %s + %d blocks)"
          % (format(lm.n_params(), ","), len(lm.p), emb_desc, format(lm.p["P"].size, ","), a.layers))
    print("OUTPUT-TABLE SHARE: %s of %s = %.1f%% of the whole model"
          % (format(int(emb_params), ","), format(lm.n_params(), ","), 100.0 * emb_params / lm.n_params()))

    rng = np.random.default_rng(a.seed + 1)
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, len(Xtr), size=a.batch)
        chunk = Xtr[idx]
        loss, acc, grads = lm.loss_and_grads(chunk[:, :a.T], chunk[:, 1:], unk=UNK)
        if grads is None:
            continue
        lm.apply(grads, opt)
        if not np.isfinite(loss):
            # DIVERGENCE IS A RESULT, NOT A CRASH (2026-11-20 17:40). The lr=0.50 arm of the alignment test went to nan at step 300
            # and my runner treated exit 1 as infrastructure failure, aborting the two arms that were the actual test. A diverged
            # arm is the evidence a too-high learning rate produces, so it now stops early and reports itself distinctly.
            print("\nARM DIVERGED at step %d: loss became non-finite (this is a RESULT, not a crash)" % step, flush=True)
            print("HELD-OUT  not scored - the arm diverged before evaluation")
            print("OUTCOME: DIVERGED")
            return 3
        if step % a.log_every == 0 or step == 1:
            print("step %4d/%d  loss %.4f  train acc %.2f%%  %.1fs"
                  % (step, a.steps, loss, 100 * acc, time.time() - t0), flush=True)

    # held-out evaluation
    losses, accs = [], []
    for i in range(0, len(Xva), 32):
        chunk = Xva[i:i + 32]
        l, ac, _ = lm.loss_and_grads(chunk[:, :a.T], chunk[:, 1:], unk=UNK)
        if l is not None:
            losses.append(l)
            accs.append(ac)
    hl, ha = float(np.mean(losses)), float(np.mean(accs))
    if not (np.isfinite(hl) and np.isfinite(ha)):
        print("\nARM DIVERGED before evaluation: held-out loss/acc non-finite (a RESULT, not a crash)")
        print("OUTCOME: DIVERGED")
        return 3
    print("\nHELD-OUT  loss %.4f  acc %.2f%%   |  matched constant %.2f%%   delta %+.2f  -> %s"
          % (hl, 100 * ha, 100 * const_acc, 100 * (ha - const_acc),
             "BEATS constant" if ha > const_acc else "AT OR BELOW constant - GATE FAILED"))
    # OUTCOME is printed so a runner can distinguish "this arm produced a bad result" (data) from "this arm crashed" (no line).
    print("OUTCOME: %s" % ("BEATS_CONSTANT" if ha > const_acc else "BELOW_CONSTANT"))
    return 0 if ha > const_acc else 1


if __name__ == "__main__":
    raise SystemExit(main())
