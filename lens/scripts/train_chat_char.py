"""train_chat_char.py — character-level talking model. The biggest remaining lever.

THE ARGUMENT, IN NUMBERS
------------------------
Word level, d=128, K=9 (chat run 1):
    embedding  V*D  = 12,000 * 128 = 1,536,000   92.9% of parameters
    answer     K*V  = 9 * 12,000   =   108,000    6.5%
    -> the answer layer picks 1 of 12,000 words (13.6 bits) through a 9-wide pipe

Character level, d=64, W=32, K=~48:
    embedding  V*D  = 432 * 64     =    27,648    ~19%
    concept    D*W*K= 2,048 * 48   =    98,304    ~67%
    answer     K*V  = 48 * 432     =    20,736    ~14%
    -> picks 1 of 432 characters (8.8 bits), and the concept layer is the MAJORITY of the model

That is the point: at character level the named concept layer stops being 0.6% of the network and
becomes the part that carries the model. There is also no <unk>, so the 1.92% of tokens currently
discarded is kept.

THE COSTS, STATED
-----------------
- 60.9M characters for the same text, so sequences are ~4.6x longer and each step sees less
  conversation per character than per word.
- Character-level models are slower to reach semantic coherence; they get spelling and punctuation
  right early and meaning late.
- The concept targets become substring matches on decoded text rather than exact token membership.
  Documented in chat_concepts_char.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import chat_concepts_char                                   # noqa: E402
import chat_data_char                                       # noqa: E402
import lens                                                 # noqa: E402
import lens.tensor as T                                     # noqa: E402
from lens import backend as B                               # noqa: E402
from lens.concepts import Concept                           # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "chat")
os.makedirs(OUT, exist_ok=True)


def log(m):
    print(m, flush=True)


class CharNet:
    """Same library architecture, character vocabulary. Thin wrapper like the word-level one."""

    def __init__(self, vocab, concepts, d=64, window=32, seed=0, lam_concept=0.3, device="gpu"):
        T.set_device(device)
        self._net = lens.Net(vocab=vocab, concepts=concepts, d=d, window=window, seed=seed,
                             lam_concept=lam_concept, lam_act=0.0, device=device)
        self.concepts = self._net.concepts
        self.K, self.V, self.D, self.W = self._net.K, self._net.V, self._net.D, self._net.W
        self.seed = int(seed)
        self.lam_concept, self.lam_act, self.lam_w = lam_concept, 0.0, 1.0

    def forward(self, X):
        out = self._net.forward(X)
        return out["concepts"], out["logits"]

    def loss(self, X, Y):
        return self._net.loss(X, Y)

    def forward_arrays(self, X):
        return self._net.forward_arrays(X)

    def parameters(self):
        return self._net.parameters()

    def n_params(self):
        return int(sum(p.data.size for p in self.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--lam-concept", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--out", default="char1")
    ap.add_argument("--max-windows", type=int, default=1200000)
    a = ap.parse_args()

    log("=" * 96)
    log("CHARACTER-LEVEL TALKING MODEL — the concept layer becomes the majority of the model")
    log("=" * 96)

    d = chat_data_char.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    vmap = {ch: i for i, ch in enumerate(vocab)}   # defined early: the concept placeholder ids need it
    Xtr, Ytr = chat_data_char.make_windows(ids, bounds, split, a.window, which=0)
    Xva, Yva = chat_data_char.make_windows(ids, bounds, split, a.window, which=1)
    if len(Xtr) > a.max_windows:
        sel = np.linspace(0, len(Xtr) - 1, a.max_windows).astype(np.int64)
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    log(f"  windows {len(Xtr):,} train / {len(Xva):,} val   vocab {len(vocab):,}   window {a.window}")

    cs, spec = chat_concepts_char.build(vocab, ids, None, windows=Xtr)
    log(f"  concepts: {spec['n_kept']} kept, {spec['n_dropped']} dropped of {spec['n_defined']} "
        f"(floor {spec['min_support']}, unit {spec['unit']})")
    for c in spec["kept"][:60]:
        log(f"     {c['name']:<22} support {c['support']:.4f}  {c['n_words']} words")
    for c in spec["dropped"]:
        log(f"     DROPPED {c['name']:<16} {c['reason']}")
    concepts = []
    for cdict in cs:
        # Concept requires token ids. At character level the DECLARED target is a regex over decoded
        # text, not token membership, so the ids here are only a placeholder (the character ids of
        # the concept's own words) and are NOT used to build targets. The audit below is computed
        # against the declared regex targets instead, because that is the thing the name claims.
        ph = sorted({vmap[ch] for w in cdict["words"] for ch in w if ch in vmap})
        c = Concept(cdict["name"], ph or [0], support=cdict["support"])
        c.pattern = cdict["pattern"]
        concepts.append(c)

    net = CharNet(len(vocab), concepts, d=a.d, window=a.window, seed=a.seed,
                  lam_concept=a.lam_concept, device=a.device)
    log(f"  model: {net.n_params():,} params  (d={a.d}, W={a.window}, K={net.K})")
    tpp = len(Xtr) / max(net.n_params(), 1)
    log(f"  characters per parameter: {tpp:.2f}  "
        f"({'well trained' if tpp > 50 else 'adequate' if tpp > 20 else 'UNDER-TRAINED'})")
    ratio = a.d * a.window / max(net.K, 1)
    log(f"  bottleneck ratio: {a.d*a.window} in / {net.K} out = {ratio:.0f}:1  "
        f"(the scaling win was 50:1; word-level run 1 was 114:1)")

    # precompute targets for train and val once -- regex over decoded text is slow per batch
    log("  building concept targets...")
    t0 = time.perf_counter()
    def tgts(X):
        texts = chat_concepts_char.decode_windows(vocab, X)
        return chat_concepts_char.targets_for(texts, cs)
    CH = 200000
    Ttr = np.zeros((len(Xtr), net.K), dtype=np.float32)
    for s in range(0, len(Xtr), CH):
        Ttr[s:s + CH] = tgts(Xtr[s:s + CH])
    Tva = np.zeros((len(Xva), net.K), dtype=np.float32)
    for s in range(0, len(Xva), CH):
        Tva[s:s + CH] = tgts(Xva[s:s + CH])
    log(f"  targets built in {time.perf_counter()-t0:.1f}s; "
        f"mean positive rate {Ttr.mean():.4f}")

    rng = np.random.default_rng(a.seed)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(Xtr))
        tot, ctot, nb, corr, seen = 0.0, 0.0, 0, 0, 0
        lr_now = a.lr * (1.0 - a.lr_decay * (ep - 1) / max(a.epochs - 1, 1)) if a.lr_decay > 0 \
            else a.lr
        for s in range(0, len(Xtr), a.batch):
            idx = order[s:s + a.batch]
            for p in net.parameters():
                p.grad = None
            out = net._net.forward(Xtr[idx])
            ce = T.softmax_ce(out["logits"], Ytr[idx])
            aux = _bce(out["concepts"], B.to_dev(Ttr[idx]), B.to_dev(net._net.pos_weight))
            total = T.add(ce, T.mul(aux, T.Tensor(np.asarray(a.lam_concept, np.float32))))
            total.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - lr_now * p.grad
            tot += float(B.to_cpu(ce.data))
            ctot += float(B.to_cpu(aux.data))
            nb += 1
            corr += int((B.to_cpu(out["logits"].data).argmax(axis=1) == Ytr[idx]).sum())
            seen += len(idx)
        vl, vc, vs = 0.0, 0, 0
        for s in range(0, len(Xva), a.batch):
            idx = np.arange(s, min(s + a.batch, len(Xva)))
            lg = net.forward_arrays(Xva[idx])["logits"]
            vl += float(B.to_cpu(T.softmax_ce(T.Tensor(B.to_dev(lg)), Yva[idx]).data)) * len(idx)
            vc += int((lg.argmax(axis=1) == Yva[idx]).sum())
            vs += len(idx)
        rec = {"epoch": ep, "train_loss": tot / max(nb, 1), "concept_loss": ctot / max(nb, 1),
               "train_acc": corr / max(seen, 1), "val_ppl": float(np.exp(vl / max(vs, 1))),
               "val_acc": vc / max(vs, 1), "lr": lr_now,
               "seconds": time.perf_counter() - t0}
        hist.append(rec)
        log(f"  ep {ep}/{a.epochs}  loss {rec['train_loss']:.4f} concept {rec['concept_loss']:.4f} "
            f"acc {rec['train_acc']*100:.2f}%   VAL ppl {rec['val_ppl']:.2f} "
            f"acc {rec['val_acc']*100:.2f}%  lr {lr_now:.3f}  ({rec['seconds']:.0f}s)")
        lens.save(net._net, os.path.join(OUT, f"{a.out}_ep{ep}.npz"))
        json.dump({"args": vars(a), "history": hist, "params": net.n_params(), "concepts": spec},
                  open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)

    # ---------------------------------------------------------------- make it talk
    log("")
    log("=" * 96)
    log("CAN IT TALK? — character-level replies")
    log("=" * 96)

    def generate(prompt, n_new=200, temp=0.8, greedy=False, seed=0):
        enc = [vmap.get(ch, 0) for ch in prompt.lower()]
        if len(enc) < a.window:
            enc = [0] * (a.window - len(enc)) + enc
        ctx = enc[-a.window:]
        out, r = [], np.random.default_rng(seed)
        for _ in range(n_new):
            lg = B.to_cpu(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"])[0]
            if greedy:
                nxt = int(lg.argmax())
            else:
                z = lg / temp
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                nxt = int(r.choice(len(p), p=p))
            out.append(vocab[nxt] if nxt < len(vocab) else " ")
            ctx = (ctx + [nxt])[-a.window:]
        return "".join(out)

    prompts = ["Human: hello, how are you?\n\nAssistant:",
               "Human: what is your name?\n\nAssistant:",
               "Human: can you help me learn to cook?\n\nAssistant:",
               "Human: i am feeling sad today.\n\nAssistant:",
               "Human: what is the capital of france?\n\nAssistant:"]
    replies = []
    for p in prompts:
        g = generate(p, greedy=True)
        s = generate(p, temp=0.8, seed=1)
        log(f"\n  PROMPT   {p.splitlines()[0]}")
        log(f"  GREEDY   {g[:200]}")
        log(f"  SAMPLED  {s[:200]}")
        replies.append({"prompt": p, "greedy": g, "sampled": s})

    log("")
    log("  --- concept fidelity audit (against the DECLARED regex targets) ---")
    aud = _audit_local(net, Xva[:3000], Tva[:3000], concepts)
    for c in aud["concepts"]:
        au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
        fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
        log(f"     {c['name']:<22} AUROC {au:>7}  fires {fr:>6}  {'PASS' if c['pass'] else 'FAIL'}")
    log(f"     -> {aud['n_pass']}/{aud['n_concepts']} pass   coverage {aud['coverage']*100:.2f}%")

    json.dump({"replies": replies, "audit": aud, "history": hist, "params": net.n_params(),
               "concepts": spec, "ratio": ratio},
              open(os.path.join(OUT, f"{a.out}_replies.json"), "w"), indent=1)
    log("")
    log("=" * 96)
    log(f"  final VAL ppl {hist[-1]['val_ppl']:.2f}  acc {hist[-1]['val_acc']*100:.2f}%   "
        f"concepts {aud['n_pass']}/{aud['n_concepts']}   ratio {ratio:.0f}:1")
    log("=" * 96)


def _audit_local(net, X, T, concepts, auroc_bar=0.55, fire_floor=0.02):
    """The fidelity audit for character-level concepts.

    `lens.audit` derives each concept's target from its token ids. At character level the declared
    target is a regex over decoded text, so this measures the activation against the DECLARED
    target matrix instead -- which is the thing the name actually claims. Same bars as everywhere
    else (AUROC 0.55, firing floor 0.02), not adjusted.
    """
    h = net.forward_arrays(X)["concepts"]
    h = np.asarray(h, dtype=np.float64)
    out = []
    for k, c in enumerate(concepts):
        t = np.asarray(T[:, k], dtype=np.float64)
        pos, neg = t > 0.5, t <= 0.5
        if pos.sum() < 5 or neg.sum() < 5:
            auroc = None
        else:
            # rank-based AUROC: P(activation of a positive > activation of a negative)
            order = np.argsort(h[:, k])
            ranks = np.empty(len(order), dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            npos, nneg = pos.sum(), neg.sum()
            auroc = float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        fires = float((h[:, k] > 0.5).mean())
        ok = (auroc is not None and auroc >= auroc_bar and fires >= fire_floor)
        out.append({"name": c.name, "auroc": auroc, "fires": fires, "pass": bool(ok)})
    npass = sum(1 for c in out if c["pass"])
    return {"concepts": out, "n_pass": npass, "n_concepts": len(out),
            "coverage": float(np.isfinite(h).all()), "auroc_bar": auroc_bar,
            "fire_floor": fire_floor}


def _bce(h, tgt, w, eps=1e-7):
    hd = h.data
    bce = -(w * tgt * B.xp.log(hd + eps) + (1.0 - tgt) * B.xp.log(1.0 - hd + eps))
    val = float(B.to_cpu(bce.mean()))
    return T.Tensor(B.to_dev(np.asarray(val, dtype=np.float32)),
                    requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                    _saved={"hd": hd, "tgt": tgt, "w": w, "n": bce.size})


if __name__ == "__main__":
    main()
