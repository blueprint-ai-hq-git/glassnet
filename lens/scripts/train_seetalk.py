"""train_seetalk.py — the model that SEES and TALKS: image in, caption out, word by word.

WHAT EXISTS AND WHAT THIS ADDS
  The vqa models pick 1 answer from a fixed list. The chat model writes words but is blind.
  THIS joins them: the image front-end feeds the named concept layer, and the concept layer plus
  the last W caption words feed an output layer that predicts the NEXT caption word. One model:
  it sees, and it talks.

THE DESIGN (one new thing at a time; every part copied from an already-verified pattern)
  image (28x28) -> PatchEmbed -> 16 projected patches -> reshape to (B, P*D) -> concept Dense
  -> sigmoid [h, K]; previous W caption words -> cap_emb -> (B, W*D) [cxt];
  concat([h, cxt]) -> cap_out Dense -> logits over the caption vocabulary.

  * reshape is the tensor op whose backward was numerically verified (gradcheck_vision_fixed).
  * the concept aux BCE is the EXACT pattern from lens/vision.py VisionNet.loss (verified there).
  * the ONLY path from pixels to words is the named concept layer: explain() can therefore show
    exactly which named concepts drove each generated word. The image cannot influence the text
    except through named, audited neurons. That is the transparency requirement.

  NOTE (honest limitation): this first build projects all 16 patches but pools nothing — the
  concept layer sees the whole (16 x D) flattened patch grid, so positions survive. Caption data
  is GENERATED from the Fashion-MNIST class names (MIT); no scraping, no outside model.

PRE-REGISTERED GATES (fixed BEFORE running, binary, no rewording):
  g1  same caption prefix + 10 DIFFERENT images -> predicted next word takes >=5 distinct values.
      Fewer means the image path is decorative. FAIL.
  g2  train loss falls from first to last epoch.
  g3  explain residual < 1e-5: hand-recomputed logits equal the engine's logits.
  g4  val word-accuracy beats a TEXT-ONLY control (same data, same budget, ZERO image input)
      by more than +0.5 absolute. If the image adds nothing measurable, the point fails.

TINY RUNG FIRST: d=32, ~4k images, few epochs, CPU. Big only after these four pass.
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import image_concepts                                    # noqa: E402
import lens.tensor as T                                  # noqa: E402
import vl_data                                           # noqa: E402
from lens import backend as B                            # noqa: E402
from lens import backend                                 # noqa: E402  (code uses backend.to_dev/to_cpu)
from lens.model import Dense                             # noqa: E402
from lens.vision import PatchEmbed                       # noqa: E402


def log(msg):
    print(msg, flush=True)


class SeeTalkNet:
    """Image -> named concepts -> next caption word. The ONLY text path is through the concepts."""

    def __init__(self, concepts, vocab_size, d=32, window=6, seed=0, lam_concept=0.3, patch=7,
                 decorr=0.0,
                 image=28, concept_direct=False, concept_mirror=False):
        self.concepts = list(concepts)
        self.K = len(self.concepts)
        self.V, self.D, self.W = int(vocab_size), int(d), int(window)
        self.lam_concept = float(lam_concept)
        self.decorr = float(decorr)   # OPTIONAL rivalry penalty on the concept activations
        self.image = int(image)
        rng = np.random.default_rng(seed)
        self.front = PatchEmbed(patch=patch, d=d, rng=rng, image=image)
        self.P = self.front.P
        pw = []
        for c in self.concepts:
            s = max(float(c.support), 1e-6) if c.support is not None else 0.5
            pw.append((1.0 - s) / s)
        self.pos_weight = np.asarray(pw, dtype=np.float32)
        # concept_direct: the concept layer also reads the RAW PATCHES (pixels in [0,1]) in
        # addition to the shared projection. The information-bottleneck hypothesis: the one
        # honest-margin shortfall (left_right_symmetric_rel, margin saturating at ~0.05) sits in
        # a concept whose signal is left/right mass symmetry -- subtle across the shared proj.
        # With the bypass the neuron can read pixel structure directly, and its explanation is
        # STILL a plain weighted sum (projection features + raw pixels).
        self.concept_direct = bool(concept_direct)
        # concept_mirror: add the |I - mirror(I)| map as an INPUT BLOCK, available to every concept
        # neuron equally. WHY (measured, not guessed): the one concept that misses its honesty
        # margin, left_right_symmetric_rel, is defined as mean|I - mirror(I)| / mean(I) -- a
        # PAIRWISE, non-local quantity. Every other concept in the set is a local aggregate
        # (means of halves/quadrants, thresholds, edge magnitudes). The concept layer is a linear
        # map over local features, and no linear map can form |I - mirror(I)| regardless of width
        # -- which is exactly why capacity (d160, d224) and direct raw-pixel access both changed
        # nothing. This block is raw material, not the formula's answer: the neuron still has to
        # combine it with brightness itself.
        self.concept_mirror = bool(concept_mirror)
        self.concept_in = (self.P * self.D
                           + (self.P * self.front.patch_dim if concept_direct else 0)
                           + (self.front.channels * self.image * self.image
                              if concept_mirror else 0))
        self.concept = Dense(self.concept_in, self.K, rng, name="concept")
        self.cap_emb = T.Tensor(backend.to_dev(rng.standard_normal((self.V, self.D))
                                              .astype(np.float32) * self.D ** -0.5),
                                requires_grad=True, name="cap_emb")
        self.cap_out = Dense(self.K + self.D * self.W, self.V, rng, name="cap_out")

    def forward(self, X, cap_ids):
        """X (B,784) images; cap_ids (B,W) the PREVIOUS caption words. Logits = next word."""
        Pp = self.front.patches(X)                          # (B, P, patch_dim) numpy
        xin = T.Tensor(backend.to_dev(np.ascontiguousarray(
            Pp.reshape(-1, Pp.shape[-1]), dtype=np.float32)), requires_grad=False)
        z = self.front.proj(xin)                            # (B*P, D) Tensor
        zflat = T.reshape(z, (Pp.shape[0], self.P * self.D))
        blocks = [zflat]
        if self.concept_direct:
            blocks.append(T.Tensor(backend.to_dev(np.ascontiguousarray(
                Pp.reshape(Pp.shape[0], -1), dtype=np.float32)), requires_grad=False))
        if self.concept_mirror:
            blocks.append(T.Tensor(backend.to_dev(np.ascontiguousarray(
                self._mirror_map(X), dtype=np.float32)), requires_grad=False))
        if len(blocks) > 1:
            zflat = T.concat(blocks, axis=1)
        h = T.sigmoid(self.concept(zflat))
        cxt = T.embedding(self.cap_emb, np.asarray(cap_ids), self.W, self.D)
        inp = T.concat([h, cxt], axis=1)
        logits = self.cap_out(inp)
        return {"concepts": h, "logits": logits}

    def forward_arrays(self, X, cap_ids):
        out = self.forward(X, cap_ids)
        return {k: backend.to_cpu(v.data) for k, v in out.items()}

    def _mirror_map(self, X):
        """|I - mirror(I)| flattened per image (the nonlinear basis for a pairwise formula)."""
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim == 2:
            Xa = Xa.reshape(-1, self.front.channels, self.image, self.image)
        return np.abs(Xa - Xa[..., ::-1]).reshape(len(Xa), -1)

    def loss(self, X, cap_ids, y, ce_weight=1.0):
        """ce_weight=0.0 runs the CONCEPT-ONLY phase: the answer loss is switched off so the
        concept neurons are trained purely to match their own formulas (the honesty margins),
        with no competition from the classification objective."""
        out = self.forward(X, cap_ids)
        ce = T.softmax_ce(out["logits"], y)
        parts = {"ce": float(backend.to_cpu(ce.data))}
        if ce_weight == 0.0:
            total = T.mul(ce, T.Tensor(np.asarray(0.0, dtype=T.current_dtype())))
        else:
            total = ce
        if self.lam_concept > 0:
            # EXACTLY vision.py's verified concept-BCE pattern, not a second implementation
            h = out["concepts"]
            hd = backend.to_cpu(h.data)
            tgt = np.stack([c.labels(np.asarray(X, dtype=np.float32)) for c in self.concepts],
                           axis=1).astype(T.current_dtype())
            w = self.pos_weight[None, :]
            eps = 1e-7
            bce = -(w * tgt * np.log(hd + eps) + (1.0 - tgt) * np.log(1.0 - hd + eps))
            aux_val = float(bce.mean())
            aux = T.Tensor(backend.to_dev(np.asarray(aux_val, dtype=T.current_dtype())),
                           requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                           _saved={"hd": h.data, "tgt": backend.to_dev(tgt),
                                   "w": backend.to_dev(w), "n": bce.size})
            total = T.add(total, T.mul(aux, T.Tensor(np.asarray(self.lam_concept, dtype=T.current_dtype()))))
            parts["concept_aux"] = aux_val
            if self.decorr > 0:
                # OPTIONAL: penalise the squared off-diagonal covariance of the concept
                # activations -- exactly the rivalry the strict audit measures. Off by default;
                # its gradient was numerically checked before being wired (decorr.py).
                import decorr as _dc
                dnode, dval = _dc.decorr_term(h)
                total = T.add(total, T.mul(dnode, T.Tensor(np.asarray(self.decorr,
                                                                     dtype=T.current_dtype()))))
                parts["decorr"] = dval
        return total, parts

    def parameters(self):
        ps = list(self.front.proj.parameters())
        ps.extend(self.concept.parameters())
        ps.extend(self.cap_out.parameters())
        ps.append(self.cap_emb)
        return ps

    # --------------------------------------------------------------- save / load
    def save(self, path, data):
        """Store EVERY weight, every concept NAME+FORMULA, and the vocab WORDS (not just sizes --
        the exact lesson from the camera vocab bug: sizes can match while words differ, and then
        the model prints wrong names silently)."""
        np.savez_compressed(
            path,
            front_W=B.to_cpu(self.front.proj.W.data), front_b=B.to_cpu(self.front.proj.b.data),
            con_W=B.to_cpu(self.concept.W.data), con_b=B.to_cpu(self.concept.b.data),
            out_W=B.to_cpu(self.cap_out.W.data), out_b=B.to_cpu(self.cap_out.b.data),
            cap_emb=B.to_cpu(self.cap_emb.data),
            meta=np.array([json.dumps({
                "V": self.V, "D": self.D, "W": self.W, "K": self.K, "P": self.P,
                "image": self.image, "patch": self.front.patch,
                "concept_in": self.concept_in, "concept_direct": self.concept_direct,
                "concept_mirror": self.concept_mirror,
                "vocab": data["vocab"], "names": data["names"],
                "concepts": [{"name": c.name, "formula": getattr(c, "formula", ""),
                              "support": float(getattr(c, "support", 0.5) or 0.5)}
                             for c in self.concepts],
                "pos_weight": self.pos_weight.tolist()})]),
        )

    @staticmethod
    def load(path):
        """Rebuild a SeeTalkNet from a checkpoint. Vocab and concepts come FROM the checkpoint.

        The concept FUNCTIONS are re-resolved from image_concepts.SPEC by NAME: a checkpoint whose
        concept is not in the frozen SPEC raises instead of running with a stub. Names+formulas
        travel with the weights so an old checkpoint can never be scored against renamed concepts.
        """
        import image_concepts
        from lens.vision import ImageConcept   # module-level import lives here, not in
        # image_concepts (its build() imports it locally) — checked, not guessed, this time.
        # CONCEPT REGISTRY: base SPEC plus the frozen extended/more/band definitions, keyed by
        # (name, formula). K=22/K=48 checkpoints use these; an UNKNOWN name or a CHANGED formula
        # still raises — the strictness is about identity, not about which file the spec lives in.
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"][0]))
        reg = {name: (fn, fm) for name, fn, fm in image_concepts.SPEC}
        try:
            from concepts_more import extended_specs
            for name, fn, fm in extended_specs():
                reg.setdefault(name, (fn, fm))
        except Exception:
            pass
        try:
            from concepts_sweep import more_specs, band_specs
            for name, fn, fm in more_specs() + band_specs():
                reg.setdefault(name, (fn, fm))
        except Exception:
            pass
        cons = []
        for c in meta["concepts"]:
            if c["name"] not in reg:
                raise ValueError(f"checkpoint concept {c['name']!r} is not in the frozen concept "
                                 f"registry (base SPEC + extended + more + band)")
            fn, fm = reg[c["name"]]
            if c["formula"] and fm and c["formula"] != fm:
                raise ValueError(f"concept {c['name']!r} FORMULA changed since the checkpoint was "
                                 f"saved: {c['formula']!r} != {fm!r}")
            s = max(float(c.get("support", 0.5)), 1e-6)
            cons.append(ImageConcept(c["name"], fn, support=s, formula=c["formula"]))
        net = SeeTalkNet(cons, meta["V"], d=meta["D"], window=meta["W"],
                        image=meta.get("image", 28), patch=meta.get("patch", 7),
                        concept_direct=bool(meta.get("concept_direct", False)),
                        concept_mirror=bool(meta.get("concept_mirror", False)))
        # shape guard: a checkpoint recorded as direct must rebuild with the recorded input width;
        # if an old loader (without meta["concept_in"]) hits a direct ship, the width check below
        # fails loudly instead of running on mis-sized weights.
        ci = int(meta.get("concept_in", net.P * net.D))
        if net.concept.W.data.shape[0] != ci:
            raise ValueError(f"concept input width mismatch: checkpoint {ci} vs rebuilt "
                             f"{net.concept.W.data.shape[0]}")
        net.pos_weight = np.asarray(meta["pos_weight"], dtype=np.float32)
        net.front.proj.W.data = backend.to_dev(z["front_W"])
        net.front.proj.b.data = backend.to_dev(z["front_b"])
        net.concept.W.data = backend.to_dev(z["con_W"])
        net.concept.b.data = backend.to_dev(z["con_b"])
        net.cap_out.W.data = backend.to_dev(z["out_W"])
        net.cap_out.b.data = backend.to_dev(z["out_b"])
        net.cap_emb.data = backend.to_dev(z["cap_emb"])
        for p in net.parameters():
            p.grad = None
        return net, meta


class TextOnlyControl:
    """Same caption side, ZERO image input. g4 is decided against this."""

    def __init__(self, vocab_size, d=32, window=6, seed=0, hidden=96):
        rng = np.random.default_rng(seed)
        self.D, self.W = d, window
        self.emb = T.Tensor(backend.to_dev(rng.standard_normal((vocab_size, d))
                                           .astype(np.float32) * d ** -0.5),
                            requires_grad=True, name="emb")
        self.h = Dense(d * window, hidden, rng, name="h")
        self.o = Dense(hidden, vocab_size, rng, name="o")

    def forward(self, X, cap_ids):
        cxt = T.embedding(self.emb, np.asarray(cap_ids), self.W, self.D)
        return {"logits": self.o(T.relu(self.h(cxt)))}

    def forward_arrays(self, X, cap_ids):
        return {"logits": backend.to_cpu(self.forward(X, cap_ids)["logits"].data)}

    def _mirror_map(self, X):
        """|I - mirror(I)| flattened per image (the nonlinear basis for a pairwise formula)."""
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim == 2:
            Xa = Xa.reshape(-1, self.front.channels, self.image, self.image)
        return np.abs(Xa - Xa[..., ::-1]).reshape(len(Xa), -1)

    def loss(self, X, cap_ids, y, ce_weight=1.0):
        # the CONTROL has no concept layer, so the concept-first phase cannot apply to it: it
        # always trains on the caption loss (ce_weight is accepted and ignored, so the shared
        # training loop can call both nets identically)
        ce = T.softmax_ce(self.forward(X, cap_ids)["logits"], y)
        return ce, {"ce": ce}

    def parameters(self):
        ps = self.h.parameters() + self.o.parameters()
        ps.append(self.emb)
        return ps


def build_caption_data(which="fashion", n=4000, seed=0, img_start=0):
    """Captions GENERATED from Fashion-MNIST class names (MIT): 'this is a ankle boot' style.
    Every word is one prediction step with the previous W words as context.

    img_start: skip this many images before taking n — used by the DISJOINT-TEACHER experiment so
    the teacher's image pool cannot overlap the student's (a teacher that memorised the student's
    rows regurgitates the answer key and makes the distillation test fake — see LOG 2026-10-08)."""
    d = vl_data.prepare_vqa(which, max_train=None, seed=seed)
    names = d["meta"]["names"]
    y = d["ytr"][img_start:img_start + n]
    img_pool = d["Xtr"][img_start:img_start + n]
    tmpl = ["this is a", "i see a", "the photo shows a", "looks like a"]
    freq = {}
    for c in y:
        for w in names[c].split("_"):
            freq[w] = freq.get(w, 0) + 1
    # THE TEMPLATE WORDS MUST BE IN THE VOCABULARY. The first version built `freq` from class names
    # only, so vmap.get("this", 0) silently mapped every template word to <pad>: the model was shown
    # padding where English was, and g1 then "failed" by predicting <pad> at a position whose true
    # target WAS <pad> under that broken mapping. A test can only see what the vocab can say.
    for t in tmpl:
        for w in t.split():
            freq.setdefault(w, 1)
    vocab = ["<pad>", "<bos>"] + sorted(freq, key=lambda w: (-freq[w], w))[:200]
    vmap = {w: i for i, w in enumerate(vocab)}
    W = 6
    X, S, Y = [], [], []
    rng = np.random.default_rng(seed)
    for img, c in zip(img_pool, y):
        words = ["<bos>"] + tmpl[rng.integers(len(tmpl))].split() + names[c].split("_")
        for k in range(1, len(words)):
            hist = words[max(0, k - W):k]
            pad = [vmap["<pad>"]] * (W - len(hist)) + [vmap.get(w, 0) for w in hist]
            X.append(img)
            S.append(pad)
            Y.append(vmap.get(words[k], 0))
    X = np.asarray(X, dtype=np.float32)
    S = np.asarray(S, dtype=np.int64)
    Y = np.asarray(Y, dtype=np.int64)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.9)
    tr, va = idx[:split], idx[split:]
    return {"Xtr": X[tr], "Str": S[tr], "Ytr": Y[tr], "Xva": X[va], "Sva": S[va], "Yva": Y[va],
            "vocab": vocab, "W": W, "names": names, "n_img": n}


def train(net, data, epochs=6, batch=128, lr=0.2, seed=0, tag="see_talk", concept_first=0):
    """concept_first=N: for the first N epochs train ONLY the concept loss (answer loss off),
    then FREEZE the front+concept layer and train the answer layer alone. One change, aimed at
    the honesty margins the tuning levers could not fix."""
    rng = np.random.default_rng(seed)
    X, S, Y = data["Xtr"], data["Str"], data["Ytr"]
    Xv, Sv, Yv = data["Xva"], data["Sva"], data["Yva"]
    hist = []
    frozen = False
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        ce_w = 0.0 if ep <= concept_first else 1.0
        if concept_first and ep == concept_first + 1 and not frozen:
            if not (hasattr(net, "front") and hasattr(net, "concept")):
                log(f"  [concept-first] {type(net).__name__} has NO concept layer -> the "
                    f"concept-first schedule is skipped for it (nothing to freeze)")
                frozen = True
            else:
                for p in list(net.front.proj.parameters()) + list(net.concept.parameters()):
                    p.requires_grad = False
                frozen = True
            log(f"  [concept-first] {concept_first} epochs done: front+concept layer FROZEN; "
                f"the answer layer trains alone from here")
        order = rng.permutation(len(X))
        tot, nb, corr, seen, ce_acc = 0.0, 0, 0, 0, 0.0
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            for p in net.parameters():
                p.grad = None
            total, parts = net.loss(X[idx], S[idx], Y[idx], ce_weight=ce_w)
            _ce = parts.get("ce", 0.0)
            # SeeTalkNet reports parts["ce"] as a float; TextOnlyControl returns the TENSOR.
            # float(tensor) raises -- that crash cost a full training run, so both shapes are
            # handled explicitly instead of assuming one.
            ce_acc += (float(backend.to_cpu(_ce.data)) if hasattr(_ce, "data") else float(_ce))
            total.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
            tot += float(B.to_cpu(total.data))
            nb += 1
            lg = B.to_cpu(net.forward(X[idx], S[idx])["logits"].data)
            corr += int((lg.argmax(axis=1) == Y[idx]).sum())
            seen += len(idx)
        vl, vc, vs = 0.0, 0, 0
        for s in range(0, len(Xv), batch):
            idx = np.arange(s, min(s + batch, len(Xv)))
            lg = net.forward_arrays(Xv[idx], Sv[idx])["logits"]
            vl += float(T.softmax_ce(T.Tensor(B.to_dev(lg)), Yv[idx]).data) * len(idx)
            vc += int((lg.argmax(axis=1) == Yv[idx]).sum())
            vs += len(idx)
        rec = {"epoch": ep, "phase": ("concept-only" if ce_w == 0.0 else "answer"),
               "ce_loss": ce_acc / max(nb, 1),
               "train_loss": tot / max(nb, 1), "train_acc": corr / max(seen, 1),
               "val_ppl": float(np.exp(vl / max(vs, 1))), "val_acc": vc / max(vs, 1),
               "seconds": time.perf_counter() - t0}
        hist.append(rec)
        log(f"  {tag} ep {ep}/{epochs} [{rec['phase']}] loss {rec['train_loss']:.4f} "
            f"acc {rec['train_acc']*100:.2f}%  VAL ppl {rec['val_ppl']:.2f} "
            f"acc {rec['val_acc']*100:.2f}%  ({rec['seconds']:.0f}s)")
    return hist


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-start", type=int, default=0,
                    help="skip this many images: builds a DISJOINT pool so a teacher can be "
                         "trained on images the student never saw (distillation integrity)")
    ap.add_argument("--out", default="seetalk_tiny")
    a = ap.parse_args()
    T.set_device(a.device)

    log("=" * 100)
    log("SEE + TALK — image in, caption out, every word through the named concept layer")
    log("=" * 100)
    data = build_caption_data(n=a.n, seed=a.seed, img_start=a.img_start)
    log(f"  caption vocab {len(data['vocab'])} words, W={data['W']}, "
        f"{len(data['Xtr']):,} train / {len(data['Xva']):,} val (image,prefix)->word pairs "
        f"from {data['n_img']} images")

    concepts, spec = image_concepts.build(data["Xtr"], min_support=0.02, verbose=False)
    log(f"  named concepts: {spec['kept']} kept of {spec['n_defined']}")

    net = SeeTalkNet(concepts, vocab_size=len(data["vocab"]), d=a.d, window=data["W"],
                     seed=a.seed)
    npar = int(sum(p.data.size for p in net.parameters()))
    ctrl = TextOnlyControl(len(data["vocab"]), d=a.d, window=data["W"], seed=a.seed)
    cpar = int(sum(p.data.size for p in ctrl.parameters()))
    log(f"  SEE+TALK {npar:,} params   TEXT-ONLY control {cpar:,} params")

    log("")
    hist = train(net, data, epochs=a.epochs, seed=a.seed, tag="see_talk")
    hist_c = train(ctrl, data, epochs=a.epochs, seed=a.seed, tag="text_only")

    log("")
    log("=" * 100)
    log("GATES — fixed BEFORE this run, binary")
    log("=" * 100)
    res = {}

    # g1: same prefix, 10 different images -> >=5 distinct next words.
    # THE PROBE POINT MATTERS: the first version used Sva[0], whose next word is the first TEMPLATE
    # word -- a position that legitimately does NOT depend on the image, so "1 distinct word" there
    # measured my probe, not the model. The honest probe ends right BEFORE the class noun, where the
    # answer CAN ONLY come from the image. Gate bar unchanged (>=5 of 10).
    vi = {w: i for i, w in enumerate(data["vocab"])}
    prefix = [vi["<bos>"], vi["this"], vi["is"], vi["a"]]
    prefix = [vi["<pad>"]] * (data["W"] - len(prefix)) + prefix
    prefix = np.asarray(prefix, dtype=np.int64)
    idx10 = np.linspace(0, len(data["Xva"]) - 1, 10).astype(int)
    outs = []
    for i in idx10:
        lg = net.forward_arrays(data["Xva"][i:i + 1], prefix[None, :])["logits"][0]
        outs.append(int(lg.argmax()))
    distinct = len(set(outs))
    g1 = distinct >= 5
    words = sorted({data["vocab"][w] for w in outs})
    log(f"  g1 image changes the word:  prefix 'this is a ___' + 10 images -> {distinct} distinct "
        f"next words (need >=5)  -> {'PASS' if g1 else 'FAIL'}   words seen: {words[:8]}")
    res["g1"] = bool(g1)
    res["g1_words"] = words

    # g2: train loss fell
    g2 = hist[-1]["train_loss"] < hist[0]["train_loss"]
    log(f"  g2 train loss fell:         {hist[0]['train_loss']:.4f} -> "
        f"{hist[-1]['train_loss']:.4f}  -> {'PASS' if g2 else 'FAIL'}")
    res["g2"] = bool(g2)

    # g3: hand-recompute the arithmetic of the last two layers and compare
    Xb, Sb = data["Xva"][:64], data["Sva"][:64]
    out = net.forward_arrays(Xb, Sb)
    Pn = net.front.patches(Xb)                                  # (B,P,49) numpy
    zp = Pn.reshape(-1, Pn.shape[-1]) @ B.to_cpu(net.front.proj.W.data) \
        + B.to_cpu(net.front.proj.b.data)                       # (B*P, D)
    zflat = zp.reshape(len(Xb), -1)
    cl = zflat @ B.to_cpu(net.concept.W.data) + B.to_cpu(net.concept.b.data)
    h3 = 1.0 / (1.0 + np.exp(-np.clip(cl, -60, 60)))
    cemb = B.to_cpu(net.cap_emb.data)
    cxt = np.concatenate([cemb[Sb[:, w]] for w in range(Sb.shape[1])], axis=1)
    manual = np.concatenate([h3, cxt], axis=1) @ B.to_cpu(net.cap_out.W.data) \
        + B.to_cpu(net.cap_out.b.data)
    resid = float(np.abs(manual - out["logits"]).max())
    g3 = resid < 1e-5
    log(f"  g3 explain residual:        {resid:.2e} (bar 1e-5)  -> {'PASS' if g3 else 'FAIL'}")
    res["g3"] = bool(g3)

    # g4: beats the text-only control on val accuracy
    va, vb = hist[-1]["val_acc"], hist_c[-1]["val_acc"]
    g4 = va > vb + 0.005
    log(f"  g4 beats text-only control: see+talk {va*100:.2f}% vs control {vb*100:.2f}% "
        f"(need > +0.50)  -> {'PASS' if g4 else 'FAIL'}")
    res["g4"] = bool(g4)

    # count ONLY the boolean gates. (The first version crashed HERE: res also held the g1 word
    # LIST, and sum(res.values()) tried to add True + ['ankle boot', ...].)
    n_pass = sum(v for k, v in res.items() if isinstance(v, bool))
    n_gates = sum(1 for v in res.values() if isinstance(v, bool))
    log("")
    log(f"  RESULT: {n_pass}/{n_gates} gates {'ALL PASS' if n_pass == n_gates else 'NOT ALL PASS'}")
    OUT = os.path.join(PROJ, "lens", "runs", "seetalk")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"args": vars(a), "see_talk": hist, "text_only": hist_c, "gates": res,
               "params": {"see_talk": npar, "control": cpar}},
              open(os.path.join(OUT, f"{a.out}_log.json"), "w"), indent=1)
    net.save(os.path.join(OUT, f"{a.out}.npz"), data)
    # ROUND-TRIP CHECK: a checkpoint that cannot be loaded is not a checkpoint. Load it, run one
    # forward, and require the logits to be BIT-IDENTICAL to the in-memory model's.
    net2, meta2 = SeeTalkNet.load(os.path.join(OUT, f"{a.out}.npz"))
    o1 = net.forward_arrays(Xb, Sb)["logits"]
    o2 = net2.forward_arrays(Xb, Sb)["logits"]
    rt = float(np.abs(o1 - o2).max())
    log(f"  checkpoint saved and round-tripped: max |orig-loaded| = {rt:.3e}  "
        f"{'IDENTICAL' if rt == 0.0 else 'MISMATCH — checkpoint unusable'}")
    if rt != 0.0:
        raise SystemExit(f"round-trip mismatch {rt}")
    log(f"  saved {a.out}_log.json")
    log("=" * 100)


if __name__ == "__main__":
    main()
