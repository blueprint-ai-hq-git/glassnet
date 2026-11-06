"""lens/vision.py — the IMAGE front-end, on the same concept layer as text.

THE FUSION IDEA, AND WHY IT IS NOT A HACK
-----------------------------------------
The text model is:

    16 word tokens x D  ->  NAMED concept layer  ->  answer layer

An image can be made to look the same:

    16 image PATCHES x D  ->  NAMED concept layer  ->  answer layer

So the middle of the model does not care whether the 16 things came from a sentence or from a
picture. That is the whole basis for a vision-language model built from scratch: ONE concept layer,
two front-ends. A patch is a token; a token is a patch.

28x28 image -> 4x4 grid of 7x7 patches = 16 patches x 49 pixels each. 16 patches, exactly like the
16-word window. Not a coincidence I chose it -- it keeps the two front-ends interchangeable.
"""
import os

import numpy as np

from lens import tensor as T
from lens import backend
from lens.model import Dense


class PatchEmbed:
    """image -> 16 patch vectors -> project each patch to D dimensions.

    `channels` exists because averaging RGB destroys information. On PathMNIST the channel means are
    188.8 / 136.0 / 180.0 and stain COLOUR separates tissue types, so the loader's `mean(axis=-1)`
    was throwing away the discriminating signal. With channels=3 a patch is 3*7*7 = 147 pixels
    instead of 49, and the projection learns from colour.
    """

    def __init__(self, patch=7, d=64, rng=None, image=28, channels=1):
        rng = rng or np.random.default_rng(0)
        self.patch = int(patch)
        self.image = int(image)
        self.channels = int(channels)
        self.D = int(d)
        self.grid = self.image // self.patch            # 4
        self.P = self.grid * self.grid                  # 16 patches -- matches the text window
        self.patch_dim = self.channels * self.patch * self.patch
        self.proj = Dense(self.patch_dim, d, rng, name="patch_proj")

    def patches(self, images):
        """(B, C*784) or (B, C, 28, 28) -> (B, P, patch_dim), row-major grid, pixels in [0,1]."""
        X = np.asarray(images, dtype=np.float32)
        if X.ndim == 2:
            X = X.reshape(-1, self.channels, self.image, self.image)
        B, C = X.shape[0], X.shape[1]
        g, p = self.grid, self.patch
        # (B, C, g, p, g, p) -> (B, g, g, C, p, p) -> (B, P, C*p*p)
        blocks = X.reshape(B, C, g, p, g, p).transpose(0, 2, 4, 1, 3, 5)
        return blocks.reshape(B, self.P, C * p * p)

    def __call__(self, images):
        """(B, 784) -> (B, P*D).

        The SAME projection is applied to every patch: the patches are flattened into the batch
        dimension, multiplied once by the shared projection matrix, then reinterpreted back. That
        is weight sharing across patch positions -- the image equivalent of using one embedding
        table for every word position instead of one per position.
        """
        P = self.patches(images)                         # (B, P, patch_dim) numpy
        Bn = P.shape[0]
        flat = T.Tensor(backend.to_dev(P.reshape(Bn * self.P, self.patch_dim)))
        proj = self.proj(flat)                           # (B*P, D)
        return T.reshape(proj, (Bn, self.P * self.D))    # (B, P*D) for the concept layer

    def parameters(self):
        return self.proj.parameters()


class VisionNet:
    """image -> patches -> NAMED concept layer -> answer layer.

    Structurally identical to LensNet after the front-end, on purpose: the same concept layer, the
    same answer layer, the same explanation machinery.
    """

    def __init__(self, concepts, n_classes, d=64, patch=7, image=28, channels=1, seed=0,
                 lam_concept=0.3, lam_act=0.0):
        concepts = list(concepts)
        if not concepts:
            raise ValueError("VisionNet needs named concepts; the concept layer IS the architecture.")
        self.concepts = concepts
        self.K = len(concepts)
        self.C = int(n_classes)
        self.D = int(d)
        self.seed = int(seed)
        self.lam_concept = float(lam_concept)
        self.lam_act = float(lam_act)
        self.channels = int(channels)

        rng = np.random.default_rng(seed)
        self.front = PatchEmbed(patch=patch, d=d, rng=rng, image=image, channels=channels)
        self.P = self.front.P
        self.concept = Dense(self.P * self.D, self.K, rng, name="concept")
        self.answer = Dense(self.K, self.C, rng, name="answer")

        pw = []
        for c in concepts:
            s = max(float(c.support), 1e-6) if c.support is not None else 0.5
            pw.append((1.0 - s) / s)
        self.pos_weight = np.asarray(pw, dtype=np.float32)

    def parameters(self):
        return self.front.parameters() + self.concept.parameters() + self.answer.parameters()

    def forward(self, images):
        x = self.front(images)                     # (B, P, D) as one Tensor
        h = T.sigmoid(self.concept(x))
        logits = self.answer(h)
        return {"concepts": h, "logits": logits}

    def forward_arrays(self, images):
        out = self.forward(np.asarray(images))
        return {"concepts": backend.to_cpu(out["concepts"].data),
                "logits": backend.to_cpu(out["logits"].data)}

    def loss(self, images, y):
        out = self.forward(images)
        ce = T.softmax_ce(out["logits"], y)
        parts = {"ce": float(backend.to_cpu(ce.data))}
        total = ce
        if self.lam_concept > 0:
            h = out["concepts"]
            hd = backend.to_cpu(h.data)
            tgt = np.stack([c.labels(images) for c in self.concepts], axis=1).astype(T.current_dtype())
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
        return total, parts


class TextConcept:
    """A named property of a CAPTION, measurable from its token ids.

    The other half of the fused layer: image concepts describe what is in the picture, text concepts
    describe what the caption says. Keeping both in one layer is what makes it a VL model instead of
    two models bolted together.
    """

    def __init__(self, name, ids, support=None, formula=""):
        self.name = name
        self.ids = list(ids)
        self.support = support
        self.formula = formula

    def labels(self, cap_ids):
        c = np.asarray(cap_ids)
        return np.isin(c, self.ids).any(axis=1).astype(np.float32)

    def __repr__(self):
        return f"TextConcept({self.name!r}, support={self.support})"


class VLNet:
    """VISION-LANGUAGE: image patches AND caption tokens -> ONE shared concept layer -> caption word.

        image (28x28) -> 16 patches x D  --\
                                            >-- SHARED concept layer -> answer -> next caption word
        caption tokens     ->  L tokens x D --/

    The concept layer contains BOTH kinds of concept:
      image concepts (is_bright, left_right_symmetric, ...) -- measured from pixels
      text concepts  (says_sneaker, says_trouser, ...)       -- measured from caption tokens
    and the answer layer reads all of them. So the model can only produce a caption word through
    named, checkable properties, on both sides.
    """

    def __init__(self, image_concepts, text_concepts, q_vocab_size, answer_vocab_size, caption_len,
                 d=64, patch=7, image=28, channels=1, seed=0, lam_concept=0.3, lam_act=0.0):
        self.img_concepts = list(image_concepts)
        self.txt_concepts = list(text_concepts)
        self.concepts = self.img_concepts + self.txt_concepts
        if not self.concepts:
            raise ValueError("VLNet needs named concepts on at least one side.")
        self.K = len(self.concepts)
        # TWO vocabularies, and they are not the same size: the question is written in question
        # words (what/name/see/...), the answer is written in answer words (sneaker/coat/...).
        # Using one size for both is what raised "index 13 is out of bounds for axis 0 with size 12".
        self.QV = int(q_vocab_size)
        self.V = int(answer_vocab_size)
        self.L = int(caption_len)          # number of QUESTION tokens fed in
        self.D = int(d)
        self.seed = int(seed)
        self.lam_concept = float(lam_concept)
        self.lam_act = float(lam_act)
        self.channels = int(channels)

        rng = np.random.default_rng(seed)
        self.front = PatchEmbed(patch=patch, d=d, rng=rng, image=image, channels=channels)
        self.P = self.front.P
        self.cap_emb = T.Tensor(backend.to_dev(rng.standard_normal((self.QV, self.D))
                                              .astype(np.float32) * self.D ** -0.5),
                                requires_grad=True, name="cap_emb")
        self.concept = Dense(self.P * self.D + self.L * self.D, self.K, rng, name="concept")
        self.answer = Dense(self.K, self.V, rng, name="answer")

        pw = []
        for c in self.concepts:
            s = max(float(c.support), 1e-6) if c.support is not None else 0.5
            pw.append((1.0 - s) / s)
        self.pos_weight = np.asarray(pw, dtype=np.float32)

    def parameters(self):
        return (self.front.parameters() + [self.cap_emb]
                + self.concept.parameters() + self.answer.parameters())

    def forward(self, images, cap_ids):
        img = self.front(images)                                  # (B, P*D)
        cap = T.embedding(self.cap_emb, np.asarray(cap_ids), self.L, self.D)   # (B, L*D)
        x = T.concat([img, cap], axis=1)                          # (B, (P+L)*D)
        h = T.sigmoid(self.concept(x))
        return {"concepts": h, "logits": self.answer(h), "joint": x}

    def forward_arrays(self, images, cap_ids):
        out = self.forward(np.asarray(images), np.asarray(cap_ids))
        return {"concepts": backend.to_cpu(out["concepts"].data),
                "logits": backend.to_cpu(out["logits"].data)}

    def concept_targets(self, images, cap_ids):
        """Both halves: image concepts from pixels, text concepts from caption tokens."""
        cols = [c.labels(images) for c in self.img_concepts]
        cols += [c.labels(cap_ids) for c in self.txt_concepts]
        return np.stack(cols, axis=1).astype(T.current_dtype())

    def loss(self, images, cap_in, cap_target, class_weights=None):
        out = self.forward(images, cap_in)
        ce = T.softmax_ce(out["logits"], np.asarray(cap_target), class_weights=class_weights)
        parts = {"ce": float(backend.to_cpu(ce.data))}
        total = ce
        if self.lam_concept > 0:
            h = out["concepts"]
            hd = backend.to_cpu(h.data)
            tgt = self.concept_targets(images, cap_in)
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
        return total, parts


class ImageConcept:
    """A named, measurable property of an image, with its formula kept alongside it.

    `fn` takes (B, 784) float pixels in [0,1] and returns a (B,) 0/1 array. The formula is stored so
    a reader can check the name against the maths -- the same honesty rule as the text concepts.
    """

    def __init__(self, name, fn, support=None, formula=""):
        self.name = name
        self.fn = fn
        self.support = support
        self.formula = formula

    def labels(self, images):
        return np.asarray(self.fn(np.asarray(images, dtype=np.float32)), dtype=np.float32)

    def __repr__(self):
        return f"ImageConcept({self.name!r}, support={self.support})"


# ------------------------------------------------------------------------------------------------
# SAVE / LOAD for the vision models. The checkpoint stores the concept NAMES and their declared
# formulas, never the functions -- a function is not data. Loading rebuilds each concept from
# image_concepts.SPEC by name, so a checkpoint cannot silently come back with different concepts
# than it was trained with; a missing name is a hard error rather than a quiet substitution.
# ------------------------------------------------------------------------------------------------
def save_vision(net, path, kind="vision", avocab=None, qvocab=None):
    """Save weights + metadata. Concepts go in BY NAME, and now so do the VOCABULARIES.

    WHY THE VOCABULARIES ARE SAVED (they were not, and it caused a real bug):
    The checkpoint recorded only the SIZES (V=14, QV=15). A loader therefore had to rebuild the words
    from whatever data it had, and nothing checked that the rebuild agreed. Loading a 14-class model
    against the 12-word fashion vocabulary produced `IndexError` deep inside frame drawing -- and
    worse, had the sizes happened to match it would have produced silently WRONG answers, mapping
    index 13 to the wrong class name with no error at all.

    The concepts already had the right treatment (stored by name, hard error on mismatch). The
    vocabularies now get the same treatment: the words travel with the weights.
    """
    import json
    d = {p.name: backend.to_cpu(p.data) for p in net.parameters()}
    meta = {"kind": kind, "D": net.D, "seed": net.seed, "lam_concept": net.lam_concept,
            "P": net.front.P, "patch": net.front.patch, "image": net.front.image,
            "channels": int(getattr(net.front, "channels", 1)),
            "concepts": [{"name": c.name, "support": c.support,
                          "formula": getattr(c, "formula", ""),
                          "ids": list(getattr(c, "ids", [])),
                          "side": "text" if isinstance(c, TextConcept) else "image"}
                         for c in net.concepts]}
    if kind == "vision":
        meta["C"] = net.C
    else:
        meta["V"] = net.V
        meta["QV"] = net.QV
        meta["L"] = net.L
        # the words themselves, so index -> class name never has to be guessed
        meta["avocab"] = list(avocab) if avocab is not None else list(net.avocab or [])
        meta["qvocab"] = list(qvocab) if qvocab is not None else list(net.qvocab or [])
        for k, n in (("avocab", meta["V"]), ("qvocab", meta["QV"])):
            if len(meta[k]) != n:
                raise ValueError(f"{k} has {len(meta[k])} words but the weights expect {n} -- "
                                 f"the checkpoint would map indices to the wrong names")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **d, __meta__=np.asarray(json.dumps(meta)))
    return path


def check_vocab_match(meta, avocab=None, qvocab=None):
    """Compare a checkpoint's vocabularies against the caller's. Raise LOUDLY on any mismatch.

    Returns (avocab, qvocab) to use -- always the CHECKPOINT's words, never the caller's, because the
    weights were trained against the checkpoint's indexing.
    """
    ck_a, ck_q = meta.get("avocab"), meta.get("qvocab")
    for label, ck, mine, size_key in (("answer", ck_a, avocab, "V"), ("question", ck_q, qvocab, "QV")):
        n = meta.get(size_key)
        if ck is None:
            raise KeyError(
                f"checkpoint has no {label} vocabulary stored (only the size {n}). It cannot be "
                f"loaded safely: the words that indices map to are unknown, so an answer could be "
                f"reported under the wrong name with no error. Re-save the checkpoint with "
                f"save_vision(..., {label[0]}vocab=...) or retrain.")
        if len(ck) != n:
            raise ValueError(f"checkpoint's {label} vocabulary has {len(ck)} words but V/QV is {n}")
        if mine is not None and list(mine) != list(ck):
            raise ValueError(
                f"{label} vocabulary mismatch: the checkpoint was trained on {ck}, the caller "
                f"supplied {list(mine)}. Refusing to run -- using either one would map indices to "
                f"the wrong words. Use the checkpoint's vocabulary.")
    return list(ck_a), list(ck_q)


def load_vision(path, device="cpu"):
    """Rebuild a VisionNet or VLNet from a checkpoint. Concepts AND vocabularies come back by name."""
    import json
    import sys
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "scripts"))
    import image_concepts
    spec = {nm: (fn, fm) for nm, fn, fm in image_concepts.SPEC}
    # the FROZEN extended concept registry (same set train_seetalk.SeeTalkNet.load resolves):
    # ktest_* / K>12 checkpoints save these by name+formula; unknown names still raise below.
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.abspath(__file__))), "scripts"))
        from concepts_more import extended_specs
        for _nm, _fn, _fm in extended_specs():
            spec.setdefault(_nm, (_fn, _fm))
    except Exception:
        pass
    try:
        from concepts_sweep import more_specs, band_specs
        for _nm, _fn, _fm in more_specs() + band_specs():
            spec.setdefault(_nm, (_fn, _fm))
    except Exception:
        pass
    # COLOUR concepts (PathMNIST line): the frozen registry is grayscale-only, so colour ships
    # used to raise KeyError and could never load after training. The colour measures are
    # quantile-threshold formulas whose RESOLVED threshold is stored in the checkpoint's own
    # formula string — we re-bind the measure with THAT threshold (never a substituted one).
    import re as _re
    _COLOUR = {}
    try:
        import colour_concepts
        _COLOUR = {nm: (fn, dr) for nm, fn, _q, dr, _fm in colour_concepts.MEASURES}
    except Exception:
        pass
    img_c, txt_c = [], []
    for c in meta["concepts"]:
        if c["side"] == "text":
            txt_c.append(TextConcept(c["name"], c.get("ids", []), support=c["support"],
                                     formula=c["formula"]))
        else:
            if c["name"] not in spec:
                if c["name"] in _COLOUR:
                    _fn_colour, _dr = _COLOUR[c["name"]]
                    _m = _re.search(r"resolved threshold = ([0-9.eE+-]+)", c.get("formula", ""))
                    if _m is None:
                        raise KeyError(f"colour concept {c['name']!r} has no resolved threshold "
                                       f"in its formula string -- refusing to guess")
                    _thr = float(_m.group(1))
                    fn = colour_concepts._bind(_fn_colour, _thr, _dr)
                else:
                    raise KeyError(f"checkpoint has concept {c['name']!r} which image_concepts.SPEC "
                                   f"does not define -- refusing to substitute a different concept")
            else:
                fn, fm = spec[c["name"]]
                if meta.get("channels", 1) == 3:
                    # BUG-MED-1: grayscale measures channel-slice colour data and return 3N
                    # values for N windows, so the SAVED colour ship crashes in concept_targets
                    # (the run's audit numbers came from in-memory objects — not reproducible
                    # from the artifact). The grayscale concept's true value on colour data is
                    # its per-window CHANNEL MEAN (the original med line ran on averaged
                    # channels). Wrap: (3N,) -> (N,).
                    def _img_mean(X, _fn=fn):
                        v = np.asarray(_fn(np.asarray(X, dtype=np.float32)))
                        if v.ndim == 1 and len(v) % len(X) == 0:
                            return v.reshape(len(X), -1).mean(axis=1)
                        return v
                    fn = _img_mean
            img_c.append(ImageConcept(c["name"], fn, support=c["support"], formula=c["formula"]))
    if meta["kind"] == "vision":
        net = VisionNet(img_c, n_classes=meta["C"], d=meta["D"], patch=meta["patch"],
                        image=meta["image"], channels=meta.get("channels", 1),
                        seed=meta["seed"], lam_concept=meta["lam_concept"])
    else:
        net = VLNet(img_c, txt_c, q_vocab_size=meta["QV"], answer_vocab_size=meta["V"],
                    caption_len=meta["L"], d=meta["D"], patch=meta["patch"],
                    image=meta["image"], channels=meta.get("channels", 1),
                    seed=meta["seed"], lam_concept=meta["lam_concept"])
    if meta["kind"] == "vision":
        net.front.patch, net.front.image = meta["patch"], meta["image"]
    # attach the checkpoint's own vocabularies so a caller can never substitute its own
    net.avocab = list(meta.get("avocab", []))
    net.qvocab = list(meta.get("qvocab", []))
    net.ckpt_meta = meta
    for p in net.parameters():
        if p.name in z:
            p.data = backend.to_dev(z[p.name].astype(np.float32))
    T.set_device(device)
    return net, meta
