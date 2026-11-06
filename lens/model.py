"""lens/model.py — the readable model, on our own tensors and our own kernels.

    window tokens -> embedding -> NAMED concept layer -> answer layer -> next token

The concept layer is the middle. The answer layer sees ONLY the concepts, so the model's reasons
are always named concepts and explain() can be exact.

DESIGN DECISIONS THAT COME FROM THE OLD PROJECT'S FAILURES
----------------------------------------------------------
* **Concept targets are a REQUIRED constructor argument.** In the old code they lived in a
  module-level global that one env setting left as None, so a run trained on the wrong concept
  set while the audit compared a different one -- and reported 0/9 on a model that was fine.
  Here there is no global to get wrong.
* **The concept loss weight is applied EXPLICITLY and is returned in `parts`**, so a run can be
  checked for whether the auxiliary loss was actually active. The old project never logged it.
* **The balanced positive weight is applied**, computed from the TRAIN support. In the old code
  the weight was computed and it was never clear it was used.
* **Activation sparsity is a switch** with a stated default, because a sparsity penalty fights
  exactly the concepts that must fire often.
"""
import numpy as np

from lens import tensor as T
from lens import _capi as C
from lens import backend
from lens.numeric import MagnitudeLayer, NumericHead, num_mse


class Dense:
    """A plain affine layer: y = x @ W + b, with our own init and our own kernels."""

    def __init__(self, in_features, out_features, rng, name="dense"):
        scale = (2.0 / in_features) ** 0.5          # He init for ReLU-family activations
        self.W = T.Tensor(backend.to_dev(rng.standard_normal((in_features, out_features))
                                         .astype(np.float32) * scale),
                          requires_grad=True, name=f"{name}.W")
        self.b = T.Tensor(backend.to_dev(np.zeros(out_features, dtype=np.float32)),
                          requires_grad=True, name=f"{name}.b")

    def __call__(self, x):
        return T.add(T.matmul(x, self.W), self.b)

    def parameters(self):
        return [self.W, self.b]


class CausalSelfAttention:
    """Self-attention over the window, BEFORE the concept layer (one architectural change).

    Window tokens are contextualized first: q,k,v projections (own Dense-style weights, own
    matmuls), batched scores WxW, causal mask (token i sees only tokens <= i), row-softmax,
    weighted values, one output projection. The concept layer then reads CONTEXTUALIZED tokens,
    so a concept like "mentions_weather" can mean "the context is weather-ish", not just
    "this exact token appears". explain() is unchanged: the answer layer still sees only concepts.
    """

    def __init__(self, d, window, rng, name="attn"):
        self.D, self.W = int(d), int(window)
        # Attention scaling (1/sqrt(d) on the scores) is folded into the Q/K init: scaling each
        # projection by d^-1/4 scales q@k^T by d^-1/2. This avoids a broadcast-multiply, which
        # the engine deliberately does not have (mul is strict same-shape; only add broadcasts).
        attn_scale = d ** -0.25
        scale = (2.0 / d) ** 0.5 * attn_scale
        self.Wq = T.Tensor(backend.to_dev(rng.standard_normal((d, d)).astype(np.float32) * scale),
                           requires_grad=True, name=f"{name}.Wq")
        self.Wk = T.Tensor(backend.to_dev(rng.standard_normal((d, d)).astype(np.float32) * scale),
                           requires_grad=True, name=f"{name}.Wk")
        self.Wv = T.Tensor(backend.to_dev(rng.standard_normal((d, d)).astype(np.float32) * scale),
                           requires_grad=True, name=f"{name}.Wv")
        self.Wo = T.Tensor(backend.to_dev(rng.standard_normal((d, d)).astype(np.float32) * scale),
                           requires_grad=True, name=f"{name}.Wo")
        # causal mask: +0 on the main diagonal and below, -1e9 above it. Constant, (W, W),
        # broadcast to (N, W, W) per call as a zero-copy view (the engine's add only supports
        # (N,M)+(M,) bias broadcasts, so the mask is widened explicitly).
        mask_np = np.triu(np.full((self.W, self.W), -1e9, dtype=np.float32), k=1)
        self.mask_np = mask_np
        self.mask_dev = backend.to_dev(mask_np)

    def __call__(self, x):
        # x is the flattened window embedding (N, W*D)
        N, WD = x.data.shape
        assert WD == self.W * self.D, f"attention expects W*D={self.W*self.D}, got {WD}"
        mask3 = T.Tensor(backend.xp.broadcast_to(self.mask_dev, (N, self.W, self.W)),
                         requires_grad=False, name="attn.mask3")
        xr = T.reshape(x, (N * self.W, self.D))
        q = T.reshape(T.matmul(xr, self.Wq), (N, self.W, self.D))
        k = T.reshape(T.matmul(xr, self.Wk), (N, self.W, self.D))
        v = T.reshape(T.matmul(xr, self.Wv), (N, self.W, self.D))
        s = T.bmm3(q, k, trans_b=True)             # (N, W, W) scores = q @ k^T (already scaled)
        s = T.add(s, mask3)                        # causal
        p = T.softmax_rows(s)                      # (N, W, W)
        o = T.bmm3(p, v)                           # (N, W, D)
        o = T.reshape(o, (N * self.W, self.D))
        o = T.matmul(o, self.Wo)                   # (N*W, D) shared output projection
        return T.reshape(o, (N, WD))               # back to the flattened window shape

    def parameters(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo]


class LensNet:
    """The readable network. `concepts` is required: a model with no named concepts is not
    this architecture."""

    def __init__(self, vocab_size, concepts, d=64, window=8, seed=0,
                 lam_concept=0.1, lam_act=0.2, lam_w=1.0, attention=False, mtp=1, numeric=False,
                 direct=False, direct_rank=16, direct_mode="lowrank"):
        concepts = list(concepts)
        if not concepts:
            raise ValueError("LensNet needs at least one named concept; the concept layer IS "
                             "the architecture. Pass concepts=[Concept(...), ...].")
        self.concepts = concepts
        self.K = len(concepts)
        self.V, self.D, self.W = int(vocab_size), int(d), int(window)
        self.lam_concept = float(lam_concept)
        self.lam_act = float(lam_act)
        self.lam_w = float(lam_w)
        self.seed = int(seed)
        # MULTI-TOKEN PREDICTION (added 2026-10-18). mtp=2 adds a SECOND answer head that predicts the
        # token AFTER the next one from the same concept representation. Published reason to do this
        # here: the benefit of 2-token prediction is reported to be LARGEST for models at or below 30M
        # non-embedding parameters and to disappear above 100M (arXiv:2404.19737), and our target model
        # is capped at 20M. Default mtp=1 changes nothing for existing runs or checkpoints, which
        # matters because load() refuses a mismatched parameter set.
        self.mtp = int(mtp)
        if self.mtp < 1:
            raise ValueError("mtp must be >= 1")
        self.lam_mtp = 1.0

        # balanced positive weight per concept, from the declared support. A concept present in
        # 25% of windows needs weight (1-0.25)/0.25 = 3.0; one present in 3% needs 32.0.
        pw = []
        for c in concepts:
            s = max(float(c.support), 1e-6) if c.support is not None else 0.5
            pw.append((1.0 - s) / s)
        self.pos_weight = np.asarray(pw, dtype=np.float32)

        rng = np.random.default_rng(seed)
        self.emb = T.Tensor(backend.to_dev(rng.standard_normal((self.V, self.D))
                                           .astype(np.float32) * self.D ** -0.5),
                            requires_grad=True, name="emb")
        self.attention = attention
        self.attn = CausalSelfAttention(self.D, self.W, rng, name="attn") if attention else None
        self.concept = Dense(self.D * self.W, self.K, rng, name="concept")
        self.answer = Dense(self.K, self.V, rng, name="answer")
        # the extra head exists ONLY when mtp >= 2, so every existing checkpoint keeps loading
        self.answer2 = Dense(self.K, self.V, rng, name="answer2") if self.mtp >= 2 else None
        # THE NUMERIC / MAGNITUDE PATH (2026-10-20, PLAN item (d)). Off by default so every existing
        # checkpoint keeps loading byte-identical: with numeric=False no parameter is added, nothing in
        # forward changes, and save()/load() see exactly the same names they saw before.
        self.numeric = bool(numeric)
        self.mag = MagnitudeLayer(self.D, self.W, rng, name="mag") if self.numeric else None
        self.num = NumericHead(self.W, rng, name="num") if self.numeric else None
        # THE POSITION-AWARE DIRECT PATH (2026-10-21). This fixes the measured cause of the word salad:
        # the answer layer sees ONLY the 41 concept values, so it cannot know WHICH token it was, only what
        # KIND. experiments/bottleneck_proof2.py measured exactly this: concepts-only 5.9% versus a
        # position-aware direct path 77.6% on a pure token-identity task (chance 2.0%).
        #
        # WHY THIS SHAPE: x is the flattened window (W*D). Multiplying it by a (W*D, rank) matrix gives, for
        # EVERY position, its own projection into the rank space, summed - so the readout is position-aware
        # without a per-position vocabulary matrix. Cost: W*D*rank + rank*V = 32*256*16 + 16*12,002 = 323,104
        # parameters at our sizes, which keeps the whole model inside the 50M cap. The naive form
        # (W*D -> V directly) would be 8,192*12,002 = 98M and is over budget, which is why it is low-rank.
        #
        # Off by default so every existing checkpoint keeps loading byte-identical.
        self.direct = bool(direct)
        self.dmode = str(direct_mode) if direct else "off"
        self.drank = int(direct_rank)
        if self.direct and self.dmode == "tied":
            # TIED, FULL-RANK (2026-10-21, after the low-rank version failed to move the talk test).
            # WHY: a rank-16 bottleneck cannot carry WHICH of 12,002 tokens it was - 16 numbers are far too
            # few, which is why the toy test (vocab 50, rank 8) passed while the real model did not improve.
            # Here every position mixes its own D numbers into D numbers (full rank, D=256), and the readout is
            # the model's OWN embedding matrix transposed - tied, so identity flows by construction and the
            # readout costs ZERO new parameters. Cost is W*D*D = 32*256*256 = 2,097,152 parameters.
            scale_m = (2.0 / self.D) ** 0.5
            self.dmix = T.Tensor(backend.to_dev(rng.standard_normal((self.D * self.W, self.D)).astype(np.float32) * scale_m),
                                 requires_grad=True, name="dmix")
            self.dproj = None
            self.dout = None
        elif self.direct:
            scale_d = (2.0 / (self.D * self.W)) ** 0.5
            self.dproj = T.Tensor(backend.to_dev(rng.standard_normal((self.D * self.W, self.drank)).astype(np.float32) * scale_d),
                                  requires_grad=True, name="dproj")
            scale_o = (2.0 / self.drank) ** 0.5
            self.dout = T.Tensor(backend.to_dev(rng.standard_normal((self.drank, self.V)).astype(np.float32) * scale_o),
                                 requires_grad=True, name="dout")
        else:
            self.dproj = None
            self.dout = None

    # ---------------------------------------------------------------- parameters
    def parameters(self):
        ps = [self.emb]
        if self.attn is not None:
            ps += self.attn.parameters()
        ps = ps + self.concept.parameters() + self.answer.parameters()
        if self.answer2 is not None:
            ps += self.answer2.parameters()
        if self.mag is not None:
            ps += self.mag.parameters() + self.num.parameters()
        if self.direct and self.dmode == "tied":
            ps += [self.dmix]
        elif self.direct:
            ps += [self.dproj, self.dout]
        return ps

    # ---------------------------------------------------------------- forward
    def forward(self, ids, mag=None, magmask=None):
        ids = np.asarray(ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.shape[1] != self.W:
            raise ValueError(f"windows must be {self.W} tokens wide, got {ids.shape[1]}")
        E_ids = np.asarray(ids)
        x = T.embedding(self.emb, E_ids, self.W, self.D)
        if self.attn is not None:
            x = self.attn(x)
        h = T.sigmoid(self.concept(x))
        logits = self.answer(h)
        if self.direct and self.dmode == "tied":
            # full-rank per-position mix, then the TIED readout: logits += relu(x @ dmix) @ E^T
            hmix = T.relu(T.matmul(x, self.dmix))
            logits = T.add(logits, T.matmul(hmix, T.transpose(self.emb)))
        elif self.direct:
            # low-rank position-aware path (kept for comparison; measured NOT sufficient at rank 16)
            logits = T.add(logits, T.matmul(T.relu(T.matmul(x, self.dproj)), self.dout))
        out = {"concepts": h, "logits": logits, "window_input": x}
        if self.answer2 is not None:
            out["logits2"] = self.answer2(h)
        if self.mag is not None:
            if mag is None:
                raise ValueError("this model has the numeric path ON (numeric=True) but no magnitude "
                                 "channel was passed: the numbers ARE the input, there is no way to "
                                 "recover them from the token ids of a vocabulary that never saw them.")
            m = np.asarray(mag, dtype=T.current_dtype()).reshape(E_ids.shape)
            if magmask is None:
                # fall back to "value != 0" and SAY so: with this mask a token "0" would be treated as a
                # non-number, which is exactly the memorisation path the mask exists to close.
                keep_np = (m != 0.0).astype(T.current_dtype())
            else:
                keep_np = np.asarray(magmask, dtype=T.current_dtype()).reshape(E_ids.shape)
            out["magnitudes"] = self.mag(x, T.Tensor(backend.to_dev(m)),
                                         keep=T.Tensor(backend.to_dev(keep_np)))
            out["num"] = self.num(out["magnitudes"])
        return out

    def forward_arrays(self, ids, mag=None, magmask=None):
        """Just the numbers, on the HOST. For evaluation, explanation, and logging.

        Everything crossing this boundary comes back to NumPy, so `explain()`, the audit, and any
        printed number never accidentally mix a cupy array with a numpy one -- which cupy refuses
        with a confusing "Unsupported type <class 'numpy.ndarray'>".
        """
        out = self.forward(np.asarray(ids), mag=mag, magmask=magmask)
        res = {"concepts": backend.to_cpu(out["concepts"].data),
               "logits": backend.to_cpu(out["logits"].data)}
        if "num" in out:
            res["magnitudes"] = backend.to_cpu(out["magnitudes"].data)
            res["num"] = backend.to_cpu(out["num"].data)
        return res

    # ---------------------------------------------------------------- loss
    def loss(self, ids, y, use_concept_loss=True, soft_concepts="off", weights=None, y2=None,
             mag=None, ynum=None, lam_num=1.0, magmask=None, ynum_mask=None, concept_ids=None):
        """Cross-entropy on the next token, plus the auxiliary concept loss.

        `weights` (optional, one value per sample) re-weights each sample's CE term. Used for
        ASSISTANT-ONLY training: weight 1 on targets that are the assistant's own words, 0 on the
        user's, so the model stops spending capacity learning to re-predict the human's text.
        None means every sample counts equally -- the original behaviour, unchanged.

        `soft_concepts` selects the concept target and loss:
          "off"  -- hard 0/1 targets, balanced BCE. The original.
          "mse"  -- graded targets, squared error. SCORED 0/9 CONCEPTS: squared error on a rare
                    target optimises to the MEAN, so every neuron sat near zero and never fired.
                    It scored the best language numbers of any arm (ppl 99.313, acc 30.03%) for
                    exactly that reason -- a constant concept layer is hardly a bottleneck, which
                    is the trap rather than a win.
          "bce"  -- graded targets, LOG loss. The soft label cannot collapse to the mean because
                    the gradient stays large while the neuron is wrong, so a rare concept keeps
                    being pushed toward firing.
        """
        out = self.forward(ids, mag=mag, magmask=magmask)
        logits = out["logits"]
        ce = T.softmax_ce(logits, y, sample_weights=weights)

        parts = {"ce": float(backend.to_cpu(ce.data))}
        total = ce

        # THE NUMERIC TERM (2026-10-20): the answer as a QUANTITY, beside the token classifier. This is
        # the half of the numeric path that can answer 27 when 27 was never a training label. It only
        # exists when the caller supplies a numeric target - the chat corpus has none, and the term is
        # then simply absent rather than being quietly trained on a made-up target.
        # (CORRECTION 2026-10-20 18:0x: the chat corpus DOES have one, and it is not made up - the value
        # of the very token the classifier is asked to predict. See lens.numeric.numeric_targets and
        # lens/runs/NUMERIC-CHAT-PREDICTIONS.txt. The sentence above is kept and corrected rather than
        # rewritten, and the rule it protects is unchanged: no target, no term.)
        #
        # `ynum_mask` (optional, one value per sample) restricts the numeric error to the samples that HAVE
        # a numeric target and divides by that mask's own total, so the ~96% of chat windows with no
        # numeric target get no gradient instead of teaching the head to predict 0.
        if self.num is not None and ynum is not None:
            nm = num_mse(out["num"], ynum, mask=ynum_mask)
            parts["num_mse"] = float(backend.to_cpu(nm.data))
            parts["num_weight"] = float(lam_num)
            total = T.add(total, T.mul(nm, T.Tensor(np.asarray(lam_num, dtype=T.current_dtype()))))

        # SECOND-TOKEN TERM (mtp=2). Both heads read the same concept representation, so the model must
        # encode information that stays useful two steps ahead, not just for the immediate next word.
        if self.answer2 is not None and y2 is not None:
            ce2 = T.softmax_ce(out["logits2"], y2, sample_weights=weights)
            parts["ce2"] = float(backend.to_cpu(ce2.data))
            # explicit ops only: this engine has no operator overloading, so the weight is a constant
            # tensor and the terms are combined with T.mul / T.add, exactly like the concept terms below
            w2 = T.Tensor(np.asarray(self.lam_mtp, dtype=T.current_dtype()))
            total = T.add(total, T.mul(ce2, w2))

        if self.lam_concept > 0 and use_concept_loss:
            # `concept_ids` (2026-10-25 self-wake): the window whose CONCEPT LABELS are the target. Defaults to
            # `ids`, so every existing caller is unchanged. The self-context (exposure-bias) trainer passes the
            # ORIGINAL window here while its CE term sees the rolled-in one, so that lever changes the CE input
            # only and does NOT silently re-aim the concept objective at the model's own generated text.
            ids_c = np.asarray(ids if concept_ids is None else concept_ids)
            h = out["concepts"]
            hd = backend.to_cpu(h.data)
            eps = 1e-7
            if soft_concepts == "mse":
                # GRADED target: the fraction of the window occupied by this concept's tokens.
                # A window with three speech verbs scores 3/8, not 1.0 -- so the neuron can learn
                # "a little" vs "a lot" instead of only "yes" vs "no".
                #
                # THIS FAILED, and the reason is worth keeping. Squared error on a RARE target has
                # its optimum at the MEAN. Most windows contain no speech words, so the cheapest
                # thing for every neuron was to sit near zero and never fire: the run scored 0/9
                # concepts. MSE quietly turned all nine into constants. It did help the language
                # model (ppl 99.313, acc 30.03%, both the best of any arm) because a constant
                # concept layer is barely a bottleneck at all -- which is the trap, not a win.
                ids_arr = ids_c
                tgt = np.stack(
                    [(np.isin(ids_arr, c.ids).sum(axis=1) / ids_arr.shape[1]) for c in self.concepts],
                    axis=1).astype(T.current_dtype())
                w = self.pos_weight[None, :]
                err = w * (hd - tgt) ** 2
                aux_val = float(err.mean())
                aux = T.Tensor(backend.to_dev(np.asarray(aux_val, dtype=T.current_dtype())),
                               requires_grad=h.requires_grad, _parents=(h,), _op="concept_mse",
                               _saved={"hd": h.data, "tgt": backend.to_dev(tgt),
                                       "w": backend.to_dev(w), "n": err.size})
                total = T.add(total, T.mul(aux, T.Tensor(np.asarray(self.lam_concept, dtype=T.current_dtype()))))
                parts["concept_mse"] = aux_val
                parts["concept_weight"] = float(self.lam_concept)
            else:
                # GRADED target with the LOG loss: the same soft label as the MSE branch, but the
                # BCE cannot collapse to the mean. Its gradient stays large while the neuron is
                # wrong, so a rare concept keeps being pushed toward firing.
                # `soft_concepts == "bce"` uses the graded fraction; anything else uses the hard 0/1.
                if soft_concepts == "bce":
                    ids_arr = ids_c
                    tgt = np.stack(
                        [(np.isin(ids_arr, c.ids).sum(axis=1) / ids_arr.shape[1])
                         for c in self.concepts], axis=1).astype(T.current_dtype())
                else:
                    tgt = np.stack([c.labels(ids_c) for c in self.concepts],
                                   axis=1).astype(T.current_dtype())
                # balanced BCE: the positive weight is APPLIED, and it is in the formula here
                w = self.pos_weight[None, :]
                bce = -(w * tgt * np.log(hd + eps) + (1.0 - tgt) * np.log(1.0 - hd + eps))
                aux_val = float(bce.mean())
                aux = T.Tensor(backend.to_dev(np.asarray(aux_val, dtype=T.current_dtype())),
                               requires_grad=h.requires_grad, _parents=(h,), _op="concept_bce",
                               _saved={"hd": h.data, "tgt": backend.to_dev(tgt),
                                       "w": backend.to_dev(w), "n": bce.size})
                total = T.add(total, T.mul(aux, T.Tensor(np.asarray(self.lam_concept, dtype=T.current_dtype()))))
                parts["concept_aux"] = aux_val
                parts["concept_weight"] = float(self.lam_concept)

        if self.lam_act > 0:
            h = out["concepts"]
            sp = T.Tensor(backend.xp.abs(h.data).mean(), requires_grad=h.requires_grad,
                          _parents=(h,), _op="l1_act", _saved={"n": h.data.size})
            total = T.add(total, T.mul(sp, T.Tensor(np.asarray(self.lam_act, dtype=T.current_dtype()))))
            parts["sparsity_act"] = float(backend.to_cpu(sp.data))
        return total, parts


# the backward for the concept losses, registered on the shared op table
def _bw_concept_bce(out, parents, saved, grad):
    """Exact gradient of the weighted BCE that the forward actually computes.

    FORWARD (model.py / vision.py):  bce = -(w*t*log(h+eps) + (1-t)*log(1-h+eps)) ; aux = mean(bce)
    so                               d aux/d h = [ -w*t/(h+eps) + (1-t)/(1-h+eps) ] / n

    THE BUG THIS REPLACES (found 2026-10-10 by the numeric gradient check): the old formula was
        dh = w * (h - t) / (h * (1 - h))
    which equals the true gradient ONLY when w == 1. Expanding at t=0 it gives w/(1-h) where the
    truth is 1/(1-h) -- wrong by a factor of w. `pos_weight` is (1-support)/support, so it is 1 only
    for support=0.5. Every concept with a different support trained against a mis-scaled gradient:
    a concept with support 0.02 has w=49, so its ABSENT side was pushed 49x too hard, and a concept
    with support 0.9 has w=0.111. The check missed it for months because the test concepts were
    built with support=0.5, which makes w=1 and hides the error exactly. The gradient check now
    includes a non-0.5 support for this reason.
    """
    h = parents[0]
    if not h.requires_grad:
        return
    hd, tgt, w = saved["hd"], saved["tgt"], saved["w"]
    eps = 1e-7
    dh = (-w * tgt / (hd + eps) + (1.0 - tgt) / (1.0 - hd + eps)) / saved["n"]
    T._accumulate(h, dh * T._scalar(grad))


def _bw_concept_mse(out, parents, saved, grad):
    """Gradient of the weighted squared error: d/dh [ w * (h - t)^2 ] = 2 w (h - t)."""
    h = parents[0]
    if not h.requires_grad:
        return
    hd, tgt, w = saved["hd"], saved["tgt"], saved["w"]
    dh = (2.0 * w * (hd - tgt)) / saved["n"]
    T._accumulate(h, dh * T._scalar(grad))


def _bw_l1_act(out, parents, saved, grad):
    h = parents[0]
    if not h.requires_grad:
        return
    T._accumulate(h, backend.xp.sign(h.data) / saved["n"] * T._scalar(grad))


T.OPS["concept_bce"] = _bw_concept_bce
T.OPS["concept_mse"] = _bw_concept_mse
T.OPS["l1_act"] = _bw_l1_act


def forward_arrays(net, ids):
    """Module-level helper so concepts.py does not have to import the class."""
    return net.forward_arrays(ids)
