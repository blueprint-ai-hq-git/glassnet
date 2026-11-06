"""lens/numeric.py - THE MAGNITUDE PATH: numbers enter as numbers, answers leave as quantities.

WHERE THIS COMES FROM (measured, not assumed)
---------------------------------------------
STATE.md / lens/scripts/slot_binding_probe.py, on addition with operands 0..9 and scoring on 10..19:

    structure                                   train     held out   extrapolation
    presence + linear (what this architecture is) 65.0%      0.0%       0.0%
    slot binding + linear                         65.0%      0.0%       0.0%
    slot + interactions + linear                 100.0%      0.0%       0.0%
    digit-decomposed + MLP                       100.0%      0.0%       0.0%
    numeric + MLP                                 95.0%     65.0%       0.0%
    numeric + regression (answer as a quantity)   MAE 0.008  MAE 0.010  MAE 0.041, EXACT 100%

450 trials over method x size x lr x epochs x RL samples all scored 0% extrapolation. The blocker is
structural, and the structure that fixes it is: the operands arrive as QUANTITIES and the answer leaves as a
QUANTITY. A classifier over the labels it has seen can never emit a sum it has never seen.

WHAT IS ADDED, AND WHERE
------------------------
1. A MAGNITUDE CHANNEL beside the token-presence concepts. One number per window position:

       q_t = (learned scalar readout of the token embedding at t) + (the number the token denotes)

   The second term is FIXED and never trained. That is the point: an operand the model has never seen still
   arrives as its own value, so there is nothing to memorise and nothing to be untrained. The first term is
   what lets a WORD be read as a number ("three" -> 3) once it has been trained. Nothing here is hidden:
   the channel is the number itself, and `trace()` prints it beside the token that carried it.

2. A NUMERIC HEAD beside the token classifier. The classifier answers "which token comes next" over a fixed
   label set; the numeric head answers "what QUANTITY" and therefore has no ceiling. It is one affine layer
   over the magnitude vector, so a sum is expressible as weights 1 and 1 on the two operand positions.

3. The loss on the quantity is mean squared error, built from the ops the engine already has (sub, mul, sum)
   so it needs no new backward pass and its gradient is the one the numeric gradient gate already checks.

READABILITY IS KEPT: every magnitude is a number that can be printed with the token it came from, and the
head that uses it is one linear map. Nothing in this file is a black box.
"""
import re

import numpy as np

from lens import backend
from lens import tensor as T


_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _parse_decimal(tok):
    """The float a token is WRITTEN as, or None when the token is not written as a number.

    THE BUG THIS REPLACES (found 2026-10-20, the first time the numeric path was pointed at real text):
    the old rule was `float(str(tok))`, and Python's float() accepts its own SPECIAL SPELLINGS - "inf",
    "infinity", "nan". So the ordinary English word "infinity" (token id 8044 of the chat vocabulary,
    8,561 occurrences in data/chat/prepared_mix_ctrl_honest) counted as a NUMBER whose value is inf. With
    the numeric path on, that put inf in the MAGNITUDE CHANNEL at every position holding the word, and
    inf in any numeric TARGET whose next token was the word, which makes num_mse inf and every gradient
    nan. It could never show on the arithmetic harness, whose token map is "0".."38", "+", "=" - the same
    class of defect as the GPU-only crash in lens/opt.py: a component fed one input distribution has not
    been tested on the other.

    THE RULE NOW: a token is a number iff it is WRITTEN as a decimal literal - optional sign, digits, an
    optional fractional part. "38" yes, "-2" yes, "3.5" yes, "000" yes (it is 0); "infinity" no, "inf"
    no, "nan" no, "1e5" no (the corpus tokenizer never produces that form, and accepting Python's
    spellings is exactly what let "infinity" through). A literal that PARSES to a non-finite float (a
    digit string longer than ~1e308) is not a number either.
    """
    if isinstance(tok, (int, float, np.integer, np.floating)):
        v = float(tok)
        return v if np.isfinite(v) else None
    s = str(tok).strip()
    if not _DECIMAL_RE.match(s):
        return None
    v = float(s)
    return v if np.isfinite(v) else None


def token_value(tok):
    """The quantity a token denotes, or 0.0. Readable by construction: it IS the number.

    Accepts the token text ("13", "-2", "3.5") and returns the float it denotes. Anything that is not
    WRITTEN as a number returns 0.0 rather than raising, because most tokens are words and the channel
    must be defined for every position of every window. See _parse_decimal for what counts as a number
    and for the "infinity" bug that made this rule strict.
    """
    v = _parse_decimal(tok)
    return 0.0 if v is None else v


def window_divisor(values):
    """Per-window divisor for the BOUNDED magnitude representation: d = max(1, max_j |v_j|).

    Returns an (N, 1) array in the SAME dtype as `values`, so dividing a float32 window by it keeps float32
    (a dtype promotion here would change every downstream number for no reason).

    WHY A WINDOW'S OWN MAXIMUM IS A LEGITIMATE SCALE. The divisor is computed from the window's numbers, and
    the window is the INPUT - nothing is fitted on train data, so no train/val leakage is possible and there
    is nothing to learn. What it buys, measured (lens/runs/NUMERIC-BOUNDED-RANGE.txt): the channel and the
    numeric target are bounded by 1 whatever the corpus contains, which is what the chat corpus needs and the
    arithmetic corpus never needed.

    WHY NOT LOG COMPRESSION (the other option in PLAN.txt item (d3b)): a linear head over log-compressed
    magnitudes cannot express a sum - log1p(a+b) != log1p(a) + log1p(b) (at a=b=9: 2.944 against 4.605) - so
    the arithmetic lever would be broken by construction. Dividing a whole window by ONE number keeps every
    linear combination linear, so a sum is still expressible (weights 1 and 1 on the two operand positions,
    whatever d is). The 9x9 figure is a measured line in NUMERIC-BOUNDED-RANGE.txt.
    """
    a = np.asarray(values)
    if a.size == 0:
        return np.ones((a.shape[0], 1), dtype=a.dtype)
    m = np.max(np.abs(a), axis=1, keepdims=True)
    return np.maximum(np.float32(1.0), m).astype(a.dtype, copy=False)


def magnitude_channel(ids, id_to_token, scale=1.0, bound=None):
    """(values, mask) for a batch of windows.

    `values` is the quantity at each position, divided by `scale` for conditioning - or, when bound="window",
    divided by the window's own divisor d = max(1, max_j |v_j|) from window_divisor(), in which case `scale`
    is IGNORED because a fixed scale cannot bound a range the corpus does not bound. Readout inverts it by
    multiplying by d, a fixed transform, exactly as it multiplies by `scale` today.
    `mask` is 1.0 where the token IS a number and 0.0 elsewhere. The mask is separate from the values on
    purpose: the token "0" is a number whose value is 0.0, and treating "no value" and "the value zero" as
    the same thing would hand the model a memorisation path at exactly that position.

    `id_to_token` maps a token id to the token (text or number). Non-number tokens get value 0.0, which is
    honest: there is no magnitude there.
    """
    ids = np.asarray(ids)
    vals = np.zeros(ids.shape, dtype=np.float32)
    mask = np.zeros(ids.shape, dtype=np.float32)
    for tid in np.unique(ids):
        tok = id_to_token[int(tid)]
        v = token_value(tok)
        vals[ids == tid] = v
        mask[ids == tid] = 1.0 if _is_number(tok) else 0.0
    if bound == "window":
        return vals / window_divisor(vals), mask
    return vals / float(scale), mask


def _is_number(tok):
    """True when the token denotes a number at all (so "0" is a number and "the" is not)."""
    return _parse_decimal(tok) is not None


class MagnitudeLayer:
    """Per-position quantities: a learned readout of the embedding, PLUS the number itself.

    learned=True  (default) : q_t = mask_t * (w . emb_t + b_t) + value_t
    learned=False           : q_t = value_t          (no parameters at all)

    THE MASK IS NOT COSMETIC - IT WAS ADDED BECAUSE A MEASUREMENT DEMANDED IT. Without it the head was free
    to fit the training quantities using the LEARNED readout of the operand tokens (which is trained only for
    the tokens it has seen) instead of the numbers, and it did: measured on the addition harness
    (lens/runs/NUMERIC-PATH-DEBUG.txt), the head's weights on the two operand positions came out 0.15 instead
    of the 1.0 the sum needs, train error 0.00000, unseen MAE 22-26 - a perfect fit that answers nothing. At a
    position whose token IS a number, the quantity is KNOWN; letting a learned value be added there is a
    memorisation path for every number the vocabulary has seen and garbage for every number it has not. The
    mask keeps the learned readout for words ("three" -> 3, once trained) and forbids it where the number is
    already known.
    """

    def __init__(self, d, window, rng, name="mag", learned=True):
        self.D, self.W = int(d), int(window)
        self.learned = bool(learned)
        self.Wm = self.bm = None
        if self.learned:
            scale = (2.0 / self.D) ** 0.5
            self.Wm = T.Tensor(backend.to_dev(rng.standard_normal((self.D, 1)).astype(np.float32) * scale),
                               requires_grad=True, name=f"{name}.Wm")
            self.bm = T.Tensor(backend.to_dev(np.zeros(self.W, dtype=np.float32)),
                               requires_grad=True, name=f"{name}.bm")

    def __call__(self, x, mag, keep=None):
        """x: (N, W*D) flattened window embeddings. mag: (N, W) Tensor of quantities.

        `keep` is the (N, W) NUMERIC MASK: 1.0 where the token is not a number (the learned readout is
        allowed there), 0.0 where the token IS a number (the value is already known, and a learned value
        added on top of it is a memorisation path). None means "no mask", which is only correct when no
        position carries a known number.
        """
        if not self.learned:
            return mag
        N = int(x.data.shape[0])
        xr = T.reshape(x, (N * self.W, self.D))
        q = T.matmul(xr, self.Wm)                    # (N*W, 1)
        q = T.reshape(q, (N, self.W))                # one scalar per position
        q = T.add(q, self.bm)                        # per-position bias, (N,W) + (W,)
        if keep is not None:
            q = T.mul(q, keep)                       # learned readout only where no number is known
        return T.add(q, mag)                         # + the number the token denotes (fixed)

    def parameters(self):
        return [] if not self.learned else [self.Wm, self.bm]


class NumericHead:
    """The answer as a QUANTITY, beside the token classifier. One affine map over the magnitudes.

    A classifier over labels 0..18 can never emit 27. This head has no label set at all, which is exactly
    why the probe's regression arm reaches 100% exact extrapolation where every one-hot structure reaches 0%.
    """

    def __init__(self, window, rng, name="num"):
        self.W = int(window)
        scale = (1.0 / max(self.W, 1)) ** 0.5
        self.Wn = T.Tensor(backend.to_dev(rng.standard_normal((self.W, 1)).astype(np.float32) * scale),
                           requires_grad=True, name=f"{name}.Wn")
        self.bn = T.Tensor(backend.to_dev(np.zeros(1, dtype=np.float32)),
                           requires_grad=True, name=f"{name}.bn")

    def __call__(self, q):
        return T.add(T.matmul(q, self.Wn), self.bn)          # (N,1)

    def parameters(self):
        return [self.Wn, self.bn]


def num_mse(pred, target, mask=None):
    """Mean squared error on the answer AS A QUANTITY, composed from ops the engine already has.

    sub/mul/sum are all gradient-checked, so this adds no new backward pass to verify. `target` is a plain
    array of quantities, one per row of `pred`.

    `mask` (optional, one value per row) restricts the error to the rows that HAVE a numeric target and
    divides by the mask's own total, so the mean is over those rows only. WHY IT IS NEEDED (measured
    2026-10-20 on the real chat corpus, lens/runs/NUMERIC-CHAT-PREDICTIONS.txt): only ~4% of chat windows
    have a numeric next-token target, so an unmasked MSE would spend 96% of its gradient telling the head
    to predict 0 everywhere - it would destroy the very signal the term exists to provide. Rows with
    mask 0 contribute exactly 0 to the loss AND to the normaliser, so they get no gradient at all.
    """
    tgt = np.asarray(target, dtype=T.current_dtype()).reshape(pred.data.shape)
    t = T.Tensor(backend.to_dev(tgt))
    d = T.sub(pred, t)
    sq = T.mul(d, d)
    if mask is None:
        n = float(pred.data.shape[0])
    else:
        mk = np.asarray(mask, dtype=T.current_dtype()).reshape(pred.data.shape)
        sq = T.mul(sq, T.Tensor(backend.to_dev(mk)))
        n = float(np.sum(mk))
    s = T.sum_all(sq)
    return T.mul(s, T.Tensor(np.asarray(1.0 / max(n, 1.0), dtype=T.current_dtype())))


def numeric_targets(vocab, Y, scale=1.0, divisor=None):
    """(values, mask) for a batch of NEXT-TOKEN ids: what QUANTITY the next token denotes.

    THIS IS ITEM (d3) OF PLAN.txt. Until now the numeric head could only be trained on the arithmetic
    harness, because "the chat corpus has none [no numeric target]" (lens/model.py). It has one, and it is
    not invented: for every window the classifier is already trained to predict the next TOKEN, this gives
    the same target AS A NUMBER. So the quantity head learns "how much" while the classifier learns
    "which", on the same data, with nothing constructed for the test.

    `values` is the token's value divided by `scale` (the same scale the magnitude channel uses, so the
    head's input and its target are in the same units), and `mask` is 1.0 only where the target token IS a
    number. Everywhere else the value is 0.0 and the mask is 0.0, which is honest: there is no quantity
    there, and num_mse(mask=...) gives those rows no gradient.

    MEASURED ON THE REAL CORPUS (lens/scripts/probe_numeric_targets.py): 329 of the 12,002 chat vocabulary
    tokens are written as numbers; 4.55% of train windows and 3.98% of val windows carry a numeric target;
    the values run from 0 to 8000 (median 4, p99 2020). No value is non-finite - the rule that makes that
    true is _parse_decimal, and before it "infinity" made the whole term inf.
    """
    Y = np.asarray(Y)
    vmap = np.zeros(len(vocab), dtype=np.float32)
    mmap = np.zeros(len(vocab), dtype=np.float32)
    for i, tok in enumerate(vocab):
        v = _parse_decimal(tok)
        if v is not None:
            vmap[i] = v
            mmap[i] = 1.0
    return vmap[Y] / float(scale), mmap[Y]


def trace(magnitudes, ids, id_to_token, rows=2):
    """Print what the magnitude channel carried, position by position. Readable, not opaque."""
    m = np.asarray(backend.to_cpu(magnitudes))
    ids = np.asarray(ids)
    out = []
    for i in range(min(rows, m.shape[0])):
        pairs = [(str(id_to_token[int(ids[i, j])]), round(float(m[i, j]), 4)) for j in range(m.shape[1])]
        out.append(" ".join(f"{tok}:{val}" for tok, val in pairs))
    return out
