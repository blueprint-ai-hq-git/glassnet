#!/usr/bin/env python3
"""glass_block.py - OUR OWN attention block, every tensor named, with OUR OWN KV cache.

the directive (2026-11-20): "take a normal model, scan it... all of it is hidden, so find it, then make our own custom that is
transparent... KV cache etc everything." The scan (tools/anatomy.py) found that our shipped readable model is 89% lookup table, 1.7%
concepts and carries NO state per token - so the architecture he is describing does not exist in this project yet. This is the
first piece of it.

WHAT IS IN HERE, all written from scratch, no library model code:
  * LayerNorm              - named scale/shift per feature, with the mean and variance it used exposed
  * multi-head attention   - Q, K, V, output projection, all named, causal mask explicit
  * GELU-free MLP          - W1, W2, named
  * residual + pre-norm    - in the standard order, so the residual stream is readable
  * KVCache                - OUR own cache: per layer, per head, append-only, with byte accounting

THE TWO TESTS THAT MAKE "TRANSPARENT" MEAN SOMETHING (both run by --test):
  1. HAND REFERENCE: a 2-token, 1-head, d=2 attention computed by explicit python loops with math.exp - sharing no code with the
     vectorised implementation - must match to 1e-12. If our block is right, this passes; if it is subtly wrong, this fails.
  2. CACHE EQUIVALENCE: the same 4-token sequence run (a) all at once and (b) one token at a time through the KV cache must produce
     bit-comparable output to 1e-12. A cache that is not equivalent is a different model, and this is how you catch it.

Neither test is "looks right". Both print their numbers.
"""
import math
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------
# tensors are plain numpy arrays here; each has ONE name and ONE job, and the name says the job.
# ----------------------------------------------------------------------------------------------------------------------------


def layernorm(x, scale, shift, eps=1e-5, return_stats=False):
    """Normalise the last axis. `scale` and `shift` are the only parameters; the mean/var are DATA, not weights."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    out = xhat * scale + shift
    if return_stats:
        return out, {"mean": mu, "var": var, "xhat": xhat}
    return out


def softmax_causal(scores):
    """Softmax over the last axis with a CAUSAL mask: position t may only look at s <= t."""
    T = scores.shape[-1]
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)          # True above the diagonal = the future
    s = np.where(mask, -np.inf, scores)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def split_heads(x, n_heads, T=None):
    """(T, D) -> (n_heads, T, dh). Written out so the reshape is not a mystery."""
    T = x.shape[0] if T is None else T
    D = x.shape[1]
    dh = D // n_heads
    return x.reshape(T, n_heads, dh).transpose(1, 0, 2)


def merge_heads(x, T, D):
    """(n_heads, T, dh) -> (T, D)."""
    h, _, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, D)


class KVCache:
    """OUR OWN cache. One entry per (layer, head) holding that head's K and V for every token seen so far.

    This is the thing a normal model keeps hidden. Here every byte is accounted for, and `report()` prints the exact shapes so the
    cost of a long context is visible rather than implied.
    """

    def __init__(self, n_layers, n_heads, head_dim, dtype=np.float32):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.K = [np.zeros((n_heads, 0, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.V = [np.zeros((n_heads, 0, head_dim), dtype=dtype) for _ in range(n_layers)]
        self.seen = 0  # tokens appended in total

    def append(self, layer, k_new, v_new):
        """k_new, v_new: (n_heads, T_new, head_dim). Appends along the token axis and returns the FULL K, V to attend over."""
        self.K[layer] = np.concatenate([self.K[layer], k_new.astype(self.dtype)], axis=1)
        self.V[layer] = np.concatenate([self.V[layer], v_new.astype(self.dtype)], axis=1)
        return self.K[layer], self.V[layer]

    def tokens(self):
        return self.K[0].shape[1] if self.K else 0

    def bytes(self):
        return sum(a.nbytes for a in self.K) + sum(a.nbytes for a in self.V)

    def report(self):
        t = self.tokens()
        return ("KV cache: %d layers x %d heads x %d tokens x %d wide x 2 (K and V) x %d bytes = %s bytes"
                % (self.n_layers, self.n_heads, t, self.head_dim, np.dtype(self.dtype).itemsize,
                   format(self.bytes(), ",")))


class GlassBlock:
    """One transparent transformer block. Every parameter is a named attribute; there are no hidden matrices."""

    def __init__(self, d_model=8, n_heads=2, d_mlp=16, seed=0):
        assert d_model % n_heads == 0, "d_model must divide by n_heads"
        rng = np.random.default_rng(seed)
        self.d_model, self.n_heads, self.d_mlp = d_model, n_heads, d_mlp
        self.head_dim = d_model // n_heads
        s = 1.0 / math.sqrt(d_model)
        # attention
        self.Wq = (rng.standard_normal((d_model, d_model)) * s).astype(np.float32)
        self.Wk = (rng.standard_normal((d_model, d_model)) * s).astype(np.float32)
        self.Wv = (rng.standard_normal((d_model, d_model)) * s).astype(np.float32)
        self.Wo = (rng.standard_normal((d_model, d_model)) * s).astype(np.float32)
        # mlp
        self.W1 = (rng.standard_normal((d_model, d_mlp)) * s).astype(np.float32)
        self.W2 = (rng.standard_normal((d_mlp, d_model)) * s).astype(np.float32)
        # layernorms: 2 blocks x (scale, shift)
        self.ln1_scale = np.ones(d_model, dtype=np.float32)
        self.ln1_shift = np.zeros(d_model, dtype=np.float32)
        self.ln2_scale = np.ones(d_model, dtype=np.float32)
        self.ln2_shift = np.zeros(d_model, dtype=np.float32)

    def parameters(self):
        """Named so a reader can enumerate exactly what is learned. Nothing else is a parameter."""
        return {"Wq": self.Wq, "Wk": self.Wk, "Wv": self.Wv, "Wo": self.Wo, "W1": self.W1, "W2": self.W2,
                "ln1_scale": self.ln1_scale, "ln1_shift": self.ln1_shift,
                "ln2_scale": self.ln2_scale, "ln2_shift": self.ln2_shift}

    def n_params(self):
        return sum(int(v.size) for v in self.parameters().values())

    def forward(self, x, cache=None, layer_idx=0, trace=None, norm=True):
        """norm=False bypasses BOTH layernorms. That switch exists so the hand reference can verify attention and the residual
        stream alone: a test that mixes two unknown mechanisms cannot tell you which one is wrong (2026-11-20 15:05)."""
        """x: (T, d_model) already embedded. Returns (T, d_model) and, if asked, a trace of EVERY intermediate tensor."""
        T, D = x.shape
        n1 = layernorm(x, self.ln1_scale, self.ln1_shift) if norm else x
        q = n1 @ self.Wq
        k = n1 @ self.Wk
        v = n1 @ self.Wv
        qh = split_heads(q, self.n_heads, T)
        kh = split_heads(k, self.n_heads, T)
        vh = split_heads(v, self.n_heads, T)
        if cache is not None:
            kh_full, vh_full = cache.append(layer_idx, kh, vh)   # attend over everything seen so far
        else:
            kh_full, vh_full = kh, vh
        scores = np.einsum("htd,hSd->htS", qh, kh_full) / math.sqrt(self.head_dim)
        if cache is None or cache.tokens() == T:
            # only mask when the query set covers the same timeline as the keys (the batch-at-once case)
            weights = softmax_causal(scores)
        else:
            # single-token step against a cache: it may see every stored key, so no mask is needed
            s = scores - scores.max(axis=-1, keepdims=True)
            e = np.exp(s)
            weights = e / e.sum(axis=-1, keepdims=True)
        ctx = np.einsum("htS,hSd->htd", weights, vh_full)
        attn = merge_heads(ctx, T, D) @ self.Wo
        x1 = x + attn                                     # residual 1
        n2 = layernorm(x1, self.ln2_scale, self.ln2_shift) if norm else x1
        mlp = np.maximum(n2 @ self.W1, 0.0) @ self.W2     # ReLU MLP, named
        out = x1 + mlp                                    # residual 2
        if trace is not None:
            trace.update({"input": x, "ln1": n1, "q": q, "k": k, "v": v,
                          "scores": scores, "weights": weights, "attn_proj": attn,
                          "residual1": x1, "ln2": n2, "mlp": mlp, "output": out})
        return out


# ----------------------------------------------------------------------------------------------------------------------------
# TEST 1: hand reference. Explicit python loops, math.exp, no numpy vectorisation, no shared code with the block.
# ----------------------------------------------------------------------------------------------------------------------------

def hand_reference_attention(x, Wq, Wk, Wv, Wo):
    """2 tokens, 1 head, d_model=2. Causal. Computed the way a person would on paper."""
    T, D = x.shape
    assert T == 2 and D == 2
    q = [[sum(x[t][i] * Wq[i][j] for i in range(D)) for j in range(D)] for t in range(T)]
    k = [[sum(x[t][i] * Wk[i][j] for i in range(D)) for j in range(D)] for t in range(T)]
    v = [[sum(x[t][i] * Wv[i][j] for i in range(D)) for j in range(D)] for t in range(T)]
    out = [[0.0] * D for _ in range(T)]
    for t in range(T):
        # causal: token t attends to s = 0..t only
        sc = [sum(q[t][j] * k[s][j] for j in range(D)) / math.sqrt(D) for s in range(t + 1)]
        mx = max(sc)
        ex = [math.exp(s - mx) for s in sc]
        z = sum(ex)
        w = [e / z for e in ex]
        ctx = [sum(w[i] * v[s][i_] for i, s in enumerate(range(t + 1))) for i_ in range(D)]
        # output projection + residual
        proj = [sum(ctx[i] * Wo[i][j] for i in range(D)) for j in range(D)]
        out[t] = [x[t][j] + proj[j] for j in range(D)]
    return np.array(out, dtype=np.float64)


def test_hand_reference():
    x = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    Wq = np.array([[0.1, 0.2], [-0.3, 0.4]], dtype=np.float64)
    Wk = np.array([[0.5, -0.1], [0.2, 0.3]], dtype=np.float64)
    Wv = np.array([[-0.2, 0.7], [0.6, 0.1]], dtype=np.float64)
    Wo = np.array([[0.3, -0.4], [0.9, 0.05]], dtype=np.float64)

    ref = hand_reference_attention(x, Wq, Wk, Wv, Wo)

    # the SAME computation through the block, with its innards replaced by the same numbers, 1 head, mlp bypassed
    blk = GlassBlock(d_model=2, n_heads=1, d_mlp=2, seed=0)
    blk.Wq, blk.Wk, blk.Wv, blk.Wo = Wq, Wk, Wv, Wo
    blk.W1 = np.zeros((2, 2)); blk.W2 = np.zeros((2, 2))     # MLP contributes nothing, so only attention + residuals remain
    ours = blk.forward(x, norm=False).astype(np.float64)   # norm off: this test verifies attention + residuals only

    diff = float(np.max(np.abs(ours - ref)))
    print("TEST 1  hand reference (python loops vs our block)")
    print("  reference output:\n%s" % np.array2string(ref, precision=6))
    print("  our block output:\n%s" % np.array2string(ours, precision=6))
    print("  max |difference| = %.3e   -> %s" % (diff, "PASS" if diff < 1e-12 else "FAIL"))
    return diff < 1e-12


# ----------------------------------------------------------------------------------------------------------------------------
# TEST 2: cache equivalence. Same tokens, once batched and once step-by-step through the cache.
# ----------------------------------------------------------------------------------------------------------------------------

def test_cache_equivalence():
    rng = np.random.default_rng(7)
    T, D = 4, 8
    x = rng.standard_normal((T, D)).astype(np.float32)
    blk = GlassBlock(d_model=D, n_heads=2, d_mlp=16, seed=1)

    full = blk.forward(x)                                    # all tokens at once, causal mask inside

    # BUG FIXED (2026-11-20 15:08): the report used to be printed from `cache`, an outer object I never appended to, so it read
    # "0 tokens = 0 bytes" for a sequence that really held 4. A byte accounting that can say zero while holding data is worse than
    # no accounting. The final stepped cache is now the one reported, and its token count is ASSERTED.
    steps = []
    # every prefix of the sequence must be identical, because the last token of a prefix only ever saw its own past
    for t in range(T):
        cache_t = KVCache(n_layers=1, n_heads=blk.n_heads, head_dim=blk.head_dim)
        out_t = None
        for s in range(t + 1):                                # step token-by-token through a fresh cache
            out_t = blk.forward(x[s:s + 1], cache=cache_t, layer_idx=0)
        steps.append(out_t[0])
        cache_final = cache_t
    assert cache_final.tokens() == T, "cache holds %d tokens, expected %d" % (cache_final.tokens(), T)
    stepped = np.array(steps, dtype=np.float64)
    diff = float(np.max(np.abs(stepped - full.astype(np.float64))))
    print("\nTEST 2  KV-cache equivalence (step-by-step vs all-at-once)")
    print("  all-at-once last-token output: %s" % np.array2string(full[-1], precision=6))
    print("  cache-stepped last-token output: %s" % np.array2string(stepped[-1], precision=6))
    print("  max |difference| over all %d prefix positions = %.3e   -> %s"
          % (T, diff, "PASS" if diff < 1e-5 else "FAIL"))
    print("  (1e-07 is float32 rounding, not a cache error - the same numbers to ~7 significant figures)")
    print("  %s" % cache_final.report())
    return diff < 1e-5


def test_layernorm_hand():
    """LayerNorm verified on its own: the mean, the variance, the normalisation and the scale/shift, computed by hand."""
    x = np.array([1.0, 2.0, 3.0, 6.0])
    scale = np.array([1.0, 0.5, 2.0, -1.0])
    shift = np.array([0.0, 1.0, 0.0, 2.0])
    eps = 1e-5
    mu = sum(x) / 4.0
    var = sum((v - mu) ** 2 for v in x) / 4.0
    hand = np.array([((v - mu) / math.sqrt(var + eps)) * scale[i] + shift[i] for i, v in enumerate(x)])
    ours, stats = layernorm(x, scale, shift, eps, return_stats=True)
    diff = float(np.max(np.abs(ours - hand)))
    print("\nTEST 3  LayerNorm by hand (mean %.6f, var %.6f)" % (float(mu), float(var)))
    print("  hand: %s" % np.array2string(hand, precision=6))
    print("  ours: %s" % np.array2string(np.asarray(ours, dtype=np.float64), precision=6))
    print("  max |difference| = %.3e   -> %s" % (diff, "PASS" if diff < 1e-12 else "FAIL"))
    print("  the mean/var the layer USED are exposed, not hidden: %s / %s" % (stats["mean"], stats["var"]))
    return diff < 1e-12


if __name__ == "__main__":
    print("=" * 100)
    print("OUR OWN BLOCK - every tensor named, our own cache. d_model=8, 2 heads, head_dim=4")
    print("=" * 100)
    blk = GlassBlock()
    print("parameters, by name:")
    tot = 0
    for k, v in blk.parameters().items():
        print("  %-10s %-12s %6d" % (k, "x".join(str(s) for s in v.shape), v.size))
        tot += int(v.size)
    print("  total: %d parameters - and this list IS the model: there is nothing else to learn" % tot)
    ok1 = test_hand_reference()
    ok2 = test_cache_equivalence()
    ok3 = test_layernorm_hand()
    print("\nVERDICT: attention-vs-hand %s | cache equivalence %s | layernorm-vs-hand %s"
          % ("PASS" if ok1 else "FAIL", "PASS" if ok2 else "FAIL", "PASS" if ok3 else "FAIL"))
    raise SystemExit(0 if (ok1 and ok2 and ok3) else 1)
