
"""lens/kvcache.py - OUR OWN KV cache, so the model remembers what it has already written.

WHY THIS EXISTS (Yahya, 2026-10-20): "every word is needed to be recomputed from start, so how the hell
will the model get smart when it does not even remember what it wrote". He is right, and it is not only a
speed issue. Without a cache, generating token N throws away everything computed for tokens 1..N-1, so the
model carries no state at all from one word to the next.

WHY IT IS CORRECT BY CONSTRUCTION: our attention is CAUSAL. Position i's output depends only on positions
0..i. So once position i has been computed it can never change, no matter what comes later. That means the
cache may keep, for every token already seen:
    K_i  the key          (D floats)
    V_i  the value        (D floats)
    O_i  the finished contextualised output for that position (D floats)
and a new token only needs its OWN q, k, v, attended against the kept keys.

WHY O_i IS KEPT TOO: our concept layer reads the contextualised tokens of the whole window, not just the
last one. Keeping O_i means the window of contextualised tokens can be rebuilt exactly, so the concepts, the
logits and explain() are unchanged.

WHAT IT MUST PROVE (and does not count as done without):
  1. the cached path is IDENTICAL to the full recompute on the same input, not merely close
  2. bytes kept per token, and the cache size for a 512-token context
  3. seconds per generated token, cached versus full recompute
"""
import os
import sys
import time
import numpy as np

# the project root must be importable when this file is run directly (it lives inside lens/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class AttnCache:
    """Keeps K, V and the finished output O for every token already seen."""

    def __init__(self, d, window, capacity=None):
        self.D, self.W = int(d), int(window)
        self.capacity = int(capacity) if capacity else self.W   # how many past tokens to keep
        self.K = np.zeros((0, self.D), dtype=np.float32)
        self.V = np.zeros((0, self.D), dtype=np.float32)
        self.O = np.zeros((0, self.D), dtype=np.float32)

    @property
    def n(self):
        return int(self.K.shape[0])

    def bytes_per_token(self):
        return 3 * self.D * 4      # K, V and O, float32

    def total_bytes(self):
        return self.n * self.bytes_per_token()

    def append(self, k, v, o):
        self.K = np.concatenate([self.K, k[None, :]], axis=0)
        self.V = np.concatenate([self.V, v[None, :]], axis=0)
        self.O = np.concatenate([self.O, o[None, :]], axis=0)
        if self.capacity and self.n > self.capacity:
            keep = self.n - self.capacity
            self.K, self.V, self.O = self.K[keep:], self.V[keep:], self.O[keep:]

    def window_outputs(self):
        """The contextualised outputs of the last W positions, in order (what the concept layer reads)."""
        return self.O[-self.W:]


def cached_step(x_row, attn, cache, position=None):
    """One incremental step: the contextualised output for a single new token.

    x_row : (D,) the new token's embedding
    attn  : the model's CausalSelfAttention block (we read its weights, we do not change it)
    """
    Wq, Wk, Wv, Wo = attn.Wq.data, attn.Wk.data, attn.Wv.data, attn.Wo.data
    q = x_row @ Wq                       # (D,) scaled already (scaling is folded into the init)
    k = x_row @ Wk
    v = x_row @ Wv
    cache.append(k, v, np.zeros_like(v))  # O filled below
    keys = cache.K                        # all past keys, this one included
    scores = keys @ q                     # (n,) q @ k^T for every kept key
    scores = scores - scores.max()        # stable softmax
    p = np.exp(scores)
    p = p / p.sum()
    o = p @ cache.V                       # (D,)
    cache.O[-1] = o @ Wo
    return cache.O[-1]



# ---------------------------------------------------------------------------------------------
# SPEED REWRITE (2026-10-20). The first version was 14x SLOWER than a full window recompute:
# 7.66 ms/token against 0.51 ms. Cause: np.concatenate rebuilt three arrays on EVERY token, and the
# softmax ran over the whole cache every step. This version preallocates the buffers once and writes
# into them, and the benchmark now breaks the time down per operation so the cost is MEASURED, not guessed.
# ---------------------------------------------------------------------------------------------
class FastAttnCache:
    """Preallocated K/V/O store. No per-token allocation, no concatenate."""

    def __init__(self, d, window, capacity=None):
        self.D, self.W = int(d), int(window)
        self.capacity = int(capacity) if capacity else self.W
        self.K = np.zeros((self.capacity, self.D), dtype=np.float32)
        self.V = np.zeros((self.capacity, self.D), dtype=np.float32)
        self.O = np.zeros((self.capacity, self.D), dtype=np.float32)
        self.n = 0
        self._buf = np.empty(self.capacity, dtype=np.float32)   # reused softmax scratch

    def bytes_per_token(self):
        return 3 * self.D * 4

    def write(self, k, v):
        if self.n < self.capacity:
            i = self.n
            self.n += 1
        else:
            # full: shift one slot (cheap at W=32) and reuse the last slot
            self.K[:-1] = self.K[1:]; self.V[:-1] = self.V[1:]
            i = self.capacity - 1
        self.K[i] = k; self.V[i] = v
        return i


def fast_step(x_row, attn, cache):
    """One incremental step, preallocated. Returns the contextualised output for the new token."""
    Wq, Wk, Wv, Wo = attn.Wq.data, attn.Wk.data, attn.Wv.data, attn.Wo.data
    q = x_row @ Wq
    k = x_row @ Wk
    v = x_row @ Wv
    i = cache.write(k, v)
    scores = cache.K[:cache.n] @ q                       # (n,)
    scores -= scores.max()
    s = cache._buf[:cache.n]
    np.exp(scores, out=s)
    s /= s.sum()
    o = s @ cache.V[:cache.n]
    cache.O[i] = o @ Wo
    return cache.O[i]


def full_attention_outputs(x_win, attn):
    """The reference: the model's own attention block on the whole window. No cache, full recompute."""
    import lens.tensor as T
    from lens import backend
    N, WD = 1, x_win.shape[0] * x_win.shape[1]
    xr = x_win.reshape(1, WD)
    t = T.Tensor(backend.to_dev(xr.astype(np.float32)), requires_grad=False)
    out = attn(t)
    arr = out.data
    for fn in ("to_host", "to_cpu", "asnumpy", "to_numpy"):
        if hasattr(backend, fn):
            arr = getattr(backend, fn)(arr); break
    return np.asarray(arr, dtype=np.float32).reshape(x_win.shape[0], x_win.shape[1])


def main():
    import lens.api as api
    # an ATTENTION ship is required: the clean models are no-attention, so the cache has nothing to hook into
    ship = sys.argv[1] if len(sys.argv) > 1 else "lens/runs/chat/sb_B1_d512_attn_ep96.npz"
    net = api.load(ship, device="cpu")
    assert net.attn is not None, "this ship has no attention block; the cache is meaningless for it"
    D, W = net.D, net.W
    print(f"ship {ship.split('/')[-1]}  d={D}  window={W}  attention=yes")

    rng = np.random.default_rng(0)
    x_win = rng.standard_normal((W, D)).astype(np.float32)

    ref = full_attention_outputs(x_win, net.attn)          # the model's own block, full recompute

    cache = AttnCache(D, W, capacity=W)
    got = []
    for i in range(W):
        got.append(cached_step(x_win[i], net.attn, cache, i))
    got = np.asarray(got)

    diff = np.abs(got - ref)
    print()
    print("CORRECTNESS (cached step by step versus the model's own full recompute):")
    print(f"  max absolute difference: {diff.max():.3e}")
    print(f"  identical to float32 precision: {bool(diff.max() < 1e-5)}")
    print(f"  cache kept {cache.n} tokens of {W}")
    print()
    print("MEMORY:")
    print(f"  bytes per token kept: {cache.bytes_per_token()} ({cache.bytes_per_token()/1024:.2f} KB)")
    print(f"  cache for a 512-token context: {512*cache.bytes_per_token()/1024:.1f} KB")
    print(f"  what the OLD path would touch per generated token: W*D*3 projections = {W*D*3} values")
    print(f"  what the cached path touches: D*3 = {D*3} values  (factor {W}x less work per token)")
    print()
    print("SPEED - all numbers PER GENERATED TOKEN (this is what matters):")
    # generation: the old decode path calls the full window recompute once per new token
    reps = 60
    t0 = time.time()
    for _ in range(reps):
        full_attention_outputs(x_win, net.attn)
    ms_full_per_token = (time.time() - t0) / reps * 1000

    t0 = time.time()
    for _ in range(reps):
        c = AttnCache(D, W, capacity=W)
        for i in range(W):
            cached_step(x_win[i], net.attn, c)
    ms_concat_per_token = (time.time() - t0) / reps / W * 1000

    t0 = time.time()
    for _ in range(reps):
        c = FastAttnCache(D, W, capacity=W)
        for i in range(W):
            fast_step(x_win[i], net.attn, c)
    ms_fast_per_token = (time.time() - t0) / reps / W * 1000

    print(f"  full window recompute per generated token (the CURRENT decode path): {ms_full_per_token:.3f} ms")
    print(f"  first cache version (rebuilds arrays per token):                     {ms_concat_per_token:.3f} ms")
    print(f"  preallocated cache (this version):                                   {ms_fast_per_token:.3f} ms")
    print(f"  -> preallocated cache is {ms_full_per_token/max(ms_fast_per_token,1e-9):.2f}x faster than the current path")
    print(f"  -> tokens per second, cached: {1000.0/max(ms_fast_per_token,1e-9):.0f}   "
          f"uncached: {1000.0/max(ms_full_per_token,1e-9):.0f}")

    # where the remaining time goes, measured rather than guessed
    Wq, Wk, Wv, Wo = net.attn.Wq.data, net.attn.Wk.data, net.attn.Wv.data, net.attn.Wo.data
    x = x_win[0]
    def timeit(fn, reps=3000):
        t0 = time.time()
        for _ in range(reps):
            fn()
        return (time.time() - t0) / reps * 1e6
    cache = FastAttnCache(D, W, capacity=W)
    for i in range(W):
        fast_step(x_win[i], net.attn, cache)
    print(f"  breakdown per token: projections {timeit(lambda: (x @ Wq, x @ Wk, x @ Wv)):.1f} us | "
          f"attend+values {timeit(lambda: (cache.K[:cache.n] @ x, cache.V[:cache.n] @ x)):.1f} us | "
          f"output proj {timeit(lambda: x @ Wo):.1f} us")


# ---------------------------------------------------------------------------------------------
# WIRED INTO DECODING (2026-10-20, PLAN item (e)). The classes above keep K, V and the finished
# output O for every token seen, which is enough to PROVE a cache correct but not enough to DECODE:
# the concept layer reads the CONTEXTUALISED OUTPUTS OF THE WHOLE WINDOW (W positions), not only the
# newest one. A cache that returns one token's output would silently change the answer, which is the
# one thing a cache may never do. So the decoding cache keeps, for the last W tokens:
#
#     Q_i  the query  (needed to recompute the window's attention exactly, every step)
#     K_i  the key
#     V_i  the value
#     Vo_i = V_i @ Wo   the value ALREADY multiplied by the output projection
#
# WHY Vo: in the model the output projection is applied to the attention output of every position,
# (p @ V) @ Wo. Because (p @ V) @ Wo == p @ (V @ Wo) in exact arithmetic, folding Wo into V means the
# D x D output projection is applied ONCE PER NEW TOKEN instead of once per token per window position.
# THAT FOLD IS AN ASSOCIATION CHANGE, so the float32 rounding differs from the unfolded order. It is
# therefore a switch (`fold=False` runs the unfolded order) and the identity test measures both.
#
# THE SIZE IS FIXED, which is the point of a hot tier: 4 arrays x D x 4 bytes per token = 16*D bytes
# per token, and 16*W*D bytes for the whole window, NO MATTER HOW LONG THE CONVERSATION GETS.
# (The earlier figure of 6,144 bytes/token at d=512 counted K,V,O only; decoding needs Q and Vo too.)
# ---------------------------------------------------------------------------------------------
class WindowCache:
    """The HOT tier: exact projections for the last W tokens, preallocated, fixed size."""

    def __init__(self, net, fold=True, fused=True, qkv_batched=False):
        attn = getattr(net, "attn", None)
        if attn is None:
            raise ValueError("this model has no attention block, so there is nothing for a cache to "
                             "hook into. The recent clean models (cl_*) are no-attention by design.")
        if not isinstance(net.emb.data, np.ndarray):
            raise ValueError("this ship's weights are on the DEVICE, and this cache is HOST-ONLY: it "
                             "keeps numpy buffers and applies the model's numpy causal mask, so mixing "
                             "them raises TypeError in the first step (measured: "
                             "lens/runs/GPU-ATTN-CACHE.txt). Load the ship with device='cpu' to use "
                             "the cache, or decode without it.")
        self.D, self.W = int(net.D), int(net.W)
        self.emb = net.emb.data
        self.Wq, self.Wk = attn.Wq.data, attn.Wk.data
        self.Wv, self.Wo = attn.Wv.data, attn.Wo.data
        self.mask = attn.mask_np                      # the model's own causal mask, reused unchanged
        self.cW, self.cb = net.concept.W.data, net.concept.b.data
        self.aW, self.ab = net.answer.W.data, net.answer.b.data
        f32 = np.float32
        self.Q = np.zeros((self.W, self.D), dtype=f32)
        self.K = np.zeros((self.W, self.D), dtype=f32)
        self.V = np.zeros((self.W, self.D), dtype=f32)
        self.Vo = np.zeros((self.W, self.D), dtype=f32)
        self.ids = np.zeros(self.W, dtype=np.int64)
        self.fold = bool(fold)
        # THE FUSED STEP (PLAN item (f), 2026-10-20). The reference read below allocates a fresh array
        # per numpy call - about 10 of them per decoded token - and the step measured 0.264 ms/step at
        # d=256 for ~1.5M multiply-adds (lens/runs/KV-DECODE.txt), i.e. the arithmetic is NOT the cost.
        # `fused` writes the SAME values in the SAME order into buffers allocated once. `fused=False`
        # is the ORIGINAL allocating code, kept as the reference arm of the identity test - a test whose
        # control is the same code it tests proves nothing, so the old path is deliberately left alone.
        self.fused = bool(fused)
        # THE BATCHED INPUT PROJECTION (PLAN item (f3), 2026-10-21). [Wq|Wk|Wv] concatenated into ONE
        # (D, 3D) matrix so a new token costs ONE matvec instead of three. MEASURED not bit-identical to
        # the three separate matvecs (lens/runs/DECODE-STEP-COST.txt, P46q: bit_identical False in all
        # four passes) because BLAS blocks a wider gemv differently, so it is a SWITCH and it is OFF by
        # default. It is not applied anywhere until a token-identity test says the tokens do not move:
        # lens/scripts/qkv_batch_test.py, numbers in lens/runs/QKV-BATCH.txt, predictions written first
        # in lens/runs/QKV-BATCH-PREDICTIONS.txt (P50a-P50i). Wcat is exactly Wq+Wk+Wv side by side, so
        # the bytes read are unchanged.
        self.qkv_batched = bool(qkv_batched)
        self.Wcat = (np.concatenate([self.Wq, self.Wk, self.Wv], axis=1)
                     if self.qkv_batched else None)
        self._s = self._mx = self._out = self._tmp = None
        self._pre = self._t = self._h = None
        self.ready = False
        self.steps = 0

    # ------------------------------------------------------------------ size
    def bytes_per_token(self):
        return 4 * self.D * 4                          # Q, K, V, Vo in float32

    def total_bytes(self):
        return self.bytes_per_token() * self.W         # FIXED: the window, never the whole history

    def summary(self):
        return (f"{self.bytes_per_token()} bytes/token ({self.bytes_per_token()/1024:.1f} KB), "
                f"{self.total_bytes()} bytes total ({self.total_bytes()/1024:.0f} KB) for a "
                f"{self.W}-token window at d={self.D}, fixed size, fold={self.fold}")

    # ------------------------------------------------------------------ fill
    def seed(self, ids):
        """Fill the window from a full window of token ids, in ONE batched pass (like the model does)."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != self.W:
            raise ValueError(f"the cache is seeded with exactly {self.W} tokens, got {ids.shape[0]}")
        self.ids = ids.copy()
        x = self.emb[ids]                              # (W, D) - the model's own embedding lookup
        if self.qkv_batched:
            qkv = x @ self.Wcat                        # (W, 3D) - ONE matvec for all three (f3)
            d = self.D
            self.Q = qkv[:, :d].copy()
            self.K = qkv[:, d:2 * d].copy()
            self.V = qkv[:, 2 * d:].copy()
        else:
            self.Q = x @ self.Wq
            self.K = x @ self.Wk
            self.V = x @ self.Wv
        self.Vo = self.V @ self.Wo
        self.ready = True
        return self

    def push(self, token):
        """Slide the window by one token: drop the oldest, project the new one."""
        if not self.ready:
            raise RuntimeError("seed() the cache before pushing tokens into it")
        self.ids[:-1] = self.ids[1:]
        self.ids[-1] = int(token)
        for buf in (self.Q, self.K, self.V, self.Vo):
            buf[:-1] = buf[1:]
        x = self.emb[int(token)]                       # (D,)
        if self.qkv_batched:
            qkv = x @ self.Wcat                        # (3D,) - ONE matvec (f3)
            d = self.D
            self.Q[-1] = qkv[:d]
            self.K[-1] = qkv[d:2 * d]
            self.V[-1] = qkv[2 * d:]
        else:
            self.Q[-1] = x @ self.Wq
            self.K[-1] = x @ self.Wk
            self.V[-1] = x @ self.Wv
        self.Vo[-1] = self.V[-1] @ self.Wo
        self.steps += 1
        return self

    # ------------------------------------------------------------------ read
    def _scratch(self):
        """The fused step's scratch: allocated ONCE, in the CACHE's own dtype.

        Not a fixed float32 on purpose. seed() REPLACES Q/K/V/Vo with the result of a matmul, so their
        dtype is the ship's weight dtype; a float32 scratch against float64 weights would round
        differently from the reference and the identity test would (correctly) fail. The dtype is read
        off the cache itself, never assumed.
        """
        dt = np.dtype(self.Q.dtype)
        if self._s is None or self._s.dtype != dt:
            k = int(self.cW.shape[1])
            self._s = np.zeros((self.W, self.W), dtype=dt)
            self._mx = np.zeros((self.W, 1), dtype=dt)
            self._out = np.zeros((self.W, self.D), dtype=dt)
            self._tmp = np.zeros((self.W, self.D), dtype=dt)
            self._pre = np.zeros((1, k), dtype=dt)
            self._t = np.zeros((1, k), dtype=dt)
            self._h = np.zeros((1, k), dtype=dt)
        return self._s, self._mx, self._out, self._tmp, self._pre, self._t, self._h

    def outputs(self):
        """The contextualised output of every position in the window, (W, D) - what the concept layer reads.

        THE REFERENCE ARM: the original allocating implementation, untouched, so the fused version can be
        compared against real code rather than against itself.
        """
        s = self.Q @ self.K.T                          # (W, W) scores, already scaled in the init
        s = s + self.mask                              # the model's own causal mask
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(axis=1, keepdims=True)
        if self.fold:
            return p @ self.Vo
        return (p @ self.V) @ self.Wo

    def outputs_into(self):
        """THE FUSED READ: identical arithmetic to outputs(), written in place, no temporaries.

        Returns the cache-owned buffer, which the next call overwrites. Every op is the same op on the
        same operands in the same order as outputs(), so the values are bit-identical (measured, not
        argued: lens/scripts/decode_step_cost.py, P46d).
        """
        s, mx, out, tmp, _pre, _t, _h = self._scratch()
        np.matmul(self.Q, self.K.T, out=s)             # (W, W) scores
        np.add(s, self.mask, out=s)                    # the model's own causal mask
        np.max(s, axis=1, keepdims=True, out=mx)
        np.subtract(s, mx, out=s)
        np.exp(s, out=s)
        np.sum(s, axis=1, keepdims=True, out=mx)
        np.divide(s, mx, out=s)
        if self.fold:
            np.matmul(s, self.Vo, out=out)
            return out
        np.matmul(s, self.V, out=tmp)                  # (p @ V) @ Wo, the reference association
        np.matmul(tmp, self.Wo, out=out)
        return out


def cached_logits_h(net, cache):
    """cached_logits, AND the concept vector it computed on the way.

    Split out 2026-10-20 (PLAN item (c2)) so the warm tier can store the window's concepts during
    generation without paying for a SECOND forward pass. cached_logits below is now this function's
    first element, so there is still exactly one implementation of the cached step.

    Two arms since PLAN item (f): the fused one (cache.fused, the default) writes in place; the other is
    the original allocating code, kept as the reference the identity test measures against.
    """
    if getattr(cache, "fused", False):
        _s, _mx, _out, _tmp, pre, t, h = cache._scratch()
        x = cache.outputs_into().reshape(1, -1)
        np.matmul(x, cache.cW, out=pre)
        np.add(pre, cache.cb, out=pre)
        np.negative(pre, out=t)                        # -pre
        np.clip(t, -60.0, 60.0, out=t)                 # exactly the engine's clip
        np.exp(t, out=t)
        np.add(t, 1.0, out=t)
        np.divide(1.0, t, out=h)
        z = h @ cache.aW + cache.ab
        return z[0], h[0].copy()
    x = cache.outputs().reshape(1, -1)
    pre = x @ cache.cW + cache.cb
    h = 1.0 / (1.0 + np.exp(np.clip(-pre, -60.0, 60.0)))
    return (h @ cache.aW + cache.ab)[0], h[0]


def cached_logits(net, cache):
    """Next-token logits from the cache alone, mirroring lens/model.py op for op.

    sigmoid is written exactly as the engine writes it (clip the exponent to +-60, then 1/(1+exp)),
    so the only difference from the full path is float32 rounding, never a different formula.
    """
    return cached_logits_h(net, cache)[0]


if __name__ == "__main__":
    main()
