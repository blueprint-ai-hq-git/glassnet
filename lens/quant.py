"""lens/quant.py - INT8 WEIGHTS FOR THE DECODE PATH (PLAN item (f2)).

WHY THIS EXISTS: the cached decode step was MEASURED to be bandwidth-bound on the weight matrices -
4.14 MB of weights per decoded token at d=256 and 9.16 MB at d=512, moved at 10.3-12.2 GB/s
(lens/runs/DECODE-STEP-COST.txt, PLAN item (f)). So the lever is FEWER BYTES PER TOKEN, and int8 for
answer.W (41.8% of the traffic at d=256), concept.W (28.5%) and the four projections (25.3%) is a 4x cut.

WHAT IT MUST PROVE before it is applied anywhere (the same bar the fused step passed):
  * EXACT array identity, OR a token-identity test on both ships, and
  * a measured bytes/token and ms/step before and after.

WHAT A MICRO-PROBE ALREADY SHOWED (lens/runs/_tmp/int8_probe.py, 256x256 GEMV, numpy 2.4.6, BEFORE the
test was written): numpy has NO fast int8 path. float32 matmul 6.6 us; int8 -> float32 then matmul
38.45 us; int8 matmul with dtype=int32 84.42 us; dequantise into a buffer 32.51 us. Every int8 form is
SLOWER per operation. That is why the arms here are not "int8 is faster" but a question about DRAM
traffic: only `chunk`, which dequantises L2-sized blocks into ONE reused buffer so the 4-byte-wide
dequantised copy never reaches DRAM, has a chance of winning. The arms are kept side by side so the
comparison is measured, not argued.

THE ARMS (all of them are in this file, none is deleted):
  ref      float32, the shipped fused step (lens/kvcache.py WindowCache, untouched)
  dequant  dequantise the WHOLE matrix per step into a preallocated buffer, then one matmul
  chunk    dequantise in ~256 KB blocks into ONE reused buffer, one matmul per block
  int8acc  quantise the ACTIVATION too and accumulate in int32 (np.matmul(x_i8, W_i8, dtype=int32))
  broken   THE CONTROL: the same code path as `dequant` with every scale forced to 1.0, so any weight
           above 127 saturates. If this does not change the tokens, the identity test is decoration.

NOT CLAIMED: nothing here is an accuracy claim, and nothing here is wired into the default path.
lens/decode.py keeps float32 unless a caller asks for a quantised cache by name.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lens.kvcache import WindowCache                      # noqa: E402


class Int8Weight:
    """One weight matrix quantised to int8, PER-OUTPUT-CHANNEL symmetric.

        scale_j = max_i |W[i, j]| / 127      q = clip(round(W / scale), -127, 127)

    Per-output-channel (not per-tensor) on purpose: a single scale for a whole matrix lets one large
    column decide the resolution of every other column, and `broken=True` is exactly that failure taken
    to its limit (scale 1.0), which is what the control arm measures.
    """

    def __init__(self, W, broken=False):
        W = np.asarray(W, dtype=np.float32)
        self.f32 = W
        self.shape = tuple(W.shape)
        self.broken = bool(broken)
        m = np.abs(W).max(axis=0, keepdims=True)
        scale = (m / 127.0).astype(np.float32)
        scale = np.where(scale == 0.0, np.float32(1.0), scale).astype(np.float32)
        if broken:
            scale = np.ones_like(scale)               # THE CONTROL: everything above 127 saturates
        self.scale = scale
        self.q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)

    # ------------------------------------------------------------------ size
    def bytes(self):
        """The bytes this matrix costs when read once (the int8 payload only)."""
        return int(self.q.size)

    def ref_bytes(self):
        return int(self.q.size) * 4

    # ------------------------------------------------------------------ error
    def max_abs_error(self):
        """max |dequant - W|, and that value as a fraction of the matrix's largest weight."""
        d = np.abs(np.multiply(self.q, self.scale, dtype=np.float32) - self.f32)
        mx = float(np.abs(self.f32).max())
        return float(d.max()), (float(d.max()) / mx if mx else 0.0)

    # ------------------------------------------------------------------ dequantise
    def into(self, dst, rsl=slice(None), csl=slice(None)):
        """Write the dequantised block into dst (float32), reusing dst's memory. No allocation."""
        np.multiply(self.q[rsl, csl], self.scale[:, csl], out=dst, dtype=np.float32)


class F16Weight:
    """The same six matrices as float16 - the SECOND byte width (PLAN item (f2), addendum).

    WHY IT IS HERE: int8 QUARTERS the bytes but pays a MULTIPLY to widen, and it was measured SLOWER on
    every arm (lens/runs/INT8-DECODE.txt). float16 is a 2x cut whose widening is a CAST, and its relative precision (2^-11 = 0.049%) is 8x better than int8's per-channel 0.39%. If the
    step is DRAM-bound and the cast is cheap, this arm can win where int8 lost; if it cannot either, the
    byte lever is closed for pure numpy at every width.
    """

    def __init__(self, W):
        self.f32 = np.asarray(W, dtype=np.float32)
        self.shape = tuple(self.f32.shape)
        self.q = self.f32.astype(np.float16)
        self.scale = None

    def bytes(self):
        return int(self.q.size) * 2

    def ref_bytes(self):
        return int(self.q.size) * 4

    def max_abs_error(self):
        d = np.abs(self.q.astype(np.float32) - self.f32)
        mx = float(np.abs(self.f32).max())
        return float(d.max()), (float(d.max()) / mx if mx else 0.0)

    def into(self, dst, rsl=slice(None), csl=slice(None)):
        np.copyto(dst, self.q[rsl, csl], casting="unsafe")


class QuantWindowCache(WindowCache):
    """The cached decode step with int8 weights, one arm at a time.

    Everything that is NOT a weight read is the parent's code: the same causal mask, the same sigmoid
    written the same way, the same window bookkeeping. Only where a weight matrix is touched does this
    class differ, so an arm's measured difference is attributable to the weights.
    """

    ARMS = ("dequant", "chunk", "int8acc", "broken", "f16", "f16chunk")

    def __init__(self, net, fold=True, arm="chunk", chunk_bytes=262144):
        super().__init__(net, fold=fold, fused=True)
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm {arm!r}; the arms are {self.ARMS}")
        self.arm = arm
        attn = net.attn
        broken = (arm == "broken")
        # THE BYTE WIDTH IS PART OF THE ARM: int8 (per-output-channel, with a scale) or float16 (a cast).
        cls = F16Weight if arm.startswith("f16") else Int8Weight
        self.wcls = cls
        mk = (lambda W: cls(W, broken=broken)) if cls is Int8Weight else (lambda W: cls(W))
        self.qWq = mk(attn.Wq.data)
        self.qWk = mk(attn.Wk.data)
        self.qWv = mk(attn.Wv.data)
        self.qWo = mk(attn.Wo.data)
        self.qcW = mk(net.concept.W.data)
        self.qaW = mk(net.answer.W.data)
        self.chunk_bytes = int(chunk_bytes)
        self._bufs = {}
        self._tmp = None
        self._pre = np.zeros((1, int(net.concept.W.shape[1])), dtype=np.float32)
        self._t = np.zeros_like(self._pre)
        self._h = np.zeros_like(self._pre)
        self.n_matmuls = 0

    # ------------------------------------------------------------------ size
    def weight_bytes_per_token(self):
        """The weight bytes a single decoded token has to read, for this arm."""
        per = (self.qWq.bytes() + self.qWk.bytes() + self.qWv.bytes() + self.qWo.bytes()
               + self.qcW.bytes() + self.qaW.bytes())
        return int(per)

    def ref_weight_bytes_per_token(self):
        per = (self.qWq.ref_bytes() + self.qWk.ref_bytes() + self.qWv.ref_bytes()
               + self.qWo.ref_bytes() + self.qcW.ref_bytes() + self.qaW.ref_bytes())
        return int(per)

    def quant_error(self):
        """max |dequant - W| and its fraction of the largest weight, for all six matrices."""
        out = {}
        for name in ("Wq", "Wk", "Wv", "Wo", "cW", "aW"):
            qw = getattr(self, "q" + name)
            e, rel = qw.max_abs_error()
            out[name] = {"shape": list(qw.shape), "int8_bytes": qw.bytes(),
                         "float32_bytes": qw.ref_bytes(),
                         "max_abs_error": e, "rel_to_max_weight": rel}
        return out

    # ------------------------------------------------------------------ buffers
    def _buf(self, key, shape):
        b = self._bufs.get(key)
        if b is None or b.shape != tuple(shape):
            b = np.zeros(tuple(shape), dtype=np.float32)
            self._bufs[key] = b
        return b

    # ------------------------------------------------------------------ the matmuls
    def _mat(self, x, qw, key):
        """x @ W for one of the six quantised matrices, on this arm."""
        x = np.asarray(x, dtype=np.float32)
        if self.arm == "int8acc":
            mx = float(np.abs(x).max())
            sx = np.float32(mx / 127.0 if mx > 0 else 1.0)
            xq = np.clip(np.round(x / sx), -127, 127).astype(np.int8)
            acc = np.matmul(xq, qw.q, dtype=np.int32)         # integer accumulate
            self.n_matmuls += 1
            return (acc.astype(np.float32) * (sx * qw.scale)).astype(np.float32)
        if self.arm in ("dequant", "broken", "f16"):
            buf = self._buf(key, qw.shape)
            qw.into(buf)
            self.n_matmuls += 1
            return x @ buf
        return self._mat_chunked(x, qw, key)

    def _mat_chunked(self, x, qw, key):
        """Dequantise in blocks that fit in L2 into ONE reused buffer, one matmul per block.

        The block buffer is reused for every block of every step, so the 4-byte-wide dequantised copy
        lives in cache and never becomes DRAM traffic: the DRAM cost of the weight read is the int8
        payload, which is the whole point of the arm.
        """
        n_in, n_out = qw.shape
        m = x.shape[0]
        out = np.empty((m, n_out), dtype=np.float32)
        if n_out >= n_in:
            c = max(1, min(n_out, self.chunk_bytes // (4 * max(1, n_in))))
            buf = self._buf(key, (n_in, c))
            for s in range(0, n_out, c):
                e = min(n_out, s + c)
                b = buf[:, :e - s]
                qw.into(b, slice(None), slice(s, e))
                np.matmul(x, b, out=out[:, s:e])
                self.n_matmuls += 1
        else:
            c = max(1, min(n_in, self.chunk_bytes // (4 * max(1, n_out))))
            buf = self._buf(key, (c, n_out))
            if self._tmp is None or self._tmp.shape != (m, n_out):
                self._tmp = np.zeros((m, n_out), dtype=np.float32)
            first = True
            for s in range(0, n_in, c):
                e = min(n_in, s + c)
                b = buf[:e - s, :]
                qw.into(b, slice(s, e), slice(None))
                if first:
                    np.matmul(x[:, s:e], b, out=out)
                    first = False
                else:
                    np.matmul(x[:, s:e], b, out=self._tmp)
                    np.add(out, self._tmp, out=out)
                self.n_matmuls += 1
        return out

    # ------------------------------------------------------------------ the step
    def seed(self, ids):
        """Fill the window with the QUANTISED projections (the parent's own bookkeeping is kept)."""
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.shape[0] != self.W:
            raise ValueError(f"the cache is seeded with exactly {self.W} tokens, got {ids.shape[0]}")
        self.ids = ids.copy()
        x = self.emb[ids]                                  # (W, D) - the model's own embedding lookup
        self.Q = self._mat(x, self.qWq, "Wq")
        self.K = self._mat(x, self.qWk, "Wk")
        self.V = self._mat(x, self.qWv, "Wv")
        self.Vo = self._mat(self.V, self.qWo, "Wo")
        self.ready = True
        return self

    def push(self, token):
        """Slide the window by one token, with the quantised projections."""
        if not self.ready:
            raise RuntimeError("seed() the cache before pushing tokens into it")
        self.ids[:-1] = self.ids[1:]
        self.ids[-1] = int(token)
        for buf in (self.Q, self.K, self.V, self.Vo):
            buf[:-1] = buf[1:]
        x = self.emb[int(token)]
        self.Q[-1] = self._mat(x[None, :], self.qWq, "Wq")[0]
        self.K[-1] = self._mat(x[None, :], self.qWk, "Wk")[0]
        self.V[-1] = self._mat(x[None, :], self.qWv, "Wv")[0]
        self.Vo[-1] = self._mat(self.V[-1][None, :], self.qWo, "Wo")[0]
        self.steps += 1
        return self

    def logits_h(self):
        """The concept vector and the logits of the window, with the QUANTISED tail.

        The attention half is the parent's fused read (outputs_into), so the only difference from the
        shipped cached step is the two quantised weight matrices at the end and the projections above.
        """
        x = self.outputs_into().reshape(1, -1)
        pre = self._mat(x, self.qcW, "cW") + self.cb
        np.negative(pre, out=self._t)
        np.clip(self._t, -60.0, 60.0, out=self._t)
        np.exp(self._t, out=self._t)
        np.add(self._t, 1.0, out=self._t)
        np.divide(1.0, self._t, out=self._h)
        z = self._mat(self._h, self.qaW, "aW") + self.ab
        return z[0], self._h[0].copy()

    def logits(self):
        return self.logits_h()[0]

    def summary(self):
        base = super().summary()
        return (f"{base} | arm={self.arm}, weights {self.weight_bytes_per_token()} B/token "
                f"({self.ref_weight_bytes_per_token() / max(1, self.weight_bytes_per_token()):.2f}x fewer "
                f"than float32's {self.ref_weight_bytes_per_token()} B), chunk {self.chunk_bytes} B, "
                f"matmuls/step {self.n_matmuls}")
