#!/usr/bin/env python3
"""lens.decode - the ONE decoder, shared by the server and every stop measurement.

WHY A RULE IS NEEDED AT ALL (measured, not assumed):
At 71 true turn boundaries in held-out data the model assigns P('human') = 0.0000 and never ranks it
top-1 (0/71), with or without --train-turn-end. The concept layer is a bag of words over a flattened
window, so "this is the last token of my answer" is not something it can represent. A learned stop is
therefore out of reach at the current text quality -- and saying it works would be a lie.

So this decoder stops on three explicit rules, all reported with the reason they fired:
  R1 turn marker      - the model itself emits the next speaker's marker (the learned path; it never
                        fires yet, which is why it is listed first rather than relied on)
  R2 sentence end     - after at least MIN_TOKENS, an ending (. ! ?) followed by nothing meaningful
  R3 repetition       - the recent output is looping (unique-token share in the last window below
                        UNIQUE_MIN, or an n-gram repeated REPEAT_N times). A model that repeats itself
                        never stops in any useful sense, so this is a real stop, not a cosmetic one.
The reason is always returned, so a "stop" can be audited instead of trusted.
"""
import time

import numpy as np

TURN_MARKERS = {"human", "assistant", "<|im_end|>"}   # <|im_end|> = the standard stop token
CHATML_START = "<|im_start|>"


def format_prompt(user_text, vocab):
    """Build the prompt in the format the model was trained on.

    A corpus carrying the standard control tokens gets the standard chat format:
        <|im_start|> user ... <|im_end|> <|im_start|> assistant
    Older corpora keep the two-token marker format, so every earlier ship still works.
    """
    if CHATML_START in vocab:
        return f"{CHATML_START} user {user_text} <|im_end|> {CHATML_START} assistant"
    return f"human : {user_text} assistant :"
ENDERS = {".", "!", "?"}
MIN_TOKENS = 12
UNIQUE_MIN = 0.45
UNIQUE_WINDOW = 20
REPEAT_N = 3
REPEAT_NGRAM = 4


def _softmax_topk(logits, temp, top_k):
    z = np.asarray(logits, dtype=np.float64) / max(temp, 1e-6)
    if top_k and top_k < len(z):
        cut = np.partition(z, -top_k)[-top_k]
        z = np.where(z >= cut, z, -np.inf)
    m = np.nanmax(z[np.isfinite(z)])
    p = np.exp(z - m)
    p = np.where(np.isfinite(p), p, 0.0)
    return p / p.sum()


def _looping(out):
    if len(out) < UNIQUE_WINDOW:
        return False
    recent = out[-UNIQUE_WINDOW:]
    if len(set(recent)) / len(recent) < UNIQUE_MIN:
        return True
    if len(out) >= REPEAT_NGRAM * REPEAT_N:
        grams = [tuple(out[i:i + REPEAT_NGRAM]) for i in range(len(out) - REPEAT_NGRAM + 1)]
        from collections import Counter
        if Counter(grams).most_common(1)[0][1] >= REPEAT_N:
            return True
    return False


def make_cache(net, ctx, fold=True):
    """The incremental decoding path, or None with the reason it is not available.

    A cache can only be used where there is something to cache: an ATTENTION block. The recent clean
    models are no-attention by design, so they take the full path - and the caller is told so by name
    rather than silently, because a decoder that quietly changes which path it takes is a trap.
    """
    if getattr(net, "attn", None) is None:
        return None, "full recompute (this ship has NO attention block, so there is nothing to cache)"
    if getattr(net, "numeric", False):
        return None, "full recompute (the numeric path has no cached decoder yet)"
    # THE CACHE IS HOST-ONLY, AND SAYS SO (real bug found 2026-10-20 by running it, not by reading it:
    # lens/runs/GPU-ATTN-CACHE.txt). The cache keeps numpy buffers and applies the model's numpy causal
    # mask; a ship loaded with device="gpu" keeps cupy arrays, and mixing them raises
    # "TypeError: `out` must be a tuple or `cupy.ndarray`" inside the first step. That made
    # `lens/serve.py <attention ship> --device gpu` fail at the first token - the GPU server could not
    # decode an attention ship at all. The fix is to take the FULL path and SAY WHY, not to silently
    # move the cache to the device: that would change the numbers, and it needs its own identity test.
    if not isinstance(net.emb.data, np.ndarray):
        return None, ("full recompute (the KV cache is HOST-ONLY: this ship's weights live on the "
                      "device, and the cache's numpy buffers and causal mask cannot mix with them)")
    from lens.kvcache import WindowCache
    return WindowCache(net, fold=fold).seed(ctx), "cached (KV, window cache)"


def _grafted(net):
    """True when this network carries the sentstate graft (lens/sentstate_wire.py).

    Read off the ATTRIBUTE rather than importing the module, so an un-edited net cannot pay an import and a
    base net's path stays exactly what it was. P115d's control is that base path returning the same numbers.
    """
    return getattr(net, "sread", None) is not None and getattr(net, "pconcept", None) is not None


def step_logits(net, ctx, cache):
    """The next-token logits for one step, on whichever path the caller chose."""
    if cache is None:
        return np.asarray(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0],
                          dtype=np.float64)
    if _grafted(net):
        # THE CACHE CANNOT SEE THE STATE: cached_logits reads the answer matrix off the cache and never calls
        # the model, so on a grafted net it would silently drop the state term (measured in P115b). The grafted
        # step recomputes that ONE term from cache.outputs() -- the same per-token representation the graft
        # taps -- and is checked against the full path in P115c.
        from lens.sentstate_wire import cached_logits_grafted
        return np.asarray(cached_logits_grafted(net, cache)[0], dtype=np.float64)
    from lens.kvcache import cached_logits
    return np.asarray(cached_logits(net, cache), dtype=np.float64)


def step_logits_and_concepts(net, ctx, cache):
    """Next-token logits AND the concept vector for one step, from ONE pass.

    The warm tier needs the concept vector of the window it just computed. Getting it by calling
    forward_arrays again would double the step cost, so the two paths are read out of a single pass.
    """
    if cache is None:
        out = net.forward_arrays(np.asarray([ctx], dtype=np.int64))
        return (np.asarray(out["logits"][0], dtype=np.float64),
                np.asarray(out["concepts"][0], dtype=np.float32))
    if _grafted(net):
        from lens.sentstate_wire import cached_logits_grafted
        z, h = cached_logits_grafted(net, cache)
        return np.asarray(z, dtype=np.float64), np.asarray(h, dtype=np.float32)
    from lens.kvcache import cached_logits_h
    z, h = cached_logits_h(net, cache)
    return np.asarray(z, dtype=np.float64), np.asarray(h, dtype=np.float32)


# ------------------------------------------------------------------------------------------------
# THE WARM TIER IN THE GENERATION LOOP (PLAN item (c2), 2026-10-20)
#
# WHAT IT IS FOR, stated before the numbers (the numbers are in lens/runs/WARM-DECODE.txt, the
# predictions were written first in lens/runs/WARM-DECODE-PREDICTIONS.txt):
#   * the HOT tier (lens/kvcache.py) keeps the last WINDOW: 16*D bytes per token, fixed size;
#   * the WARM tier (lens/warmmemory.py) keeps the NAMED CONCEPTS OF EVERY WINDOW of the context:
#     37 bytes per 32-token window (36 concepts + the exact top-1 index), so the whole conversation
#     stays answerable after it has scrolled out of the window, and what is kept is printable as names.
#
# WHAT IT IS NOT, and this is the honest part:
#   * it keeps no tokens, so nothing can be quoted back from it verbatim (lens/warmmemory.py,
#     describe_limits());
#   * MEASURED, it does NOT accelerate sliding generation: the window slides by ONE token per step
#     while the store is written at stride W, so a stored window is never revisited. The read path
#     fires at most once per generation (step 0, the prompt's last window). A decoder that quietly
#     took a different path and claimed a speedup would be a lie, so the fire count is returned in
#     the result and printed by the server.
# ------------------------------------------------------------------------------------------------
class StepSource:
    """One step's logits: from the STORE when the window is stored, else from the network.

    Shared by generate() here and by lens/serve.py, so the server cannot drift from the measured
    decoder - the same rule the KV-cache wiring followed. It also owns the counters (hits, misses,
    the write-through), so "the memory path fired N times" is measured rather than claimed.

    READ vs WRITE, and why the default is write-only (measured, not a guess):
      * WRITE (always, when a warm store is attached): every window on a store boundary is kept as
        concepts. 37 bytes per 32 tokens, the h comes from the pass already being paid for, and NO
        token can change.
      * READ (read_from_memory=True): the step's logits come from the stored bytes instead of a
        forward pass. MEASURED: the u8 store reproduces the network's top-1 token on 99.25% of
        2,000 held-out windows, so a generation that reads from it CAN differ from the shipped path
        (lens/runs/WARM-DECODE.txt, P38b). A cache that changes the answer is a bug, so the read
        path is OFF by default and has to be asked for - and the server says which mode it used.
    """

    def __init__(self, net, cache, warm=None, base_end=None, read_from_memory=False):
        self.net, self.cache, self.warm = net, cache, warm
        self.base_end = base_end
        self.read_from_memory = bool(read_from_memory)

    def step(self, ctx, end):
        if self.warm is not None and self.read_from_memory:
            row = self.warm.row_for(end)
            if row is not None:
                t0 = time.time()
                z = np.asarray(self.warm.logits(row), dtype=np.float64)
                self.warm.memory_ms += (time.time() - t0) * 1000.0
                self.warm.hits += 1
                return z, "memory"
        if self.warm is not None:
            t0 = time.time()
            z, h = step_logits_and_concepts(self.net, ctx, self.cache)
            self.warm.network_ms += (time.time() - t0) * 1000.0
            self.warm.misses += 1
            # write-through: keep this window's concepts if it sits on a store boundary. The h came
            # from the pass that was already being paid for, so this costs 37 bytes and no extra compute.
            if (self.base_end is not None and (end - self.base_end) % self.warm.stride == 0
                    and end > self.warm.max_end):
                self.warm.put(end, h)
            return z, "network"
        return step_logits(self.net, ctx, self.cache), "network"


class WarmDecode:
    """The warm tier in the decode loop: concepts per stored window, plus the end position of each.

    The store itself is lens/warmmemory.py, unchanged and already proven (lens/runs/WARM-MEMORY.txt);
    this class is the wiring: which window is stored, how it is recalled, and what it costs.
    """

    def __init__(self, net, stride=None, dtype=None, store_top1=True):
        from lens.warmmemory import WarmConceptMemory
        if not isinstance(net.emb.data, np.ndarray):
            raise ValueError("the warm tier is HOST-ONLY (it keeps a uint8 concept store and reads the "
                             "ship's weights with numpy), and this ship's weights are on the DEVICE, "
                             "so ingesting a window raises. Load the ship with device='cpu' to use it. "
                             "Measured on the KV cache's identical case: lens/runs/GPU-ATTN-CACHE.txt")
        self.net = net
        self.W, self.K = int(net.W), int(net.K)
        self.stride = int(stride or net.W)
        self.store = WarmConceptMemory(n_concepts=self.K, window=self.W, stride=self.stride,
                                       dtype=(dtype if dtype is not None else np.uint8),
                                       store_top1=store_top1)
        self.rows = {}            # window END position (absolute token index) -> row in the store
        self.max_end = -1         # highest end position stored, so a repeated prompt is not paid twice
        self.written = 0
        self.hits = 0
        self.misses = 0
        self.memory_ms = 0.0
        self.network_ms = 0.0
        self._aW = self._ab = None

    # ------------------------------------------------------------------ weights (read once)
    def answer_weights(self):
        if self._aW is None:
            from lens import backend
            self._aW = np.asarray(backend.to_cpu(self.net.answer.W.data), dtype=np.float32)
            self._ab = np.asarray(backend.to_cpu(self.net.answer.b.data), dtype=np.float32)
        return self._aW, self._ab

    # ------------------------------------------------------------------ writing
    def concepts(self, windows):
        """The network's own concept layer on a batch of windows (the thing the store keeps)."""
        X = np.asarray(windows, dtype=np.int64)
        if X.ndim == 1:
            X = X[None, :]
        return np.asarray(self.net.forward_arrays(X)["concepts"], dtype=np.float32)

    def put(self, end, h):
        row = self.store.add(h)
        self.rows[int(end)] = row
        self.written += 1
        if int(end) > self.max_end:
            self.max_end = int(end)
        return row

    def ingest(self, toks, first=0, last=None):
        """Store the concepts of every window of the context, at this stride, in ONE batched pass.

        Only windows whose end position is NEW (beyond max_end) are written, so a server that re-sends
        the same conversation does not pay for the same window twice. Returns windows written.
        """
        last = (len(toks) - 1) if last is None else int(last)
        ends, wins = [], []
        end = last
        while end - self.W + 1 >= first:
            if end > self.max_end:
                ends.append(end)
                wins.append([int(t) for t in toks[end - self.W + 1: end + 1]])
            end -= self.stride
        if wins:
            for e, h in zip(ends, self.concepts(wins)):
                self.put(e, h)
        return len(wins)

    # ------------------------------------------------------------------ reading
    def row_for(self, end):
        """The stored row for the window ending at absolute position `end`, or None."""
        return self.rows.get(int(end))

    def logits(self, row):
        """Next-token logits from the STORED BYTES ALONE: no embedding, no attention, no concept layer.

        The answer weights are the model's own, so this is the same formula the model uses - the only
        difference is that h is read back from one byte per concept instead of computed.
        """
        aW, ab = self.answer_weights()
        return self.store.recall(row) @ aW + ab

    def trace(self, row, names=None, thresh=0.5):
        """What window `row` remembers, in words."""
        names = names if names is not None else [c.name for c in self.net.concepts]
        return self.store.trace(row, names, thresh=thresh)

    # ------------------------------------------------------------------ size
    def report(self):
        rep = self.store.report()
        rep.update({"hits": self.hits, "misses": self.misses, "written": self.written,
                    "index_rows": len(self.rows), "stride": self.stride,
                    "bytes_per_window": self.store.bytes_per_window()})
        return rep


def generate(net, vocab, vmap, prompt, max_new=150, temp=0.8, top_k=40, rep_penalty=1.15, seed=0,
             greedy=False, rule_stop=True, use_cache=True, cache=None, cache_factory=None, fold=True,
             use_warm=False, warm=None, warm_stride=None, warm_dtype=None, warm_read=False):

    # A vocab that does not match the model is a silent trap: the first version of the night talk test
    # handed a 12,002-token model a 12,000-word vocab and numpy died deep inside choice() with
    # "a and p must have same size", which says nothing about the real cause. Refuse it here, by name.
    if len(vocab) != net.V:
        raise ValueError(f"vocab length {len(vocab)} does not match this model's vocabulary {net.V}. "
                         f"Control-token models (V=12002) need the two ChatML tokens appended to the word "
                         f"list; word-only models (V=12000) do not.")
    """Returns {text, tokens, token_ids, stop_reason, stop_at, forced, path}."""
    W = int(getattr(net, "W", 32))
    V = len(vocab)
    import re
    enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", prompt.lower())]
    if len(enc) < W:
        enc = [0] * (W - len(enc)) + enc
    ctx, out, used = enc[-W:], [], {}
    # THE CACHED PATH (2026-10-20, PLAN item (e)). Same window, same ops, one new token's projections
    # per step instead of the whole window's. The tokens it produces are identical to the full path -
    # that is measured, not assumed: lens/scripts/kv_decode_test.py, lens/runs/KV-DECODE.txt.
    if cache is not None:
        path = "cached (KV, supplied cache)"
    elif use_cache and cache_factory is not None:
        cache = cache_factory(net, ctx)
        path = "supplied cache (factory - used by the control test)"
    elif use_cache:
        cache, path = make_cache(net, ctx, fold=fold)
    else:
        path = "full recompute (asked for by use_cache=False)"
    # THE WARM TIER (PLAN item (c2)). Off by default: with use_warm=False every existing caller and
    # every recorded number is unchanged, which matters because the read path serves quantised
    # concepts and can therefore move a token (measured, lens/runs/WARM-DECODE.txt, P38g).
    warm_used = None
    if warm is not None or use_warm:
        warm_used = warm if warm is not None else WarmDecode(net, stride=warm_stride, dtype=warm_dtype)
        wrote = warm_used.ingest(enc, 0, len(enc) - 1)
        mode = "read+write" if warm_read else "write-only (no token can change)"
        path = path + f" + warm memory ({warm_used.store.bytes_per_window()} B/window, stride " \
                      f"{warm_used.stride}, {mode}; {wrote} prompt windows stored)"
    base_end = len(enc) - 1                 # absolute index of the last prompt token
    src = StepSource(net, cache, warm_used, base_end=base_end, read_from_memory=warm_read)
    reason, forced = "budget", False
    for step in range(max_new):
        end = base_end + step
        z, _used_path = src.step(ctx, end)
        for tok, cnt in used.items():
            z[tok] -= np.log(rep_penalty) * cnt
        p = _softmax_topk(z, temp, top_k)
        nxt = int(np.argmax(p)) if greedy else int(np.random.default_rng(seed + 1000 * step).choice(V, p=p))
        used[nxt] = used.get(nxt, 0) + 1
        w = vocab[nxt]
        if w.lower() in TURN_MARKERS:                      # R1
            reason, forced = "turn marker (model emitted it)", False
            break
        out.append(nxt)
        ctx = (ctx + [nxt])[-W:]
        if cache is not None:
            cache.push(nxt)
        if rule_stop:
            if len(out) >= MIN_TOKENS and w in ENDERS:     # R2
                reason, forced = "sentence end", True
                break
            if _looping(out):                              # R3
                reason, forced = "repetition detected", True
                break
    text = " ".join(vocab[t] for t in out).replace(" '", "'")
    warm_info = None
    if warm_used is not None:
        rep = warm_used.report()
        warm_info = {"windows": rep["windows"], "bytes": rep["bytes"],
                     "bytes_per_window": rep["bytes_per_window"],
                     "bytes_per_generated_token": rep["bytes"] / float(max(1, len(out))),
                     "bytes_per_token_asymptotic": rep["bytes_per_token"],
                     "hits": rep["hits"], "misses": rep["misses"], "stride": rep["stride"],
                     "top1_kept": rep["top1_kept"], "read_from_memory": bool(warm_read),
                     "memory_ms_per_hit": (warm_used.memory_ms / warm_used.hits
                                           if warm_used.hits else None),
                     "network_ms_per_step": warm_used.network_ms / float(max(1, warm_used.misses))}
    return {"text": text, "tokens": len(out), "token_ids": list(out), "stop_reason": reason,
            "stop_at": len(out), "forced": forced, "path": path,
            "cache_bytes_per_token": (cache.bytes_per_token() if cache is not None else 0),
            "cache_bytes_total": (cache.total_bytes() if cache is not None else 0),
            "warm": warm_info}