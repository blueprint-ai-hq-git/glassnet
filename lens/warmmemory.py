
"""lens/warmmemory.py - THE WARM TIER: remember MEANINGS, not words.

Maintainer (2026-10-20): "the KV cache and the context window, can we get unlimited context windows ... even 1
gigabyte would be around a billion tokens". This file is the answer for the middle tier of the three he
asked for (FROM-SCRATCH.md GAP 4):

  HOT   exact K/V/O for the last window            lens/kvcache.py     (3*D*4 bytes per token)
  WARM  the NAMED CONCEPTS of every window         THIS FILE           (K bytes per WINDOW)
  COLD  lossless tokens on disk (int16)            not built yet       (2 bytes per token)

WHY THIS IS POSSIBLE FOR US AND NOT FOR THEM. A transformer must keep per-token Keys and Values, so its
memory grows with the conversation and it needs a hard context limit. Our architecture summarises a fixed
window of tokens into a fixed number of NAMED concepts, so the thing worth keeping is a small readable
vector per window, not a vector per token. One byte per concept per window is enough, and it is not opaque:
we can print what the model remembers, because it is names.

WHAT IT STORES, EXACTLY. For every window of W tokens: the concept layer's output h = sigmoid(W·x + b), K
values in [0, 1], each quantised to one byte (0..255 linear over [0, 1]). Nothing else. No tokens, no keys,
no values, no hidden states. The tokens are NOT recoverable from it - that is the honest limitation, and it
is why the cold tier exists.

WHY THE QUANTISATION IS MEASURED AND NOT ASSUMED. The same standard as the KV cache applies: a store that
changes the answer is a bug. `warm_memory_test.py` therefore checks, on held-out windows, that the top-1
token and the top-1 concept NAME computed from the stored bytes match the float32 path. If they do not, the
tier does not ship at this precision.

USAGE
    from lens.warmmemory import WarmConceptMemory, answer_from_memory
    mem = WarmConceptMemory(n_concepts=36, window=32, stride=32)
    mem.add(h)                       # h: (K,) float concepts for one window
    mem.top(i, names)                # what it remembers, in words
    answer_from_memory(mem, i, answer_W, answer_b).argmax()   # answer with NO network call
"""
import numpy as np


class WarmConceptMemory:
    """One byte per named concept per window. Preallocated and grown by doubling, never per-window objects.

    The growth rule is deliberate and follows the lesson from the first KV cache: rebuilding an array per
    item (np.concatenate in a loop) is what made the cache 14x slower than no cache at all. Here the buffer
    is allocated once and doubled when full, so the per-window cost is a row write.
    """

    def __init__(self, n_concepts, window, stride=None, capacity=1024, dtype=np.uint8,
                 store_top1=True):
        """store_top1 (the DEFAULT, and why): a 1-byte store changes WHICH concept is nominally the
        strongest on 6.35% of held-out windows (measured, lens/runs/WARM-MEMORY.txt, P20), because two
        concepts whose float32 values differ by less than 1/255 swap order once rounded. The value the
        user READS is the name, so the name is kept exactly: one extra byte per window holds the argmax
        index of the original float vector. Cost 1 byte per 32 tokens (+0.031 B/token). Turning this off
        gives the pure 1.125 B/token store, which is smaller but reads the wrong name 6.35% of the time.
        """
        self.K = int(n_concepts)
        self.W = int(window)
        # stride is how many tokens pass per stored row: stride==W is non-overlapping (cheapest),
        # stride==W//2 is the streaming encoder's default (overlapping, every token seen twice).
        self.stride = int(stride or window)
        self.dtype = np.dtype(dtype)
        self.scale = 255.0 if self.dtype == np.uint8 else 65535.0
        self.store_top1 = bool(store_top1) and self.K <= 255
        self.Q = np.zeros((max(1, int(capacity)), self.K), dtype=self.dtype)
        self.T1 = np.zeros(max(1, int(capacity)), dtype=np.uint8) if self.store_top1 else None
        self.n = 0

    # ------------------------------------------------------------------ writing
    def _grow(self):
        if self.n == self.Q.shape[0]:
            self.Q = np.concatenate([self.Q, np.zeros_like(self.Q)], axis=0)   # double, rarely
            if self.T1 is not None:
                self.T1 = np.concatenate([self.T1, np.zeros_like(self.T1)], axis=0)

    def add(self, h):
        """Store one window's concepts. h is (K,) float in [0, 1]; returns the row index."""
        h = np.asarray(h, dtype=np.float32).reshape(-1)
        if h.size != self.K:
            raise ValueError(f"expected {self.K} concepts, got {h.size}")
        self._grow()
        self.Q[self.n] = np.clip(np.rint(h * self.scale), 0.0, self.scale).astype(self.dtype)
        if self.T1 is not None:
            self.T1[self.n] = int(np.argmax(h))
        self.n += 1
        return self.n - 1

    def add_many(self, H):
        """Store many windows at once. H is (n, K). Returns the index of the first row written."""
        H = np.asarray(H, dtype=np.float32)
        first = self.n
        for row in H:
            self.add(row)
        return first

    # ------------------------------------------------------------------ reading
    def recall(self, i):
        """The stored concepts of window i, back as floats in [0, 1]. This is all that was kept."""
        return self.Q[i].astype(np.float32) / self.scale

    def top1_index(self, i):
        """The strongest concept of window i: the stored index when kept, else read off the bytes."""
        if self.T1 is not None:
            return int(self.T1[i])
        return int(np.argmax(self.recall(i)))

    def top(self, i, names=None, k=6, thresh=0.5):
        """What window i remembers, in words: [(name, value), ...] best first.

        When the top-1 index is kept, it is forced to the front: the name the store reports is then the
        name the float32 model actually had, not the one the rounding produced.
        """
        h = self.recall(i)
        order = list(np.argsort(-h)[:k])
        t1 = self.top1_index(i)
        if order and order[0] != t1:
            order = [t1] + [j for j in order if j != t1]
        out = []
        for j in order[:k]:
            if h[j] < thresh:
                break
            out.append((names[j] if names is not None else f"c{j}", float(h[j])))
        return out

    def trace(self, i, names, thresh=0.5):
        """A printable one-line memory of window i. Readable by construction, not by convention."""
        t = self.top(i, names, k=self.K, thresh=thresh)
        return ", ".join(n for n, _ in t) if t else "(nothing above threshold)"

    # ------------------------------------------------------------------ size
    def bytes_per_window(self):
        return self.K * self.dtype.itemsize + (1 if self.T1 is not None else 0)

    def bytes_per_token(self):
        return self.bytes_per_window() / float(self.stride)

    def nbytes(self):
        """Bytes actually held (not the allocated capacity)."""
        return self.n * self.bytes_per_window()

    def tokens_for(self, nbytes):
        return int(nbytes / self.bytes_per_token())

    def report(self):
        return {"windows": self.n, "tokens_covered": self.n * self.stride, "concepts": self.K,
                "window": self.W, "stride": self.stride, "bytes": self.nbytes(),
                "bytes_per_window": self.bytes_per_window(), "bytes_per_token": self.bytes_per_token(),
                "allocated_bytes": self.Q.size * self.dtype.itemsize
                + (self.T1.size if self.T1 is not None else 0),
                "top1_kept": self.T1 is not None}


def answer_from_memory(mem, i, answer_W, answer_b):
    """Answer for window i using ONLY what is in the store, plus the model's answer weights.

    There is no network call here on purpose: the function is handed the store and two weight arrays and
    nothing else, so a correct answer cannot be coming from a recompute. Logits for the next token.
    """
    h = mem.recall(i)                       # (K,)
    return h @ np.asarray(answer_W, dtype=np.float32) + np.asarray(answer_b, dtype=np.float32)


def describe_limits():
    """The honest limits, in one place, so no caller can claim more than this file delivers."""
    return ("warm tier: keeps named concepts per window, not tokens. Readable, ~1 byte per concept, but it "
            "CANNOT quote a sentence back verbatim and it does not store exact wording. Exact wording needs "
            "the cold tier (lossless int16 tokens), which is not built yet.")
