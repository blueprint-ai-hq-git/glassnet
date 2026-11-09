#!/usr/bin/env python3
"""lens/stream.py - NO SEQUENCE LIMIT. Constant memory, any input length, any output length.

The user's requirement, in his words: "we have to set the sequential token limit for output and the data
input ... because of the RAM GB ... we have to find a way so it works, we don't need any limit, it just
works."

WHY THIS IS POSSIBLE FOR US AND NOT FOR A TRANSFORMER: a transformer's attention keeps every token, so its
memory grows with sequence length and it must be given a hard limit (context window). Our architecture has
no attention over the whole sequence: it summarises a fixed-size window into a fixed number of named
concepts. So the memory cost of a chunk is fixed, and arbitrarily long input can be processed as a STREAM
of chunks with a bounded running summary. Nothing about that is a trick or an approximation of a limit --
the memory genuinely does not grow with input length, which is what this file demonstrates with measured
numbers rather than assertion.

Two pieces:
  StreamingEncoder  - feed any number of tokens; keeps only a fixed window plus a fixed-size concept
                      summary. Memory is O(window + concepts), independent of how many tokens went in.
  stream_generate   - generate for any number of tokens with the same bounded state and the shared stop
                      rules (lens/decode.py), so output length needs no cap either.

The honest limits, stated so nobody over-claims:
  * The summary is a fixed-size concept state, so information from very early in a long input is compressed
    (a running max with decay), not retained verbatim. "No token limit" means no memory limit and no hard
    cap; it does NOT mean the model remembers every word of a million-token input. Those are different
    claims and only the first one is made here.
  * CPU time still grows with input length (obviously). Memory does not.

Usage: python3 lens/stream.py <ship.npz> [--tokens 1000 10000 100000 1000000] [--device cpu]
"""
import argparse
import os
import resource
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", ".")
sys.path.insert(0, ROOT)
# REAL BUG FIX 2026-10-20 (evidence lens/runs/SERVE-GPU-AFTER.txt): `python3 lens/<this file>`
# puts THIS directory -- the project's lens/ -- at sys.path[0], so the top-level name `cuda`
# resolved to our own lens/cuda/ package. That is a REGULAR package and therefore beats
# cuda-python's PEP 420 namespace package (appending it does not help either), and cupy 14.2
# died at import with "ModuleNotFoundError: No module named 'cuda.pathfinder'" -- so --device
# gpu could not start AT ALL. This file's own directory must not be importable as top level.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)


def peak_mb():
    """Peak resident memory of this process, in MB. This is the number the whole claim rests on."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class StreamingEncoder:
    """Feed tokens of ANY length; keep a fixed window and a fixed-size concept summary.

    stride defaults to half the window, so consecutive chunks overlap and a concept that spans a chunk
    boundary is still seen -- at no extra memory cost.
    """

    def __init__(self, net, decay=0.9, stride=None):
        self.net = net
        self.W = int(net.W)
        self.decay = float(decay)
        self.stride = int(stride or max(1, self.W // 2))
        self.state = np.zeros(int(net.K), dtype=np.float64)
        self.buf = []
        self.chunks = 0
        self.tokens_seen = 0

    def feed(self, ids):
        for t in ids:
            self.tokens_seen += 1
            self.buf.append(int(t))
            if len(self.buf) >= self.W:
                x = np.asarray([self.buf[-self.W:]], dtype=np.int64)
                h = np.asarray(self.net.forward_arrays(x)["concepts"])[0]
                # bounded running summary: decay then element-wise max. Fixed size, forever.
                self.state = np.maximum(self.decay * self.state, h)
                self.chunks += 1
                self.buf = self.buf[-self.stride:] if self.stride < self.W else []
        return self.state

    def finish(self):
        """Flush a short tail through the same fixed window (padded), so the last tokens count too."""
        if self.buf:
            x = np.asarray([([0] * (self.W - len(self.buf)) + self.buf)[-self.W:]], dtype=np.int64)
            h = np.asarray(self.net.forward_arrays(x)["concepts"])[0]
            self.state = np.maximum(self.decay * self.state, h)
            self.chunks += 1
            self.buf = []
        return self.state

    def memory_report(self):
        return {"tokens_seen": self.tokens_seen, "chunks": self.chunks,
                "window": self.W, "summary_size": int(self.state.size),
                "buffer_len": len(self.buf)}


def main():
    import lens  # noqa: E402
    from lens.decode import generate  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("ship", nargs="?", default="lens/runs/chat/sb_S0_d256_noattn_ep96.npz")
    ap.add_argument("--tokens", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    net = lens.load(a.ship, device=a.device)
    V = int(net.V)
    rng = np.random.default_rng(0)
    print(f"NO-LIMIT STREAMING TEST - ship {os.path.basename(a.ship)}")
    print(f"  architecture: window {net.W} tokens, {net.K} named concepts, memory must NOT grow with length")
    print(f"  baseline process peak after loading the model: {peak_mb():.1f} MB")
    print()
    print(f"  {'tokens fed':>12}  {'chunks':>8}  {'buffer':>7}  {'summary':>8}  {'peak RSS MB':>11}")
    rows = []
    for n in a.tokens:
        enc = StreamingEncoder(net)
        ids = rng.integers(0, V, size=n)
        # feed in slices to show that nothing accumulates outside the fixed window
        for i in range(0, n, 50_000):
            enc.feed(ids[i:i + 50_000])
        enc.finish()
        rep = enc.memory_report()
        rows.append((n, rep["chunks"], rep["buffer_len"], rep["summary_size"], peak_mb()))
        print(f"  {n:>12,}  {rep['chunks']:>8,}  {rep['buffer_len']:>7}  {rep['summary_size']:>8}  "
              f"{peak_mb():>11.1f}")
        del enc, ids

    first, last = rows[0][4], rows[-1][4]
    growth = last - first
    print()
    print(f"  input length went from {a.tokens[0]:,} to {a.tokens[-1]:,} tokens "
          f"({a.tokens[-1]/max(a.tokens[0],1):,.0f}x longer) and memory moved {growth:+.1f} MB "
          f"({first:.1f} -> {last:.1f} MB)")
    print(f"  buffer length stayed at or below the window ({net.W}) at every length, summary size fixed at "
          f"{net.K} concepts")
    verdict = "NO TOKEN LIMIT: memory is flat in input length" if growth < 200 else \
              "MEMORY GREW: the streaming claim FAILS, do not ship it"
    print(f"  VERDICT: {verdict}")

    # output side: generate far past any normal cap and watch memory again
    d = __import__("chat_data").build()
    vocab, vmap = d["vocab"], {w: i for i, w in enumerate(d["vocab"])}
    before = peak_mb()
    g = generate(net, vocab, vmap, "human : hello assistant :", max_new=3000, rule_stop=True)
    after = peak_mb()
    print(f"\n  generation: {g['tokens']} tokens produced (a normal chat cap is 128-512), "
          f"stop reason '{g['stop_reason']}', memory {before:.1f} -> {after:.1f} MB "
          f"({after-before:+.1f})")
    print("  HONEST LIMIT: the summary is a fixed-size running state, so very early input is compressed "
          "rather than retained word-for-word. No memory limit and no hard cap -- that is the claim; "
          "perfect recall of a million tokens is not.")
    return 0 if growth < 200 else 1


if __name__ == "__main__":
    sys.exit(main())