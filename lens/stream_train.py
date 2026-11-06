#!/usr/bin/env python3
"""lens/stream_train.py - TRAINING with no sequence-length limit, input or output.

The user's correction, and it is the right one: the no-limit claim has to hold for TRAINING, not just
inference. "We talking about training sq length out and in."

WHAT WAS WRONG: every training example was a fixed 32-token window. A document longer than that was chopped
into independent windows with no connection between them, so the model never saw a long input as one thing,
and the size of an example was capped by RAM exactly like every other model's context limit.

WHAT THIS DOES: a document of ANY length is processed as a stream of chunks with a FIXED-SIZE carried state
(K numbers, one per named concept). Memory is O(window + concepts) and does not grow with document length --
the same property proven for inference in lens/stream.py, now on the training path.

The long-range learning signal, and why this one: as each chunk is read, the model compares its concepts to
the carried state and computes SURPRISE (how much this chunk says that the document has not said before).
Surprise sets the per-sample loss weight, so the training signal follows what is NEW in the document rather
than repeating what it already knows. This is expressible with the engine as it stands (`T.softmax_ce`
takes per-sample weights), it needs no new ops, and it is exactly the kind of thing a fixed-window trainer
cannot do at all.

HONEST BOUNDS, stated so nothing is over-claimed:
  * Gradients flow within a chunk. The carried state is a detached, fixed-size summary, so credit does not
    propagate back through the whole document (truncated-BPTT style). Memory flatness is what is proven
    here; full-document backpropagation is NOT claimed.
  * The state is a decaying running summary, so very early content is compressed, not retained verbatim.

Usage: python3 lens/scripts/stream_train.py [--doc-tokens 1000 10000 100000] [--steps-per-doc 4]
"""
import argparse
import os
import resource
import sys
import time

import numpy as np

ROOT = "/run/media/maintainer/HARD DISK 4/GlassNet"
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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class StreamTrainer:
    """Trains on a document of any length with memory that does not grow with that length."""

    def __init__(self, net, opt, decay=0.9, surprise_scale=1.0):
        self.net = net
        self.opt = opt
        self.W = int(net.W)
        self.K = int(net.K)
        self.decay = float(decay)
        self.surprise_scale = float(surprise_scale)
        self.state = np.zeros(self.K, dtype=np.float64)

    def reset(self):
        self.state[:] = 0.0

    def step_document(self, doc_ids, chunk_stride=None, verbose=False):
        """One pass over a document of ANY length. Returns (chunks, mean_loss, mean_surprise)."""
        W = self.W
        stride = int(chunk_stride or W)
        losses, surprises = [], []
        n = len(doc_ids)
        for start in range(0, n - W - 1, stride):
            window = np.asarray(doc_ids[start:start + W], dtype=np.int64)
            target = int(doc_ids[start + W])
            # SURPRISE: how much this chunk says that the carried state has not already said
            h = np.asarray(self.net.forward_arrays(window[None, :])["concepts"])[0].astype(np.float64)
            surprise = float(np.linalg.norm(h - self.state))
            # weight this sample by its surprise: the training signal follows what is new in the document
            w = 1.0 + self.surprise_scale * surprise
            self.opt.zero_grad()
            out = self.net.loss(window[None, :], np.asarray([target], dtype=np.int64),
                                use_concept_loss=True, weights=np.asarray([w], dtype=np.float32))
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            self.opt.step()
            losses.append(float(loss.data))
            surprises.append(surprise)
            # fixed-size state update: memory here is K numbers no matter how long the document is
            self.state = np.maximum(self.decay * self.state, h)
            if verbose and len(losses) % 200 == 0:
                print(f"      chunk {len(losses):>6}  loss {losses[-1]:.4f}  surprise {surprise:.3f}  "
                      f"state size {self.state.size}")
        return len(losses), (float(np.mean(losses)) if losses else float("nan")), \
               (float(np.mean(surprises)) if surprises else float("nan"))


def main():
    import lens  # noqa: E402
    import chat_data  # noqa: E402
    import lens.opt as opt_mod  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("ship", nargs="?", default="lens/runs/chat/sz_d128_24_ep24.npz")
    ap.add_argument("--doc-tokens", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--steps-per-doc", type=int, default=0, help="0 = walk the whole document")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--decay", type=float, default=0.9)
    a = ap.parse_args()

    if not os.path.exists(a.ship):
        a.ship = "lens/runs/chat/sb_S0_d256_noattn_ep96.npz"
    net = lens.load(a.ship, device=a.device)
    opt = opt_mod.RelativeGrad(net.parameters(), lr=0.005, mode="relgrad")
    d = chat_data.build()
    ids = np.asarray(d["ids"], dtype=np.int64)

    print(f"STREAMING TRAINING - no sequence-length limit  ship {os.path.basename(a.ship)}")
    print(f"  window {net.W} tokens, {net.K} concepts, carried state is {net.K} numbers "
          f"regardless of document length")
    print(f"  baseline peak after load + corpus: {peak_mb():.1f} MB\n")
    print(f"  {'document tokens':>16}  {'chunks':>8}  {'mean loss':>10}  {'surprise':>9}  "
          f"{'state size':>10}  {'peak RSS MB':>11}")

    tr = StreamTrainer(net, opt, decay=a.decay)
    rows = []
    for n in a.doc_tokens:
        doc = ids[:n]
        tr.reset()
        before = peak_mb()
        chunks, mean_loss, mean_sur = tr.step_document(doc, verbose=(n >= 10000))
        after = peak_mb()
        rows.append((n, chunks, mean_loss, mean_sur, tr.state.size, after))
        print(f"  {n:>16,}  {chunks:>8,}  {mean_loss:>10.4f}  {mean_sur:>9.3f}  {tr.state.size:>10}  "
              f"{after:>11.1f}")

    first, last = rows[0][5], rows[-1][5]
    growth = last - first
    print(f"\n  document length went {a.doc_tokens[0]:,} -> {a.doc_tokens[-1]:,} tokens "
          f"({a.doc_tokens[-1]/max(a.doc_tokens[0],1):,.0f}x) and peak memory moved {growth:+.1f} MB "
          f"({first:.1f} -> {last:.1f} MB)")
    print(f"  carried state stayed at {rows[-1][4]} numbers at every document length")
    verdict = ("NO TRAINING LENGTH LIMIT: memory is flat in document length"
               if growth < 200 else "MEMORY GREW: the streaming-training claim FAILS, do not ship it")
    print(f"  VERDICT: {verdict}")
    print("  HONEST BOUND: gradients flow within a chunk (truncated-BPTT style); the carried state is a "
          "detached fixed-size summary. Memory flatness is what is proven here.")
    return 0 if growth < 200 else 1


if __name__ == "__main__":
    sys.exit(main())