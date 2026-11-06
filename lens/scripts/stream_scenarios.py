#!/usr/bin/env python3
"""stream_scenarios.py - the no-limit claim, tested in EVERY scenario rather than the easy one.

The user asked for exactly this: "test it in every scenario man, we don't need any overflow". So this
covers, with measured numbers on each:

  S1 random tokens at 1k / 10k / 100k / 1M / 2M          (does memory grow with length at all?)
  S2 REAL corpus text, 100k tokens                        (random ids can hide a bug; real text cannot)
  S3 many concurrent streams, interleaved, 8 x 50k        (does running several at once accumulate?)
  S4 very long output, 10,000 tokens generated            (is the output side really uncapped?)
  S5 GPU device if available                              (is the flatness a CPU-only accident?)

Each scenario reports peak RSS (or VRAM) before and after, plus the fixed-size state it kept. A scenario
that grows memory is reported as a FAILURE of the claim, not smoothed over.
"""
import os
import resource
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens.decode import generate  # noqa: E402
from lens.stream import StreamingEncoder  # noqa: E402

SHIP = os.environ.get("SHIP", "lens/runs/chat/sb_S0_d256_noattn_ep96.npz")


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def line(tag, before, after, note=""):
    delta = after - before
    flag = "OK  " if delta < 200 else "GROW"
    print(f"  [{flag}] {tag:<46} {before:8.1f} -> {after:8.1f} MB  ({delta:+7.1f})  {note}")
    return delta


def main():
    net = lens.load(SHIP, device="cpu")
    V, W, K = int(net.V), int(net.W), int(net.K)
    rng = np.random.default_rng(0)
    print(f"NO-LIMIT SCENARIOS - ship {os.path.basename(SHIP)}  window {W}  concepts {K}")
    print(f"  baseline after load: {peak_mb():.1f} MB\n")

    print("S1 input length, random tokens")
    for n in (1_000, 10_000, 100_000, 1_000_000, 2_000_000):
        b = peak_mb()
        enc = StreamingEncoder(net)
        ids = rng.integers(0, V, size=n)
        for i in range(0, n, 50_000):
            enc.feed(ids[i:i + 50_000])
        enc.finish()
        line(f"    {n:,} tokens", b, peak_mb(), f"buffer {len(enc.buf)} state {enc.state.size}")
        del enc, ids

    print("\nS2 REAL corpus text (random ids can hide a bug, real text cannot)")
    d = chat_data.build()
    vocab = d["vocab"]
    ids_all = np.asarray(d["ids"], dtype=np.int64)
    for n in (100_000, 500_000):
        b = peak_mb()
        enc = StreamingEncoder(net)
        real = ids_all[:n]
        for i in range(0, n, 50_000):
            enc.feed(real[i:i + 50_000])
        enc.finish()
        line(f"    {n:,} real tokens", b, peak_mb(), f"state {enc.state.size}")
        del enc

    print("\nS3 eight concurrent streams, interleaved (does running several at once accumulate?)")
    b = peak_mb()
    encs = [StreamingEncoder(net) for _ in range(8)]
    chunks = [rng.integers(0, V, size=50_000) for _ in range(8)]
    for i in range(0, 50_000, 5_000):
        for e, c in zip(encs, chunks):
            e.feed(c[i:i + 5_000])
    for e in encs:
        e.finish()
    line("    8 streams x 50,000 tokens = 400,000", b, peak_mb(),
         f"total buffers {sum(len(e.buf) for e in encs)} states {sum(e.state.size for e in encs)}")
    del encs, chunks

    print("\nS4 output length, 10,000 tokens generated (stop rules OFF so nothing stops it early)")
    vmap = {w: i for i, w in enumerate(vocab)}
    b = peak_mb()
    g = generate(net, vocab, vmap, "human : tell me a story assistant :", max_new=10_000, rule_stop=False)
    line(f"    {g['tokens']:,} tokens generated", b, peak_mb(), f"stop '{g['stop_reason']}'")

    print("\nS5 GPU device (is the flatness a CPU-only accident?)")
    try:
        import cupy  # noqa: F401
        gpu_net = lens.load(SHIP, device="gpu")
        free, total = cupy.cuda.runtime.memGetInfo()
        used0 = (total - free) / 1e6
        enc = StreamingEncoder(gpu_net)
        for i in range(0, 200_000, 50_000):
            enc.feed(rng.integers(0, V, size=50_000))
        enc.finish()
        free, total = cupy.cuda.runtime.memGetInfo()
        used1 = (total - free) / 1e6
        print(f"  [{'OK  ' if (used1 - used0) < 200 else 'GROW'}] 200,000 tokens on GPU"
              f"{'':<22} VRAM {used0:8.1f} -> {used1:8.1f} MB  ({used1-used0:+7.1f})")
    except ImportError:
        print("  [skip] cupy not importable here -- GPU scenario not run (not claimed either)")
    except Exception as e:
        print(f"  [fail] GPU scenario errored: {type(e).__name__}: {e}")

    print("\nSUMMARY: every scenario above that is not marked GROW kept memory flat. Anything marked GROW "
          "would be a failure of the no-limit claim and must be fixed before shipping.")


if __name__ == "__main__":
    main()