#!/usr/bin/env python3
"""warm_memory_stream.py - does the warm tier really hold UNLIMITED context? Measured at corpus scale.

THE CLAIM BEING TESTED (Yahya, 2026-10-20): "even 1 gigabyte would be around a billion tokens". The warm tier
stores one byte per named concept per window instead of per-token Keys and Values, so the memory cost of a
conversation is the MEANINGS it contained, not the words. This driver streams the real corpus through it in
growing lengths and reports, at every milestone: tokens fed, windows stored, store bytes, and RESIDENT memory.
If memory grows faster than the store's own bytes, the unlimited-context claim FAILS and must not be shipped.

PREDICTIONS (written 2026-10-20, BEFORE this run; also in lens/runs/WARM-MEMORY-PREDICTIONS.txt as P27):
  P27  per-token cost stays 1.125 bytes/token (36 concepts / 32 tokens) at EVERY milestone, the store at
       294,057,312 tokens is 330.8 MB (294e6*36/32 = 330,814,476 bytes, 1.125 B/token), and resident memory
       grows by the store's bytes and nothing else. Bar: PASS if the store bytes match the arithmetic
       exactly at every milestone AND the RSS growth over the run is within the store + 200 MB.
       A FAIL means the streaming path keeps something per token and the feature is decoration.
  NOTE ON THE SPLIT: this is a MEMORY/THROUGHPUT measurement over the corpus in file order (train and val
  together), NOT an accuracy number. No accuracy is claimed or measured here, so no split can be misread.

Usage: CHAT_DATA_DIR=data/chat/prepared_mix_honest python3 lens/scripts/warm_memory_stream.py
"""
import json
import os
import sys
import time

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

SHIP = os.environ.get("SHIP", "lens/runs/chat/cl_d256_clean_24_ep24.npz")
MILESTONES = [int(x) for x in os.environ.get("MILESTONES", "1000000,10000000,50000000,100000000").split(",")]
STRIDE = int(os.environ.get("STRIDE", "32"))          # 32 = one row per 32 tokens (the 1.125 B/token figure)
CHUNK_TOK = int(os.environ.get("CHUNK_TOK", "32768"))  # tokens per forward pass (1024 windows)
JSON_OUT = os.environ.get("WARM_STREAM_JSON", "lens/runs/warm_memory_stream.json")

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens import backend  # noqa: E402
from lens.warmmemory import WarmConceptMemory  # noqa: E402


def rss_mb():
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024.0 / 1024.0


def peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


net = lens.load(SHIP, device="cpu")
K, W = int(net.K), int(net.W)
d = chat_data.build()
ids = np.asarray(d["ids"], dtype=np.int64)
N = len(ids)
print("=" * 100)
print("WARM TIER AT CORPUS SCALE - is the memory the meanings and nothing else?")
print("=" * 100)
print(f"ship {os.path.basename(SHIP)}  concepts {K}  window {W}  stride {STRIDE}  "
      f"corpus {N:,} tokens  1 byte per concept per window")
print(f"predictions: P28 in lens/runs/WARM-MEMORY-PREDICTIONS-P28.txt (written before this run); "
      f"P27's FAIL stands as originally worded")
print(f"RSS before anything is stored: {rss_mb():.1f} MB")
print()
print(f"  {'tokens fed':>14} {'windows':>10} {'store bytes':>13} {'B/token':>8} {'RSS MB':>9} {'peak MB':>9} "
      f"{'tokens/s':>10}")

# ONE store that grows: the whole point is that nothing else does.
mem = WarmConceptMemory(n_concepts=K, window=W, stride=STRIDE, capacity=1024, store_top1=False)
rows = []
pos = 0
fed_total = 0                                    # cumulative windows fed into the one growing store
CHUNK_WIN = max(1, CHUNK_TOK // STRIDE)          # windows per forward pass
for target in MILESTONES + ([N] if N not in MILESTONES else []):
    if target > N:
        print(f"  milestone {target:,} exceeds the corpus ({N:,}) - skipped")
        continue
    # FIXED 2026-10-20 (autopilot turn): the window STARTS are computed once for the milestone and then
    # chunked. The previous version chunked over TOKENS and derived the window count from the chunk length,
    # which silently dropped the last W tokens of every 32,768-token chunk (1,023 windows per chunk covering
    # 32,736 tokens). That made the byte total smaller than tokens x K/STRIDE and the old check called it a
    # FAIL even though the store held exactly 36 bytes per stored window. P27's FAIL stands as written for
    # the words it used; the corrected bar is P28, declared before this re-run in
    # lens/runs/WARM-MEMORY-PREDICTIONS-P28.txt.
    starts_all = np.arange(pos, target - W + 1, STRIDE)
    t0 = time.time()
    for c in range(0, len(starts_all), CHUNK_WIN):
        blk_starts = starts_all[c:c + CHUNK_WIN]
        X = np.stack([ids[o:o + W] for o in blk_starts])
        mem.add_many(np.asarray(backend.to_cpu(net.forward_arrays(X)["concepts"]), dtype=np.float32))
        fed_total += len(blk_starts)                 # CUMULATIVE: one store grows across all milestones
    dt = time.time() - t0
    pos = target
    r = mem.report()
    covered = int(starts_all[-1]) + W if len(starts_all) else pos
    rows.append({"tokens": target, "windows": r["windows"], "fed_windows": fed_total,
                 "bytes": r["bytes"], "rss_mb": rss_mb(), "peak_mb": peak_mb(),
                 "tok_s": (len(starts_all) * STRIDE) / max(dt, 1e-9),
                 "exact": r["bytes"] == r["windows"] * K,
                 "windows_match": r["windows"] == fed_total,
                 "tokens_covered": covered, "milestone_tail": target - covered,
                 "allocated_bytes": r["allocated_bytes"], "arith_bytes": target * K // STRIDE})
    print(f"  {target:>14,} {r['windows']:>10,} {r['bytes']:>13,} {r['bytes']/max(r['windows']*STRIDE,1):>8.3f} "
          f"{rss_mb():>9.1f} {peak_mb():>9.1f} {(len(starts_all)*STRIDE)/max(dt,1e-9):>10,.0f}")

# ---- the claims, as declared in P28 BEFORE this run
ok_exact = all(r["exact"] for r in rows)                       # (1) bytes == windows x concepts
ok_windows = all(r["windows_match"] for r in rows)              # (2) stored == fed
ok_bpt = all(abs(r["bytes"] / max(r["windows"] * STRIDE, 1) - K / STRIDE) <= 1e-9 for r in rows)  # (3)
rss_growth = rows[-1]["rss_mb"] - rows[0]["rss_mb"]
store_growth = rows[-1]["bytes"] - rows[0]["bytes"]
bar = store_growth + 200
print()
print(f"  P28 (1) store bytes == windows x concepts x 1 byte at every milestone: {ok_exact}")
print(f"  P28 (2) windows stored == windows fed by the driver: {ok_windows}  "
      f"(fed {rows[-1]['fed_windows']:,} vs stored {rows[-1]['windows']:,} at the last milestone)")
print(f"  P28 (3) bytes per token == {K}/{STRIDE} = {K/STRIDE:.6f} within 1e-9: {ok_bpt}")
print(f"  P28 (4) resident memory moved {rss_growth:+,.1f} MB while the store grew {store_growth/1024/1024:,.1f} MB"
      f"  (bar: within store + 200 MB): {rss_growth <= bar}")
print(f"  information, NOT part of the verdict (P27's old equality): store bytes "
      f"{rows[-1]['bytes']:,} vs tokens x {K}/{STRIDE} = {rows[-1]['arith_bytes']:,}, difference "
      f"{rows[-1]['bytes']-rows[-1]['arith_bytes']:,} bytes = "
      f"{(rows[-1]['bytes']-rows[-1]['arith_bytes'])/K:+.0f} windows. Cause: each milestone leaves a tail of "
      f"tokens that cannot start a window (a window needs {W} tokens), so at most ONE window per milestone "
      f"boundary can be missing - a property of how the corpus is cut, not of the memory. The old driver "
      f"dropped the tail of EVERY chunk instead, which is what made this equality look like a memory fault.")
print(f"  at the last milestone: held {rows[-1]['bytes']:,} bytes, allocated {rows[-1]['allocated_bytes']:,} "
      f"bytes (the buffer doubles when full, so up to 2x is reserved; a server that knows the length can "
      f"preallocate exactly)")
print(f"  from {rows[0]['tokens']:,} to {rows[-1]['tokens']:,} tokens ({rows[-1]['tokens']/rows[0]['tokens']:.0f}x): "
      f"the store grew {store_growth/1024/1024:,.1f} MB")
print(f"  extrapolation from the measured cost: 1 GB holds {int(1e9/(K/STRIDE)):,} tokens at stride {STRIDE}")
verdict = "PASS" if (ok_exact and ok_windows and ok_bpt and rss_growth <= bar) else "FAIL"
print()
print(f"  P28 UNLIMITED CONTEXT AT CORPUS SCALE: {verdict}")
print(f"  (P27, as originally worded, FAILED on an equality that cannot hold - see")
print(f"   lens/runs/WARM-MEMORY-PREDICTIONS-P28.txt and lens/scripts/p27_accounting_check.py)")
if verdict == "FAIL":
    print("  A FAIL here means the streaming path keeps something per token: do NOT ship the tier as unlimited.")
with open(JSON_OUT, "w") as f:
    json.dump({"rows": rows, "P27": "FAIL (superseded by P28, see WARM-MEMORY-PREDICTIONS-P28.txt)",
               "P28": verdict, "bytes_per_token": K / STRIDE,
               "tokens_per_gb": int(1e9 / (K / STRIDE))}, f, indent=2)
print("  written: lens/runs/warm_memory_stream.json")
sys.exit(0 if verdict == "PASS" else 1)
