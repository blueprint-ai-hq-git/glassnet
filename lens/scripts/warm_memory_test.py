#!/usr/bin/env python3
"""warm_memory_test.py - the warm concept-memory tier, measured on held-out text.

Predictions for every claim here were written BEFORE the run in
lens/runs/WARM-MEMORY-PREDICTIONS.txt (two rounds: the original P18-P22, then P23-P26 for the fix that the
first round's FAIL forced). The verdicts at the end are binary (PASS/FAIL) against the bars declared in that
file, so nothing can be re-defined after seeing the numbers.

ROUND 1 RESULT, KEPT ON THE RECORD: the plain 1-byte store PASSED size, answer-preservation and recall but
FAILED the readability bar - it read back the wrong top-1 concept NAME on 6.35% of held-out windows
(93.65% identical against a >= 99% bar). That FAIL is not redefined; round 2 measures the fix.

WHAT IS MEASURED
  P18  bytes per token stored, and what 1 GB therefore holds
  P19  the quantised store must not change the answer (top-1 token vs the float32 path)
  P20  readability: the stored bytes must NAME what they remember            <- FAILED for plain u8
  P21  the store's bytes == arithmetic exactly, and nothing else grows with window count
  P22  recall of an EARLY window after the whole stream, with no network call in the recall path
  P23  the fix: keep the top-1 index in one extra byte per window
  P24  the alternative: two bytes per concept (uint16)
  P25  are the plain-u8 name flips float32 near-ties, or lost information?
  P26  which mode ships as the default

SPLITS, STATED EVERYWHERE: every accuracy, readability and recall number is measured on the HELD-OUT split
(which=1) of data/chat/prepared_mix_honest. The one number measured on the train split is the 1M-token store
size / throughput, and it is labelled TRAIN wherever it appears - it is a size measurement, not a claim about
learning anything.

Usage: CHAT_DATA_DIR=data/chat/prepared_mix_honest python3 lens/scripts/warm_memory_test.py
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
SEED = int(os.environ.get("SEED", "20261020"))
N_CHECK = int(os.environ.get("N_CHECK", "2000"))       # windows sampled for the correctness checks
N_RECALL = int(os.environ.get("N_RECALL", "500"))      # early windows recalled after the whole stream
CHUNK = 512                                            # windows per forward pass (keeps RAM flat)

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens import backend  # noqa: E402
from lens.warmmemory import WarmConceptMemory, answer_from_memory, describe_limits  # noqa: E402

OUT = {}


def peak_mb():
    """High-water mark of this process (ru_maxrss)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rss_mb():
    """CURRENT resident memory (not the high-water mark), so growth is visible even below an old peak."""
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1024.0 / 1024.0


def concepts_of(net, X):
    """The concept layer's output for a batch of windows, float32, on the host."""
    return np.asarray(backend.to_cpu(net.forward_arrays(np.asarray(X, dtype=np.int64))["concepts"]),
                      dtype=np.float32)


def log(line=""):
    print(line, flush=True)


def evaluate(mem, H, sample, names, answer_W, answer_b):
    """Every claim about a store, measured on the SAME held-out sample, for any precision mode."""
    max_err, top1_same, name_same, named = 0.0, 0, 0, 0
    flips = []
    t0 = time.time()
    for i in sample:
        i = int(i)
        h_f = H[i]                                  # the float32 concepts the model actually computed
        h_q = mem.recall(i)                         # what the store kept, read back
        max_err = max(max_err, float(np.abs(h_f - h_q).max()))
        lf = h_f @ answer_W + answer_b
        ls = answer_from_memory(mem, i, answer_W, answer_b)     # STORE ONLY - no network call
        top1_same += int(int(lf.argmax()) == int(ls.argmax()))
        nf, ns = names[int(h_f.argmax())], names[mem.top1_index(i)]
        name_same += int(nf == ns)
        if nf != ns:
            s = np.sort(h_f)[::-1]
            flips.append(float(s[0] - s[1]))
        named += int(len(mem.top(i, names, k=mem.K, thresh=0.5)) > 0)
    n = len(sample)
    near = int(sum(1 for d in flips if d <= 1.0 / 255.0))
    return {"n": n, "max_err": max_err, "top1_token_pct": 100.0 * top1_same / n,
            "name_pct": 100.0 * name_same / n, "named_pct": 100.0 * named / n,
            "flips": len(flips), "flips_near_tie": near,
            "near_tie_pct_of_flips": (100.0 * near / len(flips)) if flips else float("nan"),
            "ms_per_window": (time.time() - t0) / n * 1000,
            "bytes_per_token": mem.bytes_per_token(), "bytes_per_window": mem.bytes_per_window()}


# ================================================================================================
# PART 0 - load
# ================================================================================================
net = lens.load(SHIP, device="cpu")
names = [c.name for c in net.concepts]
K, W, D, V = int(net.K), int(net.W), int(net.D), int(net.V)
answer_W = np.asarray(backend.to_cpu(net.answer.W.data), dtype=np.float32)
answer_b = np.asarray(backend.to_cpu(net.answer.b.data), dtype=np.float32)

log("=" * 100)
log("WARM CONCEPT MEMORY TIER - measured on held-out text")
log("=" * 100)
log(f"ship {os.path.basename(SHIP)}   d={D} window={W} named concepts={K} vocab={V} attention={net.attn is not None}")
log(f"predictions written before the run: lens/runs/WARM-MEMORY-PREDICTIONS.txt   seed {SEED}")

d = chat_data.build()                                  # CHAT_DATA_DIR is set by the caller
ids = np.asarray(d["ids"], dtype=np.int64)
bounds, split, vocab = d["bounds"], d["split"], d["vocab"]
log(f"corpus: {len(ids):,} tokens, {len(bounds):,} conversations, RSS {rss_mb():.1f} MB (peak {peak_mb():.1f} MB)")

# ================================================================================================
# PART 1 - P18: the arithmetic, per mode
# ================================================================================================
log()
log("-" * 100)
log("P18  SIZE (arithmetic: one byte per named concept per window, +1 byte if the top-1 name is kept)")
log("-" * 100)
MODES = (("u8 (plain)", K, 0), ("u8+top1 (shipped default)", K + 1, 0), ("u16", 2 * K, 0))
for label, per_win, _ in MODES:
    for stride in (W, max(1, W // 2)):
        bpt = per_win / float(stride)
        log(f"  {label:<26} stride {stride:>3}:  {bpt:>6.3f} bytes/token   1 GB = {int(1e9/bpt):>12,} tokens   "
            f"(2^30 = {int(2**30/bpt):>12,})")
bpt_u8 = K / float(W)
bpt_def = (K + 1) / float(W)
OUT.update({"p18_bytes_per_token_u8_stride32": bpt_u8, "p18_bytes_per_token_default_stride32": bpt_def,
            "p18_tokens_per_gb_default_stride32": int(1e9 / bpt_def),
            "p18_tokens_per_gib_default_stride32": int(2 ** 30 / bpt_def)})
log(f"  doc figure to check against: FROM-SCRATCH.md GAP 4 said ~1.3 B/token / 835M tokens for a 41-concept")
log(f"  window; this ship has {K} concepts, so the measured value is the one that counts from now on.")

# ================================================================================================
# PART 2 - the held-out stream, stored three ways
# ================================================================================================
log()
log("-" * 100)
log("THE STREAM: every held-out window, in corpus order (windows never cross a conversation boundary)")
log("-" * 100)
Xva, Yva = chat_data.make_windows(ids, bounds, split, W, which=1)
log(f"  held-out windows available: {len(Xva):,}  (tokens covered at stride {W//2}: {len(Xva)*(W//2):,})")
rss_before_stream = rss_mb()

mem_u8 = WarmConceptMemory(n_concepts=K, window=W, stride=W // 2, store_top1=False)   # round-1 store
mem_def = WarmConceptMemory(n_concepts=K, window=W, stride=W // 2, store_top1=True)    # the fix
mem_u16 = WarmConceptMemory(n_concepts=K, window=W, stride=W // 2, dtype=np.uint16, store_top1=False)
H_list = []
t0 = time.time()
for s in range(0, len(Xva), CHUNK):
    H = concepts_of(net, Xva[s:s + CHUNK])
    H_list.append(H)
    mem_u8.add_many(H)
    mem_def.add_many(H)
    mem_u16.add_many(H)
store_s = time.time() - t0
H = np.concatenate(H_list, axis=0)
del H_list
rep = mem_u8.report()
log(f"  stored {rep['windows']:,} windows covering {rep['tokens_covered']:,} held-out tokens in {store_s:.2f} s "
    f"({rep['windows']/max(store_s,1e-9):,.0f} windows/s, {rep['tokens_covered']/max(store_s,1e-9):,.0f} tokens/s)")
log(f"  plain u8 store  {rep['bytes']:,} bytes  == windows x concepts: {rep['bytes'] == rep['windows']*K}")
log(f"  default store   {mem_def.nbytes():,} bytes  == windows x (concepts+1): "
    f"{mem_def.nbytes() == mem_def.n * (K+1)}  (top-1 index kept: {mem_def.report()['top1_kept']})")
OUT.update({"stream_windows": rep["windows"], "stream_tokens": rep["tokens_covered"],
            "stream_store_bytes_u8": rep["bytes"], "stream_store_bytes_default": mem_def.nbytes()})

rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(len(Xva), size=min(N_CHECK, len(Xva)), replace=False))

# ================================================================================================
# PART 3 - ROUND 1 verdicts, reproduced exactly on the plain u8 store
# ================================================================================================
log()
log("-" * 100)
log("P19 / P20  ROUND 1, on the PLAIN 1-byte store (reproduced, so the recorded FAIL stays on the record)")
log("-" * 100)
m_u8 = evaluate(mem_u8, H, sample, names, answer_W, answer_b)
log(f"  held-out windows sampled at random (seed {SEED}): {m_u8['n']:,}")
log(f"  max absolute concept error from the 1-byte store: {m_u8['max_err']:.3e}   (bar <= 2.0e-3)")
log(f"  top-1 next token, store versus float32:           {m_u8['top1_token_pct']:.2f}% identical   (bar >= 99.00%)")
log(f"  top-1 concept NAME, store versus float32:         {m_u8['name_pct']:.2f}% identical   (bar >= 99.00%)")
log(f"  windows with at least one named concept > 0.5:    {m_u8['named_pct']:.2f}%              (bar = 100.00%)")
log(f"  mean recall time per window (store only, no net): {m_u8['ms_per_window']:.3f} ms")
OUT.update({"p19_max_concept_error_u8": m_u8["max_err"], "p19_top1_token_pct_u8": m_u8["top1_token_pct"],
            "p20_top1_name_pct_u8": m_u8["name_pct"], "p20_named_pct_u8": m_u8["named_pct"]})

# ================================================================================================
# PART 3b - P23/P24/P25: the three modes on the SAME held-out sample
# ================================================================================================
log()
log("-" * 100)
log("P23 / P24 / P25  THE THREE MODES, same held-out windows, same seed")
log("-" * 100)
m_def = evaluate(mem_def, H, sample, names, answer_W, answer_b)
m_u16 = evaluate(mem_u16, H, sample, names, answer_W, answer_b)
log(f"  {'mode':<24} {'B/token@32':>10} {'max error':>11} {'top-1 token':>12} {'top-1 NAME':>11} {'named':>7}")
for label, m in (("u8 (plain)", m_u8), ("u8+top1 (default)", m_def), ("u16", m_u16)):
    log(f"  {label:<24} {m['bytes_per_token']/2:>10.3f} {m['max_err']:>11.3e} {m['top1_token_pct']:>11.2f}% "
        f"{m['name_pct']:>10.2f}% {m['named_pct']:>6.2f}%")
log(f"  (B/token@32 column is bytes per token at stride 32; the stream above ran at stride {W//2}, i.e. double)")
log()
log(f"  P25 WHY PLAIN u8 FLIPPED A NAME: {m_u8['flips']} flips of {m_u8['n']:,} windows; "
    f"{m_u8['flips_near_tie']} of them ({m_u8['near_tie_pct_of_flips']:.1f}%) are float32 NEAR-TIES "
    f"(top-1 minus top-2 <= 1/255 = 3.922e-03),")
log(f"  i.e. in those windows the float32 top-1 was already decided by a difference smaller than one "
    f"quantisation step. The other {m_u8['flips']-m_u8['flips_near_tie']} are order information the 1-byte")
log(f"  store genuinely loses. The name is what a person READS, so the default keeps it exactly (P23).")
OUT.update({"p23_name_pct_default": m_def["name_pct"], "p23_bytes_per_token_default": m_def["bytes_per_token"],
            "p23_top1_token_pct_default": m_def["top1_token_pct"], "p24_name_pct_u16": m_u16["name_pct"],
            "p24_max_err_u16": m_u16["max_err"], "p24_top1_token_pct_u16": m_u16["top1_token_pct"],
            "p25_flips": m_u8["flips"], "p25_flips_near_tie": m_u8["flips_near_tie"],
            "p25_near_tie_pct_of_flips": m_u8["near_tie_pct_of_flips"]})

log()
log("  WHAT IT REMEMBERS, read back from the shipped default store (first / middle / last held-out window).")
log("  The words under each trace are the ACTUAL held-out tokens, printed beside the memory so the memory")
log("  can be judged - the store itself contains none of them.")
for label, i in (("first", int(sample[0])), ("middle", int(sample[len(sample) // 2])), ("last", int(sample[-1]))):
    txt = " ".join(vocab[t] for t in Xva[i])
    log(f"    [{label}] window {i:,}  ->  {mem_def.trace(i, names)}")
    log(f"           actual held-out tokens: {txt[:150]}")

# ================================================================================================
# PART 4 - P22 recall without recompute
# ================================================================================================
log()
log("-" * 100)
log("P22  RECALL OF AN EARLY WINDOW AFTER THE WHOLE STREAM (store + answer weights only, no network)")
log("-" * 100)
early = np.sort(rng.choice(np.arange(0, min(1000, len(Xva))), size=min(N_RECALL, 1000), replace=False))
for label, mem in (("plain u8", mem_u8), ("default (u8+top1)", mem_def)):
    same = sum(int(int((H[int(i)] @ answer_W + answer_b).argmax())
                   == int(answer_from_memory(mem, int(i), answer_W, answer_b).argmax())) for i in early)
    pct = 100.0 * same / len(early)
    log(f"  {label:<20} answer from the store equals the float32 answer: {pct:.2f}%   (bar >= 99.00%)")
    OUT[f"p22_recall_pct_{label.split()[0]}"] = pct
log(f"  windows recalled: {len(early):,} drawn from the first 1,000 of {rep['windows']:,} stored windows")
log(f"  the recall path is answer_from_memory(mem, i, W, b): it is handed the store and two weight arrays")
log(f"  and nothing else, so a correct answer cannot be coming from a recompute.")

# ================================================================================================
# PART 5 - P21 the store is the only thing growing
# ================================================================================================
log()
log("-" * 100)
log("P21  MEMORY: the store's bytes == arithmetic, and nothing else grows with the window count")
log("-" * 100)
rss_after_stream = rss_mb()
# a real 1M-token store, measured. TRAIN split, labelled: this is a SIZE measurement only.
Xtr, _ = chat_data.make_windows(ids, bounds, split, W, which=0, max_windows=32000, seed=SEED)
big = WarmConceptMemory(n_concepts=K, window=W, stride=W, store_top1=False)   # stride 32: the headline figure
t0 = time.time()
for s in range(0, len(Xtr), CHUNK):
    big.add_many(concepts_of(net, Xtr[s:s + CHUNK]))
big_s = time.time() - t0
del Xtr
rep_big = big.report()
rss_after_big = rss_mb()
log(f"  TRAIN split (size only, no accuracy claim): {rep_big['windows']:,} windows = {rep_big['tokens_covered']:,} tokens")
log(f"    plain u8 store {rep_big['bytes']:,} bytes == windows x concepts: {rep_big['bytes'] == rep_big['windows']*K}")
log(f"    arithmetic for 1,000,000 tokens at stride {W}: u8 {int(1e6*bpt_u8):,} bytes "
    f"({1e6*bpt_u8/1024/1024:.2f} MB), default {int(1e6*bpt_def):,} bytes ({1e6*bpt_def/1024/1024:.2f} MB)")
log(f"    time to store them: {big_s:.2f} s ({rep_big['tokens_covered']/max(big_s,1e-9):,.0f} tokens/s)")
log(f"  RESIDENT memory (current, not the high-water mark) - read the exact byte accounting above as the")
log(f"  real measurement: this process holds the whole {len(ids):,}-token corpus, so RSS moves in tens of MB")
log(f"  from allocator noise while the stores are single-digit MB.")
log(f"    after loading model + corpus + windows: {rss_before_stream:.1f} MB (before any store)")
log(f"    after the held-out stream, 3 stores:   {rss_after_stream:.1f} MB   "
    f"(the three stores hold {(mem_def.nbytes()+mem_u8.nbytes()+mem_u16.nbytes())/1024/1024:.2f} MB of concepts)")
log(f"    after +{rep_big['tokens_covered']:,} more tokens:   {rss_after_big:.1f} MB   "
    f"(delta {rss_after_big-rss_after_stream:+.2f} MB; the new store is exactly {rep_big['bytes']:,} bytes "
    f"= {rep_big['bytes']/1024/1024:.2f} MB, the rest is noise)")
log(f"    peak RSS for the whole run: {peak_mb():.1f} MB (dominated by the corpus, not the store)")
log()
log(f"  comparison at this d={D}, one row per token:")
log(f"    hot tier (K/V/O per token, lens/kvcache.py):  {3*D*4:,} bytes/token  = {3*D*4/bpt_def:,.0f}x the shipped warm tier")
log(f"    raw tokens losslessly (int16):                {2:,} bytes/token  = {2/bpt_def:,.1f}x the shipped warm tier")
log(f"    WARM tier (named concepts + exact top-1 name): {bpt_def:.3f} bytes/token at stride {W}, "
    f"{(K+1)/float(W//2):.3f} at stride {W//2}")
OUT.update({"big_windows": rep_big["windows"], "big_tokens": rep_big["tokens_covered"],
            "big_store_bytes_u8": rep_big["bytes"], "bytes_1m_tokens_u8": int(1e6 * bpt_u8),
            "bytes_1m_tokens_default": int(1e6 * bpt_def), "hot_bytes_per_token": 3 * D * 4,
            "rss_after_stream_mb": rss_after_stream, "rss_after_big_mb": rss_after_big,
            "peak_mb": peak_mb()})

store_path = "lens/runs/warm_memory_heldout.npz"
np.savez_compressed(store_path, Q=mem_def.Q[:mem_def.n], T1=mem_def.T1[:mem_def.n], n=np.int64(mem_def.n),
                    K=np.int64(K), W=np.int64(W), stride=np.int64(mem_def.stride), split=np.int64(1),
                    ship=np.asarray(SHIP), names=np.asarray(names))
disk = os.path.getsize(store_path)
log()
log(f"  the held-out store written to {store_path}: {disk:,} bytes on disk "
    f"({mem_def.nbytes():,} raw, the rest is the zip container and the names)")

# ================================================================================================
# VERDICTS - binary, against the bars declared before each run
# ================================================================================================
log()
log("=" * 100)
log("VERDICTS (bars fixed before the run in lens/runs/WARM-MEMORY-PREDICTIONS.txt)")
log("=" * 100)
v18 = "PASS" if bpt_u8 <= 1.30 else "FAIL"
v19 = "PASS" if (m_u8["max_err"] <= 2e-3 and m_u8["top1_token_pct"] >= 99.0) else "FAIL"
v20 = "PASS" if (m_u8["name_pct"] >= 99.0 and m_u8["named_pct"] >= 99.9) else "FAIL"
v21 = "PASS" if (rep["bytes"] == rep["windows"] * K and rep_big["bytes"] == rep_big["windows"] * K
                and mem_def.nbytes() == mem_def.n * (K + 1) and rep_big["bytes"] <= 2 * 1024 * 1024) else "FAIL"
v22 = "PASS" if OUT["p22_recall_pct_default"] >= 99.0 else "FAIL"
v23 = "PASS" if (m_def["name_pct"] == 100.0 and (K + 1) / float(W) <= 1.30) else "FAIL"
v24 = "PASS" if (m_u16["max_err"] <= 8e-6 and m_u16["top1_token_pct"] >= 99.90) else "FAIL"
v25 = "PASS" if m_u8["near_tie_pct_of_flips"] >= 50.0 else "FAIL"
v26 = "PASS" if (v23 == "PASS" and v20 == "FAIL") else "FAIL"
log(f"  P18 size {bpt_u8:.3f} B/token at stride {W}, 1 GB = {int(1e9/bpt_u8):,} tokens (bar <= 1.30)        {v18}")
log(f"  P19 quantisation max error {m_u8['max_err']:.3e}, top-1 token {m_u8['top1_token_pct']:.2f}% (bars <=2e-3, >=99%)    {v19}")
log(f"  P20 readable, PLAIN u8: name {m_u8['name_pct']:.2f}%, named windows {m_u8['named_pct']:.2f}% (bars >=99%)      {v20}")
log(f"  P21 store bytes exact in all modes, 1M tokens in {rep_big['bytes']/1024/1024:.2f} MB (bar <2MB)      {v21}")
log(f"  P22 early-window recall {OUT['p22_recall_pct_default']:.2f}% from the store alone (bar >=99%)          {v22}")
log(f"  P23 fix: default name agreement {m_def['name_pct']:.2f}%, {bpt_def:.3f} B/token at stride {W} (bars =100%, <=1.30)  {v23}")
log(f"  P24 u16: max error {m_u16['max_err']:.3e}, top-1 token {m_u16['top1_token_pct']:.2f}% (bars <=8e-6, >=99.9%)   {v24}")
log(f"  P25 flips that are float32 near-ties: {m_u8['near_tie_pct_of_flips']:.1f}% (bar >= 50%)                {v25}")
log(f"  P26 the cheapest passing mode is the default: u8+top1 (P23 {v23}, P20 plain u8 {v20})   {v26}")
OUT.update({"P18": v18, "P19": v19, "P20": v20, "P21": v21, "P22": v22, "P23": v23, "P24": v24,
            "P25": v25, "P26": v26,
            "shipped_default": "u8 + top1 index" if v23 == "PASS" else "u16"})
log()
log("  HONEST LIMITS, unchanged by any of the above:")
log(f"    {describe_limits()}")
log(f"    The warm tier has still never been used inside a generation loop - wiring it into lens/decode.py")
log(f"    is a separate step and is NOT claimed here.")
log(f"    Every accuracy/readability/recall number above is the HELD-OUT split (which=1); the one TRAIN")
log(f"    number is the 1M-token store size and it is labelled.")

with open("lens/runs/warm_memory.json", "w") as f:
    json.dump(OUT, f, indent=2)
log()
log("  written: lens/runs/warm_memory.json")
sys.exit(0 if all(v == "PASS" for v in (v18, v19, v21, v22, v23, v24, v25, v26)) and v20 == "FAIL" else 1)
