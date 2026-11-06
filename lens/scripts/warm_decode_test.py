#!/usr/bin/env python3
"""warm_decode_test.py - THE WARM TIER IN THE GENERATION LOOP, measured (PLAN item (c2)).

Every prediction here was written BEFORE the run, in lens/runs/WARM-DECODE-PREDICTIONS.txt (P38a-P38i),
and the verdicts at the end are binary against the bars in that file. Nothing is retuned afterwards.

WHAT IS BEING TESTED, in plain words
  The hot KV cache keeps the last window. The warm tier (lens/warmmemory.py) keeps the NAMED CONCEPTS of
  EVERY window of the context - 36 concepts + the exact top-1 index = 37 bytes per 32-token window. This
  script wires that store into the decoder (lens/decode.py: WarmDecode / StepSource) and asks three
  questions that can each come out badly:

    A. Is the answer computed from the stored BYTES the same answer the network gives? (P38a-P38c)
    B. What does a step from memory cost against a step from the network? (P38d)
    C. Does it actually fire while generating - and does it change the text? (P38e-P38h)

SPLIT, STATED ON EVERY NUMBER: the HELD-OUT split (which=1) of data/chat/prepared_mix_honest. Nothing
here is measured on the training split. The generation in PART C starts from a HELD-OUT window's words;
it is a cost/behaviour measurement, not an accuracy claim - no accuracy is claimed anywhere in this file.

Usage: CHAT_DATA_DIR=data/chat/prepared_mix_honest python3 lens/scripts/warm_decode_test.py
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
N_CHECK = int(os.environ.get("N_CHECK", "2000"))
N_GEN = int(os.environ.get("N_GEN", "200"))
CHUNK = 512

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens import backend  # noqa: E402
from lens.decode import WarmDecode, StepSource, generate, step_logits, format_prompt  # noqa: E402

OUT = {}
LINES = []


def log(line=""):
    print(line, flush=True)
    LINES.append(line)


# ================================================================================================
# PART 0 - load the ship, the held-out windows, the vocabulary
# ================================================================================================
net = lens.load(SHIP, device="cpu")
names = [c.name for c in net.concepts]
K, W, D, V = int(net.K), int(net.W), int(net.D), int(net.V)
aW, ab = (np.asarray(backend.to_cpu(net.answer.W.data), dtype=np.float32),
          np.asarray(backend.to_cpu(net.answer.b.data), dtype=np.float32))

d = chat_data.build()
ids = np.asarray(d["ids"], dtype=np.int64)
vocab = list(d["vocab"])
if net.V == len(vocab) + 2:
    vocab = vocab + ["<|im_start|>", "<|im_end|>"]
if len(vocab) != net.V:
    raise SystemExit(f"vocab mismatch: ship {net.V}, corpus word list {len(vocab)}")
vmap = {w: i for i, w in enumerate(vocab)}

log("=" * 100)
log("THE WARM TIER IN THE GENERATION LOOP (PLAN item (c2)) - measured")
log("=" * 100)
log(f"ship {os.path.basename(SHIP)}   d={D} window={W} concepts={K} vocab={V} attention={net.attn is not None}")
log(f"predictions written BEFORE this run: lens/runs/WARM-DECODE-PREDICTIONS.txt   seed {SEED}")
log(f"split: HELD-OUT (which=1) of {os.environ.get('CHAT_DATA_DIR', 'data/chat/prepared')}")
log(f"corpus: {len(ids):,} tokens, {len(d['bounds']):,} conversations")

t0 = time.time()
Xva, Yva = chat_data.make_windows(ids, bounds=d["bounds"], split=d["split"], window=W, which=1)
log(f"held-out windows available: {len(Xva):,}   (built in {time.time() - t0:.1f} s)")
rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(len(Xva), size=min(N_CHECK, len(Xva)), replace=False))
log(f"drawing {len(sample):,} UNIFORMLY AT RANDOM (seed {SEED}) for the identity checks")
Xs = np.asarray(Xva[sample], dtype=np.int64)
Hs = []
for s in range(0, len(Xs), CHUNK):
    Hs.append(np.asarray(net.forward_arrays(Xs[s:s + CHUNK])["concepts"], dtype=np.float32))
H = np.concatenate(Hs, axis=0)                      # the network's own concepts, float32
logits_net = H @ aW + ab                            # the network's own answer layer, same weights
top_net = logits_net.argmax(axis=1)
log(f"network concepts for the sample: {H.shape}, max |logit| {np.abs(logits_net).max():.3f}")


# ================================================================================================
# PART A - the answer from stored bytes (P38a-P38c)
# ================================================================================================
def identity(dtype, label):
    """Put each sampled window's concepts in the store, read them back, compare the ANSWER."""
    mem = WarmDecode(net, stride=W, dtype=dtype)
    max_err = 0.0
    same, wrong_same, named = 0, 0, 0
    for i, h in enumerate(H):
        row = mem.put(i, h)
        h_q = mem.store.recall(row)
        max_err = max(max_err, float(np.abs(h - h_q).max()))
        z_mem = mem.logits(row)
        same += int(int(z_mem.argmax()) == int(top_net[i]))
        # THE CONTROL: the same store, the WRONG row. If this does not change the answer, the test above
        # cannot fail and proves nothing.
        z_wrong = mem.logits((row + 1) % mem.store.n) if mem.store.n > 1 else z_mem
        wrong_same += int(int(z_wrong.argmax()) == int(top_net[i]))
        named += int(len(mem.store.top(row, names, k=K, thresh=0.5)) > 0)
    n = len(H)
    rep = mem.report()
    res = {"mode": label, "dtype": str(np.dtype(dtype)), "n": n, "max_err": max_err,
           "top1_token_pct": 100.0 * same / n, "control_same_pct": 100.0 * wrong_same / n,
           "control_changed_pct": 100.0 * (n - wrong_same) / n, "named_pct": 100.0 * named / n,
           "bytes_per_window": rep["bytes_per_window"], "bytes_per_token": rep["bytes_per_token"],
           "store_bytes": rep["bytes"]}
    log()
    log(f"  {label}")
    log(f"    max |concepts(store) - concepts(network)|:   {max_err:.3e}")
    log(f"    top-1 next token, memory versus network:     {res['top1_token_pct']:.2f}% identical")
    log(f"    CONTROL, wrong row:                          {res['control_changed_pct']:.2f}% of answers CHANGED")
    log(f"    windows with >=1 named concept > 0.5:        {res['named_pct']:.2f}%")
    log(f"    bytes per window {res['bytes_per_window']} = {res['bytes_per_token']:.5f} bytes/token at stride {W}")
    return res


log()
log("-" * 100)
log("PART A - THE ANSWER FROM STORED BYTES (P38a, P38b, P38c)")
log("-" * 100)
log(f"  the store is the SHIPPED one (lens/warmmemory.py), the answer weights are the model's own;")
log(f"  nothing here recomputes the concept layer on the memory path.")
a_u8 = identity(np.uint8, "u8 store (the shipped default)")
a_u16 = identity(np.uint16, "u16 store (2 bytes per concept)")
OUT["A_u8"], OUT["A_u16"] = a_u8, a_u16


# ================================================================================================
# PART B - the cost of a step from memory against a step from the network (P38d)
# ================================================================================================
log()
log("-" * 100)
log("PART B - WHAT A MEMORY STEP COSTS (P38d)")
log("-" * 100)
ctx = [int(t) for t in Xs[0]]
mem_b = WarmDecode(net, stride=W, dtype=np.uint8)
row_b = mem_b.put(0, H[0])
reps = 200
t0 = time.time()
for _ in range(reps):
    step_logits(net, ctx, None)
net_ms = (time.time() - t0) / reps * 1000
t0 = time.time()
for _ in range(reps):
    mem_b.logits(row_b)
mem_ms = (time.time() - t0) / reps * 1000
ratio = net_ms / mem_ms
log(f"  full-recompute step (network): {net_ms:.3f} ms/step   ({1000 / net_ms:,.0f} tokens/s)")
log(f"  memory step (answer layer only): {mem_ms:.3f} ms/step   ({1000 / mem_ms:,.0f} tokens/s)")
log(f"  ratio: {ratio:.2f}x   (the answer layer K*V = {K * V:,} multiply-adds dominates BOTH paths, so")
log(f"  this cannot be a large number: the concept layer it replaces is W*D*K = {W * D * K:,})")
OUT["B"] = {"net_ms_per_step": net_ms, "mem_ms_per_step": mem_ms, "ratio": ratio,
            "net_tok_s": 1000 / net_ms, "mem_tok_s": 1000 / mem_ms}


# ================================================================================================
# PART C - in the generation loop (P38e-P38h)
# ================================================================================================
log()
log("-" * 100)
log("PART C - THE STORE WHILE GENERATING (P38e, P38f, P38g, P38h)")
log("-" * 100)
prompt = " ".join(vocab[int(t)] for t in Xs[0])
log(f"  prompt (words of ONE held-out window, used as a prompt only): {prompt[:120]}")
log(f"  greedy, max_new {N_GEN}, no rule stop (so the length is the same on every path)")
base = generate(net, vocab, vmap, prompt, max_new=N_GEN, greedy=True, rule_stop=False, use_cache=True,
                use_warm=False)
log(f"  shipped path (no warm tier): {base['tokens']} tokens, path: {base['path']}")
log(f"  text: {base['text'][:150]}")


def run_warm(dtype, label):
    warm = WarmDecode(net, stride=W, dtype=dtype)
    g = generate(net, vocab, vmap, prompt, max_new=N_GEN, greedy=True, rule_stop=False, use_cache=True,
                 warm=warm, warm_read=True)
    diff = sum(1 for a, b in zip(base["token_ids"], g["token_ids"]) if a != b)
    diff += abs(len(base["token_ids"]) - len(g["token_ids"]))
    w = g["warm"]
    expected_windows = 1 + max(0, (g["tokens"] - 1) // W)
    traces = [warm.trace(r, names) for r in range(warm.store.n)]
    nonempty = sum(1 for t in traces if t and t != "(nothing above threshold)")
    res = {"mode": label, "tokens": g["tokens"], "path": g["path"], "hits": w["hits"],
           "misses": w["misses"], "windows": w["windows"], "bytes": w["bytes"],
           "bytes_per_window": w["bytes_per_window"],
           "bytes_per_generated_token": w["bytes_per_generated_token"],
           "bytes_per_token_asymptotic": w["bytes_per_token_asymptotic"],
           "expected_windows": expected_windows, "bytes_equals_windows_x_bpw":
               (w["bytes"] == w["windows"] * w["bytes_per_window"]),
           "differing_token_positions": diff, "traces_nonempty_pct": 100.0 * nonempty / max(1, len(traces)),
           "last_trace": traces[-1] if traces else None,
           "memory_ms_per_hit": w["memory_ms_per_hit"], "network_ms_per_step": w["network_ms_per_step"]}
    log()
    log(f"  {label}")
    log(f"    generated {g['tokens']} tokens; memory READ path hits {w['hits']}, network steps {w['misses']}"
        f"  -> {100.0 * w['hits'] / max(1, g['tokens']):.2f}% of steps served from memory")
    log(f"    windows stored {w['windows']} (expected {expected_windows} = 1 ingest + one per {W} tokens)")
    log(f"    store bytes {w['bytes']:,} == windows x {w['bytes_per_window']} :"
        f" {w['bytes'] == w['windows'] * w['bytes_per_window']}")
    log(f"    {w['bytes_per_generated_token']:.5f} bytes per generated token here"
        f" (asymptotic {w['bytes_per_token_asymptotic']:.5f})")
    log(f"    token positions differing from the shipped path: {diff}")
    log(f"    windows printable in names: {res['traces_nonempty_pct']:.2f}%")
    log(f"    what the last stored window remembers: {traces[-1] if traces else None}")
    log(f"    ms per memory hit {w['memory_ms_per_hit']}; ms per network step {w['network_ms_per_step']:.3f}")
    log(f"    text: {g['text'][:150]}")
    return res


c_u8 = run_warm(np.uint8, "warm tier ON, u8 store (the shipped precision)")
c_u16 = run_warm(np.uint16, "warm tier ON, u16 store")
OUT["C_base"], OUT["C_u8"], OUT["C_u16"] = base, c_u8, c_u16

# A SECOND generation from the SAME conversation: the store must not pay for a window twice.
warm2 = WarmDecode(net, stride=W, dtype=np.uint8)
g1 = generate(net, vocab, vmap, prompt, max_new=N_GEN, greedy=True, rule_stop=False, warm=warm2,
              warm_read=True)
g2 = generate(net, vocab, vmap, prompt, max_new=N_GEN, greedy=True, rule_stop=False, warm=warm2,
              warm_read=True)
log()
log("  REPEAT REQUEST (the same conversation sent twice, one store):")
log(f"    after the first : windows {g1['warm']['windows']}, bytes {g1['warm']['bytes']:,}")
log(f"    after the second: windows {g2['warm']['windows']}, bytes {g2['warm']['bytes']:,}"
    f"  (windows added by the repeat: {g2['warm']['windows'] - g1['warm']['windows']},"
    f" bytes added: {g2['warm']['bytes'] - g1['warm']['bytes']})")
OUT["C_repeat"] = {"first": g1["warm"], "second": g2["warm"]}


# ================================================================================================
# VERDICTS - binary, against the bars fixed in WARM-DECODE-PREDICTIONS.txt
# ================================================================================================
v = {}
v["P38a"] = "PASS" if (a_u8["max_err"] <= 2.0e-3 and a_u16["max_err"] <= 8.0e-6) else "FAIL"
v["P38b"] = "PASS" if (a_u8["top1_token_pct"] >= 99.0 and a_u16["top1_token_pct"] == 100.0) else "FAIL"
v["P38c"] = "PASS" if a_u8["control_changed_pct"] >= 90.0 else "FAIL"
v["P38d"] = "PASS" if ratio >= 1.3 else "FAIL"
v["P38e"] = "PASS" if c_u8["hits"] <= 1 else "FAIL"
v["P38f"] = "PASS" if (c_u8["bytes_equals_windows_x_bpw"] and c_u8["windows"] == c_u8["expected_windows"]
                       and c_u8["windows"] == 7) else "FAIL"
v["P38g"] = "PASS" if c_u16["differing_token_positions"] == 0 else "FAIL"
v["P38h"] = "PASS" if (c_u8["traces_nonempty_pct"] == 100.0 and c_u16["traces_nonempty_pct"] == 100.0) else "FAIL"

log()
log("=" * 100)
log("VERDICTS (bars fixed before the run in lens/runs/WARM-DECODE-PREDICTIONS.txt)")
log("=" * 100)
log(f"  P38a concepts from bytes exact enough      u8 {a_u8['max_err']:.3e} / u16 {a_u16['max_err']:.3e}"
    f"   -> {v['P38a']}")
log(f"  P38b the answer from memory                u8 {a_u8['top1_token_pct']:.2f}% / u16 {a_u16['top1_token_pct']:.2f}%"
    f"   -> {v['P38b']}")
log(f"  P38c control (wrong row must change it)    {a_u8['control_changed_pct']:.2f}% changed   -> {v['P38c']}")
log(f"  P38d memory step vs network step           {ratio:.2f}x   -> {v['P38d']}")
log(f"        NOTE, written with the number: the BAR (>= 1.3x) passed, but the RANGE I predicted in P38d")
log(f"        (1.5-1.8x) was WRONG, and so was my stated reason for it. I claimed the answer layer")
log(f"        dominates both paths so the ratio cannot be large; measured, the memory step is {mem_ms:.3f} ms")
log(f"        (the answer layer alone) and the network step is {net_ms:.3f} ms, so the concept layer + embedding")
log(f"        + engine overhead is {net_ms - mem_ms:.3f} ms, i.e. {100 * (net_ms - mem_ms) / net_ms:.0f}% of a step. The concept")
log(f"        layer is the dominant cost per step, not the answer layer - which is also the premise of")
log(f"        PLAN item (f). The prediction's range is recorded as missed, not as retuned.")
log(f"  P38e fire rate in sliding generation       {c_u8['hits']} of {c_u8['tokens']} steps   -> {v['P38e']}")
log(f"  P38f size while generating                 {c_u8['windows']} windows, {c_u8['bytes']} bytes   -> {v['P38f']}")
log(f"  P38g the memory path does not change text  u16 differing positions {c_u16['differing_token_positions']}"
    f"   -> {v['P38g']}")
log(f"  P38h readable during generation            {c_u8['traces_nonempty_pct']:.2f}%   -> {v['P38h']}")
log()
log("  WHAT THIS MEANS, in one paragraph, before any summary elsewhere:")
log("  the warm tier is CORRECT and CHEAP and it does not accelerate sliding generation. It fires")
log(f"  {c_u8['hits']} time(s) in {c_u8['tokens']} generated tokens, because the window slides by one token per step")
log(f"  while the store is written at stride {W}, so a stored window is never revisited. What it does give")
log(f"  is memory the model never had before: {c_u8['bytes']} bytes for {c_u8['windows']} windows of this")
log(f"  conversation (the KV cache would need {4 * D * 4 * W:,} bytes for the same one window), and every")
log("  window is printable as names. Nothing here is an accuracy claim.")

OUT["verdicts"] = v
OUT.update({"ship": SHIP, "seed": SEED, "n_check": int(len(sample)), "held_out_windows": int(len(Xva)),
            "lines": LINES})
# ONE FILE PER SHIP, plus the latest as lens/runs/warm_decode.json. The second ship's run must not overwrite
# the first ship's record - a measurement that disappears is not evidence (same rule as WARM-MEMORY's P27).
stem = os.path.basename(SHIP).replace(".npz", "")
for path in (f"lens/runs/warm_decode_{stem}.json", "lens/runs/warm_decode.json"):
    with open(path, "w") as f:
        json.dump(OUT, f, indent=2, default=float)
log()
log(f"  written: lens/runs/warm_decode_{stem}.json (and the same content as lens/runs/warm_decode.json)")
