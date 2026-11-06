#!/usr/bin/env python3
"""int8_decode_test.py - PLAN item (f2): do int8 weights make the decode step cheaper, and do they
change the answer?

The step was measured to be BANDWIDTH-bound on the weight matrices (4.14 MB/token at d=256, 9.16 MB at
d=512, at 10-12 GB/s - lens/runs/DECODE-STEP-COST.txt). So the lever is FEWER BYTES. This test measures
all four arms side by side against the shipped float32 step:

  ref      float32, the shipped fused cached step (lens/kvcache.py)
  dequant  whole-matrix int8 dequantise per step, then one matmul
  chunk    dequantise in ~256 KB blocks into ONE reused buffer (the arm predicted to win)
  int8acc  activation quantised too, integer accumulate
  broken   THE CONTROL: same path as `dequant`, every scale forced to 1.0 (weights saturate)

Bars, written BEFORE the run in lens/runs/INT8-DECODE-PREDICTIONS.txt (P47a-h): bytes 4.0x fewer (P47a),
max |dequant-W| <= max|W|/254 (P47b), dequant slower 2-6x (P47c), int8acc slower 5-15x (P47d), chunk
FASTER by 1.1-2.0x at d=256 (P47e), NOT token-identical, 0.5-8% of runs differing (P47f), flips at
near-ties (P47g), the control differs on >= 50% of runs (P47h).

Run:  bash lens/scripts/run_int8_decode.sh      (output: lens/runs/INT8-DECODE.txt)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                     # noqa: E402
import sys                      # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# Do NOT add ROOT + "/lens" here: that put the top-level name `cuda` on our own lens/cuda/ package and
# broke cupy (BUG-CUDA-NAME-SHADOW-CLASS3, lens/runs/SYSPATH-SHADOW-FIX-class3.txt).
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                                      # noqa: E402
from lens.kvcache import WindowCache, cached_logits              # noqa: E402
from lens.quant import QuantWindowCache                          # noqa: E402

PROMPTS = ["hello , how are you ?", "what is 2 + 2 ?", "who are you ?",
           "tell me about the sun .", "what is water made of ?"]
SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest")]
GEN = 40            # tokens per generation
WARMUP = 30
TIMED = 150
ARMS = ("dequant", "chunk", "int8acc", "broken", "f16", "f16chunk")


def load_vocab(net, corpus):
    p = os.path.join(ROOT, corpus, "vocab_v12000.json")
    vocab = json.load(open(p))
    if net.V == len(vocab) + 2:
        vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    if net.V != len(vocab):
        raise SystemExit(f"vocab mismatch: ship has {net.V}, the word list at {p} has {len(vocab)}")
    return vocab, p


def encode(prompt, vocab, vmap, W):
    import re
    enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", prompt.lower())]
    if len(enc) < W:
        enc = [0] * (W - len(enc)) + enc
    return enc[-W:]


def read_logits(net, cache):
    """One step's logits on whichever path the cache is (this harness always supplies a cache)."""
    if isinstance(cache, QuantWindowCache):
        return np.asarray(cache.logits(), dtype=np.float64)
    return np.asarray(cached_logits(net, cache), dtype=np.float64)


def greedy_run(net, ctx, cache, steps=GEN, gaps=False):
    """GREEDY generation, each path deciding its own token (a single argmax flip is then visible).

    Returns (tokens, gaps) where gaps[k] = top-1 logit minus top-2 logit at step k (the reference's own
    margin, used to ask whether a flip happened where the model's decision was a near-tie).
    """
    toks, gs = [], []
    ctx = list(ctx)
    for _ in range(steps):
        if cache is None:
            z = np.asarray(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0],
                           dtype=np.float64)
        elif isinstance(cache, QuantWindowCache):
            z = np.asarray(cache.logits(), dtype=np.float64)
        else:
            z = np.asarray(cached_logits(net, cache), dtype=np.float64)
        if gaps:
            top = np.partition(z, -2)[-2:]
            gs.append(float(top.max() - top.min()))
        nxt = int(np.argmax(z))
        toks.append(nxt)
        ctx = (ctx + [nxt])[-net.W:]
        if cache is not None:
            cache.push(nxt)
    return toks, gs


def timed_arm(net, ctx, cache, n=TIMED, warm=WARMUP):
    """Per-step cost of the real loop: read, then push. Median over n steps, after `warm` steps."""
    cache.seed(ctx)
    z = None
    for _ in range(warm):
        z = read_logits(net, cache)
        cache.push(int(np.argmax(z)))
    if hasattr(cache, "n_matmuls"):
        cache.n_matmuls = 0                     # count only the timed loop
    reads, pushes, argmaxes = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        z = read_logits(net, cache)
        t1 = time.perf_counter()
        nxt = int(np.argmax(z))
        t2 = time.perf_counter()
        cache.push(nxt)
        t3 = time.perf_counter()
        reads.append((t1 - t0) * 1e6)
        argmaxes.append((t2 - t1) * 1e6)
        pushes.append((t3 - t2) * 1e6)
    med = lambda a: float(np.median(a))                       # noqa: E731
    return {"read_us": med(reads), "push_us": med(pushes), "argmax_us": med(argmaxes),
            "step_us": med(reads) + med(pushes), "n": n}


def diff_tokens(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def main():
    print("=" * 100)
    print("INT8 WEIGHTS FOR THE DECODE PATH (PLAN item (f2)) - the measured lever is FEWER BYTES")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}   numpy {np.__version__}   "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print("predictions (written first): lens/runs/INT8-DECODE-PREDICTIONS.txt   (P47a-h)")
    print("NOTE: this is a decode-path measurement. NO accuracy claim of any kind is made here.")
    print("=" * 100)
    out = {"pass": time.strftime("%Y-%m-%d %H:%M:%S"), "numpy": np.__version__,
           "threads": os.environ.get("OPENBLAS_NUM_THREADS"), "ships": {}}

    for ship, corpus in SHIPS:
        path = os.path.join(ROOT, ship)
        net = lens.load(path, device="cpu")
        vocab, vfile = load_vocab(net, corpus)
        vmap = {w: i for i, w in enumerate(vocab)}
        name = os.path.basename(ship)
        print(f"\n--- SHIP {name}  d={net.D} window={net.W} concepts={net.K} V={net.V} vocab={vfile}")
        rec = {"d": int(net.D), "window": int(net.W), "concepts": int(net.K), "V": int(net.V),
               "vocab_file": vfile, "arms": {}}
        out["ships"][name] = rec

        # ---------------------------------------------------------------- the cache arms
        caches = {"ref": WindowCache(net, fused=True).seed(
            encode(PROMPTS[0], vocab, vmap, net.W))}
        for arm in ARMS:
            caches[arm] = QuantWindowCache(net, arm=arm).seed(
                encode(PROMPTS[0], vocab, vmap, net.W))

        # ---------------------------------------------------------------- P47a/P47b bytes + error
        q = caches["chunk"]
        ib, fb = q.weight_bytes_per_token(), q.ref_weight_bytes_per_token()
        print(f"\nP47a BYTES per decoded token (counted from shapes and dtypes, not timed):")
        print(f"  int8 weights: {ib:,} B ({ib/1e6:.4f} MB)   float32: {fb:,} B ({fb/1e6:.4f} MB)"
              f"   ratio {fb/max(1,ib):.4f}x fewer")
        per = q.quant_error()
        worst_rel = max(v["rel_to_max_weight"] for v in per.values())
        print(f"P47b QUANTISATION ERROR (per-output-channel symmetric int8):")
        for k in ("Wq", "Wk", "Wv", "Wo", "cW", "aW"):
            v = per[k]
            print(f"    {k:3s} {str(v['shape']):14s} int8 {v['int8_bytes']:>8,} B "
                  f"max|dq-W| {v['max_abs_error']:.6e}  ({v['rel_to_max_weight']*100:.4f}% of max|W|)")
        print(f"  worst relative error {worst_rel*100:.4f}%  (bar: <= 1/254 = 0.3937%)")
        rec["bytes_per_token"] = {"int8": ib, "float32": fb, "ratio": fb / max(1, ib)}
        rec["quant_error"] = per

        # THE SECOND BYTE WIDTH (addendum predictions P48a-e): float16, a 2x cut whose widening is a CAST.
        qf = caches["f16chunk"]
        ibf, fbf = qf.weight_bytes_per_token(), qf.ref_weight_bytes_per_token()
        perf = qf.quant_error()
        worstf = max(v["rel_to_max_weight"] for v in perf.values())
        print(f"\nP48a BYTES, float16 (the second width, predictions in "
              f"lens/runs/INT8-DECODE-PREDICTIONS-F16-ADDENDUM.txt):")
        print(f"  f16 weights: {ibf:,} B ({ibf/1e6:.4f} MB)   float32: {fbf:,} B ({fbf/1e6:.4f} MB)"
              f"   ratio {fbf/max(1,ibf):.4f}x fewer")
        for k in ("Wq", "Wk", "Wv", "Wo", "cW", "aW"):
            v = perf[k]
            print(f"    {k:3s} {str(v['shape']):14s} f16 {v['int8_bytes']:>8,} B "
                  f"max|cast-W| {v['max_abs_error']:.6e}  ({v['rel_to_max_weight']*100:.4f}% of max|W|)")
        print(f"  worst relative error {worstf*100:.4f}%  (int8's was {worst_rel*100:.4f}%)")
        rec["bytes_per_token_f16"] = {"f16": ibf, "float32": fbf, "ratio": fbf / max(1, ibf)}
        rec["quant_error_f16"] = perf

        # ---------------------------------------------------------------- P47c/d/e timing
        print(f"\nP47c/d/e SPEED - the real loop, read then push, median of {TIMED} steps "
              f"({WARMUP} warmup), greedy tokens:")
        print(f"    {'arm':8s} {'read us':>9s} {'push us':>9s} {'step us':>9s} {'argmax us':>10s} "
              f"{'vs ref':>8s} {'matmuls/step':>12s}")
        times = {}
        for key in ("ref",) + ARMS:
            t = timed_arm(net, encode(PROMPTS[0], vocab, vmap, net.W), caches[key])
            times[key] = t
            mm = caches[key].n_matmuls / max(1, t["n"]) if key != "ref" else 5
            print(f"    {key:8s} {t['read_us']:9.1f} {t['push_us']:9.1f} {t['step_us']:9.1f} "
                  f"{t['argmax_us']:10.1f} {t['step_us']/max(1e-9, times['ref']['step_us']):8.3f} "
                  f"{mm:12.1f}")
        rec["times"] = times
        ref_us = times["ref"]["step_us"]
        rec["speed_ratio_vs_ref"] = {k: times[k]["step_us"] / max(1e-9, ref_us) for k in ARMS}
        rec["implied_if_bytes_only"] = {"note": "EXTRAPOLATION, not measured: ref ms/step x int8/float32 "
                                                "bytes", "ratio": ib / float(fb)}
        print(f"    reference step {ref_us:.1f} us; if bytes were the only cost the int8 arm would be "
              f"{ib/fb:.3f}x of that = {ref_us*ib/fb:.1f} us (EXTRAPOLATION, not a measurement)")

        # ---------------------------------------------------------------- P47f/g identity
        print(f"\nP47f/g TOKEN IDENTITY - greedy, {GEN} tokens, {len(PROMPTS)} prompts, "
              f"each path deciding its own token:")
        ident = {}
        for key in ARMS:
            diffs, runs_diff, first = 0, 0, None
            flip_gaps, ref_gaps_all = [], []
            for prompt in PROMPTS:
                ctx = encode(prompt, vocab, vmap, net.W)
                ref_toks, ref_gaps = greedy_run(net, ctx, None, gaps=True)
                qc = QuantWindowCache(net, arm=key).seed(ctx)
                q_toks, _ = greedy_run(net, ctx, qc, gaps=False)
                d = diff_tokens(ref_toks, q_toks)
                diffs += d
                if d:
                    runs_diff += 1
                    if first is None:
                        k = next(i for i in range(min(len(ref_toks), len(q_toks)))
                                 if ref_toks[i] != q_toks[i])
                        first = (prompt, k, ref_toks[:8], q_toks[:8])
                    # the reference's own margin at every position where the paths differ
                    for i in range(min(len(ref_toks), len(q_toks))):
                        if ref_toks[i] != q_toks[i]:
                            flip_gaps.append(ref_gaps[i])
                ref_gaps_all.extend(ref_gaps)
            p10 = float(np.percentile(ref_gaps_all, 10)) if ref_gaps_all else float("nan")
            med_flip = float(np.median(flip_gaps)) if flip_gaps else float("nan")
            print(f"    {key:8s} differing tokens {diffs:4d} of {len(PROMPTS)*GEN:4d}   "
                  f"runs differing {runs_diff}/{len(PROMPTS)}   median margin at a flip "
                  f"{med_flip:.4f} vs p10 of all margins {p10:.4f}")
            ident[key] = {"differing_tokens": diffs, "total_tokens": len(PROMPTS) * GEN,
                          "runs_differing": runs_diff, "runs": len(PROMPTS),
                          "median_margin_at_flip": med_flip, "p10_margin_all": p10,
                          "first_flip": first}
        rec["identity"] = ident

        # ---------------------------------------------------------------- P47g logit-level
        print(f"\nLOGIT-LEVEL (info): max |quantised - float32| logits over a {GEN}-step greedy walk")
        ctx = encode(PROMPTS[0], vocab, vmap, net.W)
        walks = {}
        for key in ARMS:
            qc = QuantWindowCache(net, arm=key).seed(ctx)
            rc = WindowCache(net, fused=True).seed(ctx)
            worst = 0.0
            for _ in range(GEN):
                zr = np.asarray(cached_logits(net, rc), dtype=np.float64)
                zq = np.asarray(qc.logits(), dtype=np.float64)
                worst = max(worst, float(np.abs(zr - zq).max()))
                nxt = int(np.argmax(zr))
                rc.push(nxt)
                qc.push(nxt)
            walks[key] = worst
            print(f"    {key:8s} max |dz| {worst:.6e}")
        rec["max_logit_diff"] = walks
        # internal consistency: `chunk` and `dequant` share the same int8 weights, so any difference
        # between them is float32 summation ORDER (chunk accumulates per input block), never a formula.
        qa = QuantWindowCache(net, arm="dequant").seed(ctx)
        qb = QuantWindowCache(net, arm="chunk").seed(ctx)
        worst_ab = 0.0
        for _ in range(GEN):
            za = np.asarray(qa.logits(), dtype=np.float64)
            zb = np.asarray(qb.logits(), dtype=np.float64)
            worst_ab = max(worst_ab, float(np.abs(za - zb).max()))
            nxt = int(np.argmax(za))
            qa.push(nxt)
            qb.push(nxt)
        print(f"    chunk vs dequant (same int8 weights, different summation order): max |dz| "
              f"{worst_ab:.6e}  -> chunking changes the ORDER, not the formula")
        rec["chunk_vs_dequant_max_logit_diff"] = worst_ab
        # the same check for the f16 pair (P48e)
        qc16 = QuantWindowCache(net, arm="f16").seed(ctx)
        qd16 = QuantWindowCache(net, arm="f16chunk").seed(ctx)
        worst_f = 0.0
        for _ in range(GEN):
            zc = np.asarray(qc16.logits(), dtype=np.float64)
            zd = np.asarray(qd16.logits(), dtype=np.float64)
            worst_f = max(worst_f, float(np.abs(zc - zd).max()))
            nxt = int(np.argmax(zc))
            qc16.push(nxt)
            qd16.push(nxt)
        print(f"    f16chunk vs f16 (same f16 weights, different summation order): max |dz| "
              f"{worst_f:.6e}")
        rec["f16chunk_vs_f16_max_logit_diff"] = worst_f

    print("\n" + "=" * 100)
    print("VERDICTS ARE WRITTEN FROM THESE NUMBERS IN AUTOPILOT-LOG.md AND lens/runs/PLAN.txt.")
    print("=" * 100)
    with open(os.path.join(ROOT, "lens/runs/int8_decode.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"json: lens/runs/int8_decode.json")


if __name__ == "__main__":
    main()
