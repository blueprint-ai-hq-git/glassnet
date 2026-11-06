#!/usr/bin/env python3
"""f16_gemv_test.py - PLAN item (f2b) PART 2: the bar. Bytes/token, ms/step, token identity, control.

The bar, as declared in PLAN.txt item (f2b) and in lens/runs/F16-GEMV-PREDICTIONS.txt (P49f-i):
"array identity or token identity on both ships, plus a measured bytes/token and ms/step before and
after" - and speed is part of it, because (f2) already failed on speed alone.

Arms, on BOTH attention ships (the same two ships (f2) used, so the numbers are comparable):
  ref        float32, the shipped fused cached step (lens/kvcache.py) - unchanged
  f16numpy   float16 weights through numpy's own widening matmul: the (f2) arm that lost 4.067x/4.209x
  f16gemv    float16 weights through OUR C KERNEL (lens/f16gemv.c, block-outer register accumulators)
  f16gemv1   the same kernel pinned to ONE thread (the per-call OpenMP team is not free)
  control    the same kernel, same bytes, fed a column-REVERSED answer.W: if this does not change the
             tokens, the identity test is decoration and P49i fails.

This is a decode-path measurement. NO accuracy claim of any kind is made.
Run:  bash lens/scripts/run_f16_gemv.sh      (output: lens/runs/F16-GEMV.txt)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                     # noqa: E402
import sys                      # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                                      # noqa: E402
from lens.kvcache import WindowCache, cached_logits              # noqa: E402
from lens.quant import QuantWindowCache                          # noqa: E402
from lens.fusedgemv import FusedF16WindowCache                   # noqa: E402

PROMPTS = ["hello , how are you ?", "what is 2 + 2 ?", "who are you ?",
           "tell me about the sun .", "what is water made of ?"]
SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest")]
GEN = 40
WARMUP = 30
TIMED = 150
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


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
    if isinstance(cache, QuantWindowCache):
        return np.asarray(cache.logits(), dtype=np.float64)
    return np.asarray(cached_logits(net, cache), dtype=np.float64)


def greedy_run(net, ctx, cache, steps=GEN):
    toks = []
    ctx = list(ctx)
    for _ in range(steps):
        z = read_logits(net, cache)
        nxt = int(np.argmax(z))
        toks.append(nxt)
        ctx = (ctx + [nxt])[-net.W:]
        if cache is not None:
            cache.push(nxt)
    return toks


def timed_arm(net, ctx, cache, n=TIMED, warm=WARMUP):
    cache.seed(ctx)
    z = None
    for _ in range(warm):
        z = read_logits(net, cache)
        cache.push(int(np.argmax(z)))
    reads, pushes = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        z = read_logits(net, cache)
        t1 = time.perf_counter()
        nxt = int(np.argmax(z))
        t2 = time.perf_counter()
        cache.push(nxt)
        t3 = time.perf_counter()
        reads.append((t1 - t0) * 1e6)
        argm = (t2 - t1) * 1e6
        pushes.append((t3 - t2) * 1e6)
    med = lambda a: float(np.median(a))                       # noqa: E731
    return {"read_us": med(reads), "push_us": med(pushes), "step_us": med(reads) + med(pushes)}


def diff_tokens(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def main():
    say("=" * 100)
    say("PLAN ITEM (f2b) PART 2 - OUR OWN FUSED f16 GEMV IN THE REAL DECODE LOOP, BOTH SHIPS")
    say(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}   numpy {np.__version__}   "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}   load {os.getloadavg()}")
    say("predictions (written first): lens/runs/F16-GEMV-PREDICTIONS.txt   (P49f-i are judged here)")
    say("part 1 (the micro-benchmark on the real shapes): lens/runs/F16-GEMV-BENCH.txt")
    say("NOTE: decode-path measurement only. NO accuracy claim of any kind.")
    say("=" * 100)
    out = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "numpy": np.__version__,
           "threads_env": os.environ.get("OPENBLAS_NUM_THREADS"), "ships": {}}
    verdicts = {}

    for ship, corpus in SHIPS:
        path = os.path.join(ROOT, ship)
        net = lens.load(path, device="cpu")
        vocab, vfile = load_vocab(net, corpus)
        vmap = {w: i for i, w in enumerate(vocab)}
        name = os.path.basename(ship)
        say(f"\n--- SHIP {name}  d={net.D} window={net.W} concepts={net.K} V={net.V} vocab={vfile}")
        rec = {"d": int(net.D), "window": int(net.W), "concepts": int(net.K), "V": int(net.V),
               "vocab_file": vfile}
        out["ships"][name] = rec
        ctx = encode(PROMPTS[0], vocab, vmap, net.W)

        def mk(key, seed_ctx=ctx):
            if key == "ref":
                return WindowCache(net, fused=True).seed(seed_ctx)
            if key == "f16numpy":
                return QuantWindowCache(net, arm="f16").seed(seed_ctx)
            if key == "f16gemv":
                return FusedF16WindowCache(net, nthreads=3).seed(seed_ctx)
            if key == "f16gemv1":
                return FusedF16WindowCache(net, nthreads=1).seed(seed_ctx)
            if key == "control":
                return FusedF16WindowCache(net, nthreads=3, control=True).seed(seed_ctx)
            raise ValueError(key)

        # ---------------------------------------------------------------- bytes per token
        q = mk("f16gemv")
        ib, fb = q.weight_bytes_per_token(), q.ref_weight_bytes_per_token()
        say(f"\nBYTES per decoded token, counted from shapes and dtypes (not timed):")
        say(f"    f16 weights {ib:,} B ({ib/1e6:.4f} MB)   float32 {fb:,} B ({fb/1e6:.4f} MB)   "
            f"{fb/max(1,ib):.4f}x fewer")
        rec["bytes_per_token"] = {"f16": ib, "float32": fb, "ratio": fb / max(1, ib)}

        # ---------------------------------------------------------------- P49f speed, the real loop
        say(f"\nP49f SPEED - the real loop (read, then push), median of {TIMED} steps after {WARMUP} "
            f"warmup, greedy:")
        say(f"    {'arm':9s} {'read us':>9s} {'push us':>9s} {'step us':>9s} {'vs ref':>8s}")
        times = {}
        for key in ("ref", "f16numpy", "f16gemv", "f16gemv1", "control"):
            t = timed_arm(net, ctx, mk(key))
            times[key] = t
            say(f"    {key:9s} {t['read_us']:9.1f} {t['push_us']:9.1f} {t['step_us']:9.1f} "
                f"{t['step_us']/max(1e-9, times['ref']['step_us']):8.3f}")
        rec["times"] = times
        ratio = times["f16gemv"]["step_us"] / max(1e-9, times["ref"]["step_us"])
        rec["f16gemv_ratio_vs_ref"] = ratio
        say(f"    P49f: f16gemv/ref = {ratio:.3f}   (bar 0.55-0.90, i.e. 1.1-1.8x faster)  -> "
            f"{'PASS' if 0.55 <= ratio <= 0.90 else 'FAIL'}")
        say(f"    (the same ratio for the numpy f16 arm (f2) and the single-thread kernel: "
            f"{times['f16numpy']['step_us']/max(1e-9, times['ref']['step_us']):.3f} / "
            f"{times['f16gemv1']['step_us']/max(1e-9, times['ref']['step_us']):.3f})")

        # ---------------------------------------------------------------- P49g identity + P49i control
        say(f"\nP49g/P49i TOKEN IDENTITY - greedy, {GEN} tokens, {len(PROMPTS)} prompts, each path "
            f"deciding its own token:")
        ident = {}
        for key in ("f16numpy", "f16gemv", "f16gemv1", "control"):
            diffs, runs_diff, first = 0, 0, None
            for prompt in PROMPTS:
                c = encode(prompt, vocab, vmap, net.W)
                ref_toks = greedy_run(net, c, None)
                q_toks = greedy_run(net, c, mk(key, c))
                d = diff_tokens(ref_toks, q_toks)
                diffs += d
                if d:
                    runs_diff += 1
                    if first is None:
                        k = next(i for i in range(min(len(ref_toks), len(q_toks)))
                                 if ref_toks[i] != q_toks[i])
                        first = (prompt, k, ref_toks[:8], q_toks[:8])
            say(f"    {key:9s} differing tokens {diffs:4d} of {len(PROMPTS)*GEN:4d}   "
                f"runs differing {runs_diff}/{len(PROMPTS)}")
            ident[key] = {"differing_tokens": diffs, "total_tokens": len(PROMPTS) * GEN,
                          "runs_differing": runs_diff, "runs": len(PROMPTS), "first_flip": first}
        rec["identity"] = ident
        say(f"    P49g: f16gemv {ident['f16gemv']['differing_tokens']} differing tokens, "
            f"{ident['f16gemv']['runs_differing']}/{len(PROMPTS)} runs  (bar 0 / 0)  -> "
            f"{'PASS' if ident['f16gemv']['differing_tokens'] == 0 and ident['f16gemv']['runs_differing'] == 0 else 'FAIL'}")
        say(f"    P49i: control {ident['control']['differing_tokens']} differing tokens, "
            f"{ident['control']['runs_differing']}/{len(PROMPTS)} runs  (bar >=50% tokens and 5/5 runs)  -> "
            f"{'PASS' if ident['control']['runs_differing'] == len(PROMPTS) and ident['control']['differing_tokens'] >= len(PROMPTS)*GEN/2 else 'FAIL'}")

        # ---------------------------------------------------------------- P49h logit-level walk
        say(f"\nP49h LOGIT-LEVEL (info + the bar): max |dz| over a {GEN}-step greedy walk")
        walks = {}
        for key in ("f16numpy", "f16gemv", "f16gemv1"):
            rc = WindowCache(net, fused=True).seed(ctx)
            qc = mk(key)
            worst = 0.0
            for _ in range(GEN):
                zr = np.asarray(cached_logits(net, rc), dtype=np.float64)
                zq = np.asarray(qc.logits(), dtype=np.float64)
                worst = max(worst, float(np.abs(zr - zq).max()))
                nxt = int(np.argmax(zr))
                rc.push(nxt)
                qc.push(nxt)
            walks[key] = worst
            say(f"    {key:9s} max |dz| {worst:.6e}  -> {'PASS' if worst <= 1e-4 else 'FAIL'} (bar 1e-4)")
        rec["max_logit_diff"] = walks

    # ---------------------------------------------------------------- the item's verdict, values beside it
    say("\n" + "=" * 100)
    say("THE BAR, AS DECLARED, WITH THE COMPARED VALUES BESIDE IT")
    for name, rec in out["ships"].items():
        r = rec["f16gemv_ratio_vs_ref"]
        say(f"  {name}: bytes {rec['bytes_per_token']['ratio']:.4f}x fewer | "
            f"step {r:.3f}x ref (bar 0.55-0.90) | identity "
            f"{rec['identity']['f16gemv']['differing_tokens']} differing tokens | "
            f"max|dz| {rec['max_logit_diff']['f16gemv']:.3e} | control "
            f"{rec['identity']['control']['differing_tokens']} differing tokens")
    say("=" * 100)
    # The .txt is APPENDED to by lens/scripts/run_f16_gemv.sh (a dated block per run, never overwritten);
    # this script owns only the json.
    with open(os.path.join(ROOT, "lens/runs/f16_gemv.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote lens/runs/f16_gemv.json (the .txt block is appended by run_f16_gemv.sh)")


if __name__ == "__main__":
    main()
