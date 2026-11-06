#!/usr/bin/env python3
"""kv_decode_test.py - is the KV cache a real decoder, or decoration?

PLAN item (e): wire the preallocated KV cache into lens/decode.py and the server. The cache existed and was
correct against the model's own attention block, but nothing USED it while generating. This test decides
whether the wiring counts as done, against bars written BEFORE the run in
lens/runs/KV-DECODE-PREDICTIONS.txt (P30a..P30g).

What it measures, and why each part is needed:
  P30a  the cached path and the current full-recompute path produce the SAME TOKEN IDS, exactly, on real
        generations. "Close enough" is a fail: a cache that changes the answer is a bug, not a speed-up.
  P30b  the logit-level difference, so a near-miss is visible before it becomes a token flip.
  P30c  seconds per generated token, both paths, same ship/prompt/seed/threads.
  P30d  bytes kept per token and for the whole window (the hot tier must be FIXED size, not growing).
  P30e  THE CONTROL: a deliberately WRONG cache (the classic unbounded KV cache, attending over the whole
        history instead of the window the model was trained on) must produce DIFFERENT tokens. If the wrong
        cache passes P30a, then P30a proves nothing and this gate fails as decoration.
  P30f  the server (lens/serve.py) returns exactly the same text as the measured decoder, and reports its
        own tokens/second and cache size.
  P30g  a no-attention ship takes the full path AND SAYS SO by name (no silent path change).

Run:  bash lens/scripts/run_kv_decode.sh      (output: lens/runs/KV-DECODE.txt)
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                     # noqa: E402
import sys                      # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# REMOVED 2026-10-20, THE SAME REAL BUG IN A FORM THE FIRST GATE COULD NOT SEE (class 3;
# evidence lens/runs/SYSPATH-SHADOW-FIX-class3.txt, probe lens/runs/GPU-ATTN-CACHE.txt): this
# tuple put <project>/lens on sys.path as well, so the top-level name `cuda` resolved to our own
# lens/cuda/ package and cupy 14.2 died with "ModuleNotFoundError: No module named
# 'cuda.pathfinder'". lens/ is imported as `lens.X` via the ROOT entry and lens/scripts/ has its
# own entry, so the `/lens` element is not needed. Do not add it back.
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                             # noqa: E402
from lens.decode import generate, format_prompt, step_logits  # noqa: E402

PROMPTS = ["hello , how are you ?", "what is 2 + 2 ?", "who are you ?",
           "tell me about the sun .", "what is water made of ?"]
SEEDS = [0, 1, 2]
MAXNEW = 40
# (ship, the corpus it was trained on - the vocabulary must come from THAT corpus, the files differ)
SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest")]
NOATTN = ("lens/runs/chat/cl_d256_clean_24_ep24.npz", "data/chat/prepared_mix_ctrl_honest")


def load_vocab(net, corpus):
    """The word list of the corpus this ship was trained on, named by file."""
    p = os.path.join(ROOT, corpus, "vocab_v12000.json")
    vocab = json.load(open(p))
    if net.V == len(vocab) + 2:
        vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    if net.V != len(vocab):
        raise SystemExit(f"vocab mismatch: ship has {net.V}, the word list at {p} has {len(vocab)}")
    return vocab, p


class UnboundedCache:
    """THE CONTROL, and it is DELIBERATELY WRONG for this model.

    This is the classic KV cache everyone else uses: attention over EVERY token of the conversation. The
    model was trained on a fixed 32-token window, so once the conversation passes the window this attends
    over positions the model never saw in that configuration. It is the most tempting wrong cache - it is
    what "unlimited context" sounds like - and P30a must reject it. If it does not, P30a is decoration.
    """

    def __init__(self, net):
        attn = net.attn
        self.D, self.W = int(net.D), int(net.W)
        self.emb = net.emb.data
        self.Wq, self.Wk, self.Wv, self.Wo = attn.Wq.data, attn.Wk.data, attn.Wv.data, attn.Wo.data
        self.cW, self.cb = net.concept.W.data, net.concept.b.data
        self.aW, self.ab = net.answer.W.data, net.answer.b.data
        self.Q = np.zeros((0, self.D), np.float32)
        self.K = np.zeros((0, self.D), np.float32)
        self.V = np.zeros((0, self.D), np.float32)
        self.Vo = np.zeros((0, self.D), np.float32)
        self.ids = np.zeros(0, np.int64)

    def seed(self, ids):
        x = self.emb[np.asarray(ids, dtype=np.int64)]
        self.Q, self.K = x @ self.Wq, x @ self.Wk
        self.V = x @ self.Wv
        self.Vo = self.V @ self.Wo
        self.ids = np.asarray(ids, dtype=np.int64).copy()
        return self

    def push(self, token):
        x = self.emb[int(token)]
        self.Q = np.vstack([self.Q, (x @ self.Wq)[None, :]])
        self.K = np.vstack([self.K, (x @ self.Wk)[None, :]])
        self.V = np.vstack([self.V, (x @ self.Wv)[None, :]])
        self.Vo = np.vstack([self.Vo, (x @ self.Wo)[None, :]])
        self.ids = np.append(self.ids, int(token))
        return self

    def outputs(self):
        n = self.Q.shape[0]
        s = self.Q @ self.K.T
        s = s + np.triu(np.full((n, n), -1e9, np.float32), k=1)   # causal over the WHOLE history
        s = s - s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(axis=1, keepdims=True)
        return (p @ self.Vo)[-self.W:]                            # the concept layer reads the last W

    def bytes_per_token(self):
        return 4 * self.D * 4

    def total_bytes(self):
        return self.bytes_per_token() * int(self.ids.size)


def token_diff(a, b):
    """How many tokens differ, counting a length difference as a difference."""
    n = min(len(a), len(b))
    d = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    return d


def logit_walk(net, ctx, steps=MAXNEW, fold=True):
    """max |cached logits - full-recompute logits| along one generation, token for token."""
    from lens.kvcache import WindowCache
    cache = WindowCache(net, fold=fold).seed(ctx)
    ctx = list(ctx)
    worst, worst_step = 0.0, -1
    for s in range(steps):
        z_ref = step_logits(net, ctx, None)
        z_c = step_logits(net, ctx, cache)
        d = float(np.abs(z_ref - z_c).max())
        if d > worst:
            worst, worst_step = d, s
        nxt = int(np.argmax(z_ref))                # the reference decides the path both follow
        ctx = (ctx + [nxt])[-net.W:]
        cache.push(nxt)
    return worst, worst_step


def timed(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.time()
        r = fn()
        ts.append(time.time() - t0)
    return sorted(ts)[len(ts) // 2], r           # median seconds, last result


def main():
    print("=" * 100)
    print("KV CACHE WIRED INTO DECODING - does it decode IDENTICALLY and FASTER?")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}  (this file is APPENDED to: earlier blocks are")
    print("earlier passes, kept and never overwritten)")
    print("predictions (written first): lens/runs/KV-DECODE-PREDICTIONS.txt")
    print("=" * 100)
    summary = {"ships": {}}
    verdicts = {}

    for ship, corpus in SHIPS:
        path = os.path.join(ROOT, ship)
        net = lens.load(path, device="cpu")
        vocab, vfile = load_vocab(net, corpus)
        vmap = {w: i for i, w in enumerate(vocab)}
        name = os.path.basename(ship)
        print(f"\n--- SHIP {name}  d={net.D} window={net.W} concepts={net.K} V={net.V} "
              f"attention=yes  vocab={vfile}")
        summary["ships"][name] = {"d": net.D, "window": net.W, "concepts": net.K, "V": net.V,
                                  "vocab_file": vfile}

        # ---------------------------------------------------------------- P30a identity
        runs = flips_fold = flips_unfold = 0
        first_flip = None
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            for seed in SEEDS:
                ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=False)
                cac = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=True, fold=True)
                d = token_diff(ref["token_ids"], cac["token_ids"])
                runs += 1
                if d:
                    flips_fold += 1
                    if first_flip is None:
                        first_flip = (prompt, seed, d, ref["token_ids"][:12], cac["token_ids"][:12])
                if seed == 0:
                    unf = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                                   use_cache=True, fold=False)
                    if token_diff(ref["token_ids"], unf["token_ids"]):
                        flips_unfold += 1
        # the strictest case: greedy, where a single argmax flip shows up as a divergence
        greedy_flips = 0
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            g_ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True, rule_stop=False,
                             use_cache=False)
            g_cac = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True, rule_stop=False,
                             use_cache=True)
            greedy_flips += token_diff(g_ref["token_ids"], g_cac["token_ids"])
        print(f"P30a token-for-token identity: {runs - flips_fold}/{runs} runs identical (fold ON), "
              f"{len(PROMPTS) - flips_unfold}/{len(PROMPTS)} identical (fold OFF, seed 0), "
              f"greedy token differences: {greedy_flips}")
        if first_flip:
            print(f"     FIRST FLIP: prompt={first_flip[0]!r} seed={first_flip[1]} differing={first_flip[2]}")
            print(f"       full  : {first_flip[3]}")
            print(f"       cached: {first_flip[4]}")
        verdicts[f"P30a {name}"] = (flips_fold == 0 and greedy_flips == 0, f"{runs - flips_fold}/{runs} runs")

        # ---------------------------------------------------------------- P30b logit agreement
        prompt = PROMPTS[3]
        ctx = format_prompt(prompt, vocab)
        import re
        enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", ctx.lower())]
        if len(enc) < net.W:
            enc = [0] * (net.W - len(enc)) + enc
        w_fold, s_fold = logit_walk(net, enc[-net.W:], fold=True)
        w_unf, s_unf = logit_walk(net, enc[-net.W:], fold=False)
        print(f"P30b max |cached - full| logits over {MAXNEW} steps: fold ON {w_fold:.3e} (step {s_fold}), "
              f"fold OFF {w_unf:.3e} (step {s_unf})")
        verdicts[f"P30b {name}"] = (w_fold < 1e-4 and w_unf < 1e-4, f"fold {w_fold:.2e} / unfolded {w_unf:.2e}")

        # ---------------------------------------------------------------- P30c speed
        prompt = format_prompt(PROMPTS[3], vocab)
        generate(net, vocab, vmap, prompt, max_new=8, seed=0, rule_stop=False, use_cache=True)   # warm up
        sec_full, g_full = timed(lambda: generate(net, vocab, vmap, prompt, max_new=60, seed=7,
                                                  rule_stop=False, use_cache=False))
        sec_cac, g_cac = timed(lambda: generate(net, vocab, vmap, prompt, max_new=60, seed=7,
                                                rule_stop=False, use_cache=True))
        n_tok = g_full["tokens"]
        ms_full, ms_cac = sec_full / n_tok * 1000, sec_cac / n_tok * 1000
        # a clean per-step measurement with no sampling in the loop
        rng = np.random.default_rng(0)
        stream = list(rng.integers(1, net.V - 2, size=400))
        ctx0 = stream[:net.W]
        def step_loop(cached):
            c = None
            if cached:
                from lens.kvcache import WindowCache
                c = WindowCache(net).seed(ctx0)
            cx = list(ctx0)
            for t in stream[net.W:]:
                step_logits(net, cx, c)
                cx = (cx + [t])[-net.W:]
                if c is not None:
                    c.push(t)
        t0 = time.time()
        step_loop(False)
        step_full = (time.time() - t0) / (len(stream) - net.W)
        t0 = time.time()
        step_loop(True)
        step_cac = (time.time() - t0) / (len(stream) - net.W)
        print(f"P30c speed: generation of {n_tok} tokens -> full {ms_full:.3f} ms/token "
              f"({1000/ms_full:.0f} tok/s), cached {ms_cac:.3f} ms/token ({1000/ms_cac:.0f} tok/s), "
              f"{ms_full/ms_cac:.2f}x faster")
        print(f"     per-step only (no sampling, 368 steps): full {step_full*1000:.3f} ms, "
              f"cached {step_cac*1000:.3f} ms, {step_full/step_cac:.2f}x")
        verdicts[f"P30c {name}"] = (ms_cac < ms_full, f"{ms_full/ms_cac:.2f}x")

        # ---------------------------------------------------------------- P30h where the rest of the time goes
        # The sampler is the same call in both paths, so it dilutes the end-to-end ratio. Measure it.
        from lens.decode import _softmax_topk
        z0 = step_logits(net, ctx0, None)
        p0 = _softmax_topk(z0, 0.8, 40)
        reps = 200
        t0 = time.time()
        for i in range(reps):
            np.random.default_rng(i).choice(net.V, p=p0)     # exactly what generate() does per token
        samp_ms = (time.time() - t0) / reps * 1000
        t0 = time.time()
        for _ in range(reps):
            _softmax_topk(z0, 0.8, 40)                       # the rest of the per-token non-model work
        soft_ms = (time.time() - t0) / reps * 1000
        print(f"P30h where the per-token time goes (d={net.D}): model step full {step_full*1000:.3f} ms / "
              f"cached {step_cac*1000:.3f} ms = {step_full/step_cac:.2f}x; sampler {samp_ms:.3f} ms; "
              f"softmax+top-k over {net.V} logits {soft_ms:.3f} ms; end to end full {ms_full:.3f} ms / "
              f"cached {ms_cac:.3f} ms = {ms_full/ms_cac:.2f}x")
        summary["ships"][name]["sampler_ms"] = samp_ms
        summary["ships"][name]["softmax_topk_ms"] = soft_ms

        # ---------------------------------------------------------------- P30d size
        from lens.kvcache import WindowCache
        c = WindowCache(net)
        print(f"P30d size: {c.summary()}  | check 4*D*4={4*net.D*4} bytes/token, "
              f"16*W*D={16*net.W*net.D} bytes total -> "
              f"{c.bytes_per_token() == 4*net.D*4 and c.total_bytes() == 16*net.W*net.D}")
        verdicts[f"P30d {name}"] = (c.bytes_per_token() == 4 * net.D * 4, f"{c.bytes_per_token()} B/token")

        # ---------------------------------------------------------------- P30e the control
        ctrl_diff = 0
        ctrl_runs = 0
        example = None
        for prompt in PROMPTS[:3]:
            fmt = format_prompt(prompt, vocab)
            ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, rule_stop=False,
                           use_cache=False)
            bad = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, rule_stop=False,
                           use_cache=True, cache_factory=lambda n, c: UnboundedCache(n).seed(c))
            d = token_diff(ref["token_ids"], bad["token_ids"])
            ctrl_runs += 1
            if d:
                ctrl_diff += 1
                if example is None:
                    example = (prompt, d, ref["token_ids"][:12], bad["token_ids"][:12])
        print(f"P30e control (deliberately WRONG unbounded cache): {ctrl_diff}/{ctrl_runs} runs differ "
              f"from the reference")
        if example:
            print(f"     e.g. prompt={example[0]!r} differing={example[1]}")
            print(f"       reference: {example[2]}")
            print(f"       wrong cache: {example[3]}")
        verdicts[f"P30e {name}"] = (ctrl_diff > 0, f"{ctrl_diff}/{ctrl_runs} runs differ")

        summary["ships"][name].update(
            {"runs_identical": runs - flips_fold, "runs": runs,
             "greedy_flips": greedy_flips, "max_logit_diff_fold": w_fold,
             "max_logit_diff_unfold": w_unf, "ms_per_token_full": ms_full,
             "ms_per_token_cached": ms_cac, "speedup": ms_full / ms_cac,
             "step_ms_full": step_full * 1000, "step_ms_cached": step_cac * 1000,
             "step_speedup": step_full / step_cac,
             "bytes_per_token": c.bytes_per_token(), "bytes_total": c.total_bytes(),
             "control_runs_differing": ctrl_diff, "control_runs": ctrl_runs,
             "path_used": g_cac["path"]})

        # ---------------------------------------------------------------- P30f server parity
        os.environ["CHAT_DATA_DIR"] = os.path.join(ROOT, corpus)
        import lens.serve as serve
        serve.load_ship(path, "cpu")
        same = 0
        for prompt in PROMPTS[:3]:
            fmt = format_prompt(prompt, vocab)
            s = serve.generate(fmt, max_new=MAXNEW, seed=0)
            d = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, rule_stop=False, use_cache=True)
            same += int(s["text"] == d["text"] and s["tokens"] == d["tokens"])
        print(f"P30f server parity: {same}/3 prompts identical text to the measured decoder; "
              f"server path='{s['path']}', {s['tokens_per_sec']:.0f} tok/s, "
              f"cache {s['cache_bytes_total']} bytes total, vocab from {serve.STATE['vocab_src']}")
        verdicts[f"P30f {name}"] = (same == 3, f"{same}/3")
        summary["ships"][name]["server_parity"] = f"{same}/3"
        summary["ships"][name]["server_tokens_per_sec"] = s["tokens_per_sec"]

    # ---------------------------------------------------------------- P30i scaling with the window
    # A TIMING measurement on a synthetic net of the same architecture (real concept list, random weights).
    # Random weights, so no accuracy claim comes from this and it is not a trained ship.
    print("\n--- P30i does the cache pay off at longer windows? (SYNTHETIC net, random weights, TIMING ONLY)")
    base = lens.load(os.path.join(ROOT, SHIPS[0][0]), device="cpu")
    rows = []
    for W in (32, 128, 512):
        snet = lens.Net(base.V, base.concepts, d=base.D, window=W, seed=0, attention=True)
        rng = np.random.default_rng(0)
        stream = list(rng.integers(1, base.V - 2, size=W + 200))
        c0 = stream[:W]
        def loop(cached, snet=snet, stream=stream, c0=c0):
            c = None
            if cached:
                from lens.kvcache import WindowCache
                c = WindowCache(snet).seed(c0)
            cx = list(c0)
            for t in stream[len(c0):]:
                step_logits(snet, cx, c)
                cx = (cx + [t])[-snet.W:]
                if c is not None:
                    c.push(t)
        loop(False)                                        # warm up both
        loop(True)
        t0 = time.time(); loop(False); full = (time.time() - t0) / (len(stream) - W)
        t0 = time.time(); loop(True); cac = (time.time() - t0) / (len(stream) - W)
        rows.append((W, full * 1000, cac * 1000, full / cac))
        print(f"    window {W:4d}: full {full*1000:7.3f} ms/step   cached {cac*1000:7.3f} ms/step   "
              f"{full/cac:.2f}x")
    verdicts["P30i synthetic scaling (cached faster at every window)"] = (
        all(r[3] > 1.0 for r in rows), " ".join(f"W{r[0]}={r[3]:.2f}x" for r in rows))
    verdicts["P30i prediction held (ratio >= 1.5x at every window)"] = (
        all(r[3] >= 1.5 for r in rows), " ".join(f"W{r[0]}={r[3]:.2f}x" for r in rows))
    summary["scaling"] = [{"window": r[0], "ms_full": r[1], "ms_cached": r[2], "ratio": r[3]} for r in rows]

    # ---------------------------------------------------------------- P30g regression, no attention
    ship, corpus = NOATTN
    net = lens.load(os.path.join(ROOT, ship), device="cpu")
    vocab, _ = load_vocab(net, corpus)
    vmap = {w: i for i, w in enumerate(vocab)}
    a = generate(net, vocab, vmap, PROMPTS[0], max_new=20, seed=0, rule_stop=False)          # default
    b = generate(net, vocab, vmap, PROMPTS[0], max_new=20, seed=0, rule_stop=False, use_cache=False)
    named = "NO attention" in a["path"]
    print(f"\nP30g regression, no-attention ship {os.path.basename(ship)}: path='{a['path']}' "
          f"(names the reason: {named}), token ids equal to an explicit full run: "
          f"{a['token_ids'] == b['token_ids']}")
    verdicts[f"P30g {os.path.basename(ship)}"] = (named and a["token_ids"] == b["token_ids"], a["path"])

    print("\n" + "=" * 100)
    print("VERDICTS (binary; the bar was written before the run in KV-DECODE-PREDICTIONS.txt)")
    for k, (ok, detail) in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}   [{detail}]")
    allpass = all(ok for ok, _ in verdicts.values())
    print(f"OVERALL: {'PASS' if allpass else 'FAIL'}  ({sum(ok for ok, _ in verdicts.values())}/"
          f"{len(verdicts)} parts)")
    summary["verdicts"] = {k: bool(ok) for k, (ok, _) in verdicts.items()}
    summary["overall"] = bool(allpass)
    with open(os.path.join(ROOT, "lens/runs/kv_decode.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("machine-readable numbers: lens/runs/kv_decode.json")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
