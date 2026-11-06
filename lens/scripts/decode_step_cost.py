#!/usr/bin/env python3
"""decode_step_cost.py - PLAN item (f): the cached decode step is OVERHEAD-bound, not FLOP-bound.

The cached step decodes at ~0.26 ms/step at d=256 for ~1.5M multiply-adds (lens/runs/KV-DECODE.txt,
15:38 pass), i.e. the arithmetic is ~6 GFLOP/s and the machine is idle. The step is about 20 small numpy
calls, each with its own temporary. This script:

  1. BREAKS THE STEP DOWN, so "overhead-bound" is a measured claim and not a story (P46a/P46b);
  2. times the FUSED step against the ORIGINAL allocating code, in one process at one instant (P46c);
  3. proves the fused step returns BIT-IDENTICAL arrays (P46d) and generates IDENTICAL tokens (P46e),
     with a deliberately WRONG fused variant as the control that must differ (P46f);
  4. re-measures the numbers stored in lens/runs/KV-DECODE.txt on the DEFAULT path (P46g).

Predictions, written BEFORE any of this ran: lens/runs/DECODE-STEP-COST-PREDICTIONS.txt.
Run:  bash lens/scripts/run_decode_step_cost.sh    (output: lens/runs/DECODE-STEP-COST.txt)
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
from lens.decode import generate, format_prompt, step_logits   # noqa: E402
from lens.kvcache import WindowCache, cached_logits_h   # noqa: E402

SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest")]
PROMPTS = ["hello , how are you ?", "what is 2 + 2 ?", "who are you ?",
           "tell me about the sun .", "what is water made of ?"]
SEEDS = [0, 1, 2]
MAXNEW = 40
# the values stored in lens/runs/KV-DECODE.txt (15:38/15:39 passes) that P46g has to reproduce exactly
STORED = {"cl_d256_attn_12_ep12.npz": {"fold": 2.861e-06, "unfold": 3.576e-06, "bpt": 4096},
          "sb_B1_d512_attn_ep96.npz": {"fold": 1.478e-05, "unfold": 1.335e-05, "bpt": 8192}}


def load_vocab(net, corpus):
    p = os.path.join(ROOT, corpus, "vocab_v12000.json")
    vocab = json.load(open(p))
    if net.V == len(vocab) + 2:
        vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    if net.V != len(vocab):
        raise SystemExit(f"vocab mismatch: ship has {net.V}, the word list at {p} has {len(vocab)}")
    return vocab, p


def bench(fn, reps=5, inner=1):
    """Median seconds per call. The median, because one arm can be preempted by the training run."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t0) / inner)
    return sorted(ts)[len(ts) // 2]


def step_loop(net, stream, fused):
    """Exactly the decoder's per-token work: one step, then push the next token. No sampling."""
    W = int(net.W)
    c = WindowCache(net, fused=fused).seed(stream[:W])
    for t in stream[W:]:
        cached_logits_h(net, c)
        c.push(t)
    return c


class StaleVoCache(WindowCache):
    """THE CONTROL, and it is DELIBERATELY WRONG.

    The fused step writes the new token's value projection into the window. This variant copies the
    PREVIOUS token's Vo over it instead, so the window's last position carries the wrong value vector.
    It is the smallest possible corruption of the fused path, so if P46e passes this too, P46e proves
    nothing and the gate fails as decoration.
    """

    def push(self, token):
        super().push(token)
        self.Vo[-1] = self.Vo[-2]
        return self


def token_diff(a, b):
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def logit_walk(net, ctx, steps=MAXNEW, fold=True):
    """max |cached logits - full-recompute logits| along one generation (the P30b measurement)."""
    cache = WindowCache(net, fold=fold).seed(ctx)
    ctx = list(ctx)
    worst, worst_step = 0.0, -1
    for s in range(steps):
        d = float(np.abs(step_logits(net, ctx, None) - step_logits(net, ctx, cache)).max())
        if d > worst:
            worst, worst_step = d, s
        nxt = int(np.argmax(step_logits(net, ctx, None)))
        ctx = (ctx + [nxt])[-net.W:]
        cache.push(nxt)
    return worst, worst_step


def encode(vmap, text, W):
    import re
    enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", text.lower())]
    if len(enc) < W:
        enc = [0] * (W - len(enc)) + enc
    return enc[-W:]


def main():
    print("=" * 100)
    print("CACHED DECODE STEP: the breakdown, and a FUSED step that must stay token-for-token identical")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}   OPENBLAS_NUM_THREADS="
          f"{os.environ.get('OPENBLAS_NUM_THREADS')}   numpy {np.__version__}")
    print("predictions (written first): lens/runs/DECODE-STEP-COST-PREDICTIONS.txt")
    print("=" * 100)
    verdicts = {}
    out = {"ships": {}, "host": {"openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                                 "numpy": np.__version__,
                                 "loadavg": list(os.getloadavg())}}
    print(f"machine load at start: {os.getloadavg()}  (a train_chat arm is running: the ABSOLUTE times")
    print("below are inflated by it, every bar in this file is a RATIO measured in this process)")

    for ship, corpus in SHIPS:
        path = os.path.join(ROOT, ship)
        net = lens.load(path, device="cpu")
        vocab, vfile = load_vocab(net, corpus)
        vmap = {w: i for i, w in enumerate(vocab)}
        name = os.path.basename(ship)
        W, D = int(net.W), int(net.D)
        print(f"\n--- SHIP {name}  d={D} window={W} concepts={net.K} V={net.V} vocab={vfile}")
        print(f"    dtypes in play: emb {net.emb.data.dtype}  concept.W {net.concept.W.data.dtype}  "
              f"mask {net.attn.mask_np.dtype}  (the cache's own arrays take the weight dtype in seed())")
        srow = {"d": D, "window": W, "concepts": int(net.K), "V": int(net.V),
                "dtypes": {"emb": str(net.emb.data.dtype), "concept_W": str(net.concept.W.data.dtype),
                           "mask": str(net.attn.mask_np.dtype)}}
        out["ships"][name] = srow

        rng = np.random.default_rng(0)
        stream = [int(t) for t in rng.integers(1, net.V - 2, size=W + 2200)]

        # ---------------------------------------------------------------- P46c speed, both arms, one process
        step_loop(net, stream[:W + 60], False)                  # warm both arms
        step_loop(net, stream[:W + 60], True)
        n_steps = len(stream) - W
        reps = 5
        t_ref = bench(lambda: step_loop(net, stream, False), reps=reps) / n_steps
        t_fus = bench(lambda: step_loop(net, stream, True), reps=reps) / n_steps
        ms_ref, ms_fus = t_ref * 1000, t_fus * 1000
        print(f"P46c step, REFERENCE arm (the original allocating code, fused=False): {ms_ref:.3f} ms "
              f"({1000/ms_ref:.0f} tok/s)")
        print(f"     step, FUSED arm (in place, buffers allocated once, the new default) : {ms_fus:.3f} ms "
              f"({1000/ms_fus:.0f} tok/s)")
        print(f"     -> fused/reference = {ms_fus/ms_ref:.3f}x   (bar <= 0.70x)")
        verdicts[f"P46c {name}"] = (ms_fus / ms_ref <= 0.70, f"{ms_fus/ms_ref:.3f}x")
        srow.update({"ms_step_reference": ms_ref, "ms_step_fused": ms_fus,
                     "fused_ratio": ms_fus / ms_ref, "tokens_per_sec_fused": 1000.0 / ms_fus})

        # ---------------------------------------------------------------- P46a/P46b breakdown, REFERENCE arm
        c = WindowCache(net, fused=False).seed(stream[:W])
        p_push = bench(lambda: c.push(1), reps=200)
        p_out = bench(lambda: c.outputs(), reps=200)
        x = np.asarray(c.outputs()).reshape(1, -1)
        p_con = bench(lambda: x @ c.cW + c.cb, reps=200)
        pre = x @ c.cW + c.cb
        p_sig = bench(lambda: 1.0 / (1.0 + np.exp(np.clip(-pre, -60.0, 60.0))), reps=200)
        h = 1.0 / (1.0 + np.exp(np.clip(-pre, -60.0, 60.0)))
        p_ans = bench(lambda: h @ c.aW + c.ab, reps=200)
        parts = p_push + p_out + p_con + p_sig + p_ans
        frac = parts / t_ref
        print(f"P46a breakdown of the REFERENCE step ({t_ref*1e6:.0f} us total), parts as fractions:")
        for label, v in (("push (slide the 4 buffers, project the new token)", p_push),
                         ("outputs (Q@K.T, mask, softmax, p@Vo)", p_out),
                         ("concept GEMM x@cW + cb", p_con),
                         ("sigmoid of the concept pre-activations", p_sig),
                         ("answer GEMM h@aW + ab", p_ans)):
            print(f"       {label:<52s} {v*1e6:8.1f} us   {v/t_ref*100:5.1f}%")
        print(f"       {'SUM OF PARTS':<52s} {parts*1e6:8.1f} us   {frac*100:5.1f}%   (bar >= 80%)")
        biggest = max((("push", p_push), ("outputs", p_out), ("concept", p_con),
                       ("sigmoid", p_sig), ("answer", p_ans)), key=lambda kv: kv[1])[0]
        print(f"     the largest single part: {biggest}   (P46b predicted: answer)")
        verdicts[f"P46a {name}"] = (frac >= 0.80, f"parts = {frac*100:.1f}% of the step")
        verdicts[f"P46b {name}"] = (biggest == "answer", f"largest = {biggest}")
        srow.update({"breakdown_us": {"push": p_push * 1e6, "outputs": p_out * 1e6,
                                      "concept": p_con * 1e6, "sigmoid": p_sig * 1e6,
                                      "answer": p_ans * 1e6, "sum": parts * 1e6,
                                      "step": t_ref * 1e6},
                     "parts_fraction": frac, "largest_part": biggest})

        # ---------------------------------------------------------------- P46d identity, EXACT arrays
        rows = []
        for fold in (True, False):
            c1 = WindowCache(net, fold=fold, fused=False).seed(stream[:W])
            c2 = WindowCache(net, fold=fold, fused=True).seed(stream[:W])
            worst_z = worst_h = 0.0
            unequal_z = unequal_h = 0
            n = 0
            for t in stream[W:W + 2000]:
                z1, h1 = cached_logits_h(net, c1)
                z2, h2 = cached_logits_h(net, c2)
                worst_z = max(worst_z, float(np.abs(z1 - z2).max()))
                worst_h = max(worst_h, float(np.abs(h1 - h2).max()))
                unequal_z += int(not np.array_equal(z1, z2))
                unequal_h += int(not np.array_equal(h1, h2))
                c1.push(t)
                c2.push(t)
                n += 1
            print(f"P46d fold={str(fold):<5s} over {n} steps: max |fused - reference| logits "
                  f"{worst_z:.3e}  concepts {worst_h:.3e}  arrays not equal: {unequal_z}/{n} logits, "
                  f"{unequal_h}/{n} concepts  (bar: exactly 0.0)")
            rows.append((fold, n, worst_z, worst_h, unequal_z, unequal_h))
        exact = all(r[2] == 0.0 and r[3] == 0.0 and r[4] == 0 and r[5] == 0 for r in rows)
        verdicts[f"P46d {name}"] = (exact, "max diff " + ", ".join(f"fold{r[0]} {r[2]:.1e}/{r[3]:.1e}"
                                                                   for r in rows))
        srow["identity_exact"] = [{"fold": r[0], "steps": r[1], "max_logit_diff": r[2],
                                   "max_concept_diff": r[3], "unequal_logits": r[4],
                                   "unequal_concepts": r[5]} for r in rows]

        # ---------------------------------------------------------------- P46e/P46f tokens, and P30a regression
        def fac(fused, cls=WindowCache):
            return lambda n, c: cls(n, fused=fused).seed(c)

        flips_new = flips_greedy = 0
        runs = 0
        flips_ref_vs_default = flips_greedy_ref = 0
        ctrl_diff = ctrl_runs = 0
        example = None
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            for seed in SEEDS:
                full = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                                use_cache=False)
                old = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=True, cache_factory=fac(False))
                new = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=True, cache_factory=fac(True))
                bad = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=True, cache_factory=fac(True, StaleVoCache))
                runs += 1
                flips_new += int(token_diff(old["token_ids"], new["token_ids"]) > 0)
                flips_ref_vs_default += int(token_diff(full["token_ids"], new["token_ids"]) > 0)
                ctrl_runs += 1
                if token_diff(full["token_ids"], bad["token_ids"]) > 0:
                    ctrl_diff += 1
                    if example is None:
                        example = (prompt, seed, full["token_ids"][:12], bad["token_ids"][:12])
            gfull = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True, rule_stop=False,
                             use_cache=False)
            gold = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True, rule_stop=False,
                            use_cache=True, cache_factory=fac(False))
            gnew = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True, rule_stop=False,
                            use_cache=True, cache_factory=fac(True))
            flips_greedy += token_diff(gold["token_ids"], gnew["token_ids"])
            flips_greedy_ref += token_diff(gfull["token_ids"], gnew["token_ids"])
        print(f"P46e tokens: fused vs reference arm, {runs - flips_new}/{runs} runs identical, "
              f"greedy token differences {flips_greedy}   (bar: all runs, 0 flips)")
        print(f"P46f control (DELIBERATELY WRONG fused variant, stale Vo): differs from the full recompute "
              f"on {ctrl_diff}/{ctrl_runs} runs   (bar: >= 1)")
        if example:
            print(f"     e.g. prompt={example[0]!r} seed={example[1]}")
            print(f"       full recompute : {example[2]}")
            print(f"       wrong fused    : {example[3]}")
        print(f"P46g P30a regression on the DEFAULT (fused) path: full recompute vs fused "
              f"{runs - flips_ref_vs_default}/{runs} runs identical, greedy token differences "
              f"{flips_greedy_ref}   (stored in KV-DECODE.txt: 15/15 and 0)")
        verdicts[f"P46e {name}"] = (flips_new == 0 and flips_greedy == 0,
                                    f"{runs - flips_new}/{runs} runs, {flips_greedy} greedy flips")
        verdicts[f"P46f {name}"] = (ctrl_diff > 0, f"{ctrl_diff}/{ctrl_runs} runs differ")
        srow.update({"runs": runs, "fused_vs_reference_identical": runs - flips_new,
                     "greedy_flips_fused_vs_reference": flips_greedy,
                     "control_runs_differing": ctrl_diff, "control_runs": ctrl_runs,
                     "p30a_regression_identical": runs - flips_ref_vs_default,
                     "p30a_regression_greedy_flips": flips_greedy_ref})

        # ---------------------------------------------------------------- P46g P30b / P30d on the default path
        enc = encode(vmap, format_prompt(PROMPTS[3], vocab), W)
        w_fold, s_fold = logit_walk(net, enc, fold=True)
        w_unf, s_unf = logit_walk(net, enc, fold=False)
        st = STORED[name]
        print(f"P46g P30b on the default path: fold ON {w_fold:.3e} (step {s_fold}), fold OFF {w_unf:.3e} "
              f"(step {s_unf})   stored: {st['fold']:.3e} (step 10/39), {st['unfold']:.3e}")
        cc = WindowCache(net)
        bpt_ok = cc.bytes_per_token() == 4 * D * 4 and cc.total_bytes() == 16 * W * D
        print(f"P46g P30d size: {cc.bytes_per_token()} B/token, {cc.total_bytes()} B total "
              f"(stored {st['bpt']} B/token) -> {bpt_ok}")
        same_fold = abs(w_fold - st["fold"]) <= 1e-12 * max(1.0, st["fold"]) * 1e3
        verdicts[f"P46g {name}"] = (bpt_ok and same_fold and w_fold < 1e-4 and w_unf < 1e-4,
                                    f"fold {w_fold:.2e} (stored {st['fold']:.2e}), {cc.bytes_per_token()} B/token")
        srow.update({"p30b_fold": w_fold, "p30b_unfold": w_unf, "p30b_stored_fold": st["fold"],
                     "bytes_per_token": cc.bytes_per_token(), "bytes_total": cc.total_bytes()})

    print("\n" + "=" * 100)
    print("VERDICTS (binary; every bar was written before the run in DECODE-STEP-COST-PREDICTIONS.txt)")
    for k, (ok, detail) in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}   [{detail}]")
    allpass = all(ok for ok, _ in verdicts.values())
    print(f"OVERALL: {'PASS' if allpass else 'FAIL'}  ({sum(ok for ok, _ in verdicts.values())}/"
          f"{len(verdicts)} parts)")
    out["verdicts"] = {k: bool(ok) for k, (ok, _) in verdicts.items()}
    out["overall"] = bool(allpass)
    with open(os.path.join(ROOT, "lens/runs/decode_step_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("machine-readable numbers: lens/runs/decode_step_cost.json")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
