#!/usr/bin/env python3
"""qkv_batch_test.py - PLAN item (f3): can the batched [Wq|Wk|Wv] projection be APPLIED?

The item, as written in lens/runs/PLAN.txt line 337: the batched projection is FASTER (measured
0.737x/0.739x of the three separate matvecs at d=256, 0.831x/0.299x at d=512,
lens/runs/DECODE-STEP-COST.txt P46q) but NOT bit-identical, so it is not applied until a TOKEN-identity
test exists. This is that test.

What it measures (bars and predictions written BEFORE the run and BEFORE the code change, in
lens/runs/QKV-BATCH-PREDICTIONS.txt, P50a-P50i):
  P50a raw array identity over 50 real rows                       BAR True        predicted FAIL (honest)
  P50b token identity, d=256, 20 runs (15 sampled + 5 greedy)     BAR 0 differ    predicted PASS
  P50c token identity, d=512, same 20 runs                        BAR 0 differ    predicted PASS
  P50d logit walk, 40 steps, both caches fed the SAME token       BAR < 1e-4      predicted PASS
  P50e CONTROL: the same batch with the K and V blocks swapped    BAR >=1 differs predicted PASS
  P50f loop cost, 1,200 real steps, both arms, 1 and 3 threads    BAR faster somewhere  predicted d=512
  P50g default path unchanged: the stored P30b numbers reproduce  BAR exact       predicted PASS
  P50h bytes per token unchanged (Wcat is exactly Wq+Wk+Wv)       BAR equal       predicted PASS

The batched arm runs through lens.decode.generate's OWN cache_factory hook, so what is tested is the
shipped decoder, not a copy of it. Ships: the only two with an attention block.

Run:  bash lens/scripts/run_qkv_batch.sh        (appends to lens/runs/QKV-BATCH.txt)
"""
import os

os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("QKV_BATCH_THREADS", "3")

import json                     # noqa: E402
import re                       # noqa: E402
import sys                      # noqa: E402
import time                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# The project's lens/ directory must NOT go on sys.path as its own element: the top-level name `cuda`
# would then resolve to our own lens/cuda/ package and cupy dies with "No module named 'cuda.pathfinder'"
# (real bug, lens/runs/SYSPATH-SHADOW-FIX-class3.txt). lens/ is imported as `lens.X` via ROOT.
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                             # noqa: E402
from lens.decode import generate, format_prompt, step_logits   # noqa: E402
from lens.kvcache import WindowCache, cached_logits_h   # noqa: E402

PROMPTS = ["hello , how are you ?", "what is 2 + 2 ?", "who are you ?",
           "tell me about the sun .", "what is water made of ?"]
SEEDS = [0, 1, 2]
MAXNEW = 40
NSTEPS = 1200
THREADS = os.environ["OPENBLAS_NUM_THREADS"]
pc = time.perf_counter
# (ship, the corpus it was trained on - the vocabulary MUST come from that corpus, the files differ)
SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest")]
# The P30b numbers already on the record, with the file they came from. P50g must reproduce them EXACTLY
# with qkv_batched left at its default (False), otherwise this change touched the shipped decode path.
STORED_P30B = {"cl_d256_attn_12_ep12.npz": (2.86102294921875e-06, 10, "lens/runs/KV-DECODE.txt"),
               "sb_B1_d512_attn_ep96.npz": (1.4781951904296875e-05, 39, "lens/runs/KV-DECODE.txt")}


def load_vocab(net, corpus):
    """The word list of the corpus this ship was trained on, named by file."""
    p = os.path.join(ROOT, corpus, "vocab_v12000.json")
    vocab = json.load(open(p))
    if net.V == len(vocab) + 2:
        vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    if net.V != len(vocab):
        raise SystemExit(f"vocab mismatch: ship has {net.V}, the word list at {p} has {len(vocab)}")
    return vocab, p


def batched_factory(net, ctx):
    """The arm under test: ONE matvec for [Wq|Wk|Wv], through the decoder's own cache_factory hook."""
    return WindowCache(net, qkv_batched=True).seed(ctx)


def swap_factory(net, ctx):
    """THE CONTROL: the same batch, the same bytes, the K and V blocks SWAPPED inside the concatenation.

    So K is projected with Wv and V with Wk. If P50b passes with this arm too, then P50b proves nothing
    and the gate fails as decoration.
    """
    c = WindowCache(net, qkv_batched=True)
    c.Wcat = np.concatenate([c.Wq, c.Wv, c.Wk], axis=1)
    return c.seed(ctx)


def token_diff(a, b):
    """How many tokens differ, counting a length difference as a difference."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def array_identity(net, stream):
    """P50a: is the batched (D,3D) matvec bit-identical to the three separate (D,D) ones?"""
    W = int(net.W)
    c = WindowCache(net, qkv_batched=False).seed(stream[:W])
    xs = c.emb[np.asarray(stream[W:W + 50], dtype=np.int64)]
    wcat = np.concatenate([c.Wq, c.Wk, c.Wv], axis=1)
    one = xs @ wcat
    three = np.concatenate([xs @ c.Wq, xs @ c.Wk, xs @ c.Wv], axis=1)
    return bool(np.array_equal(one, three)), float(np.abs(one - three).max())


def window_identity(net, stream):
    """INFO: the whole window's contextualised output, batched versus separate (not a prediction)."""
    W = int(net.W)
    a = WindowCache(net, qkv_batched=False).seed(stream[:W])
    b = WindowCache(net, qkv_batched=True).seed(stream[:W])
    d = float(np.abs(a.outputs() - b.outputs()).max())
    qkv = float(max(np.abs(a.Q - b.Q).max(), np.abs(a.K - b.K).max(), np.abs(a.V - b.V).max()))
    return qkv, d


def logit_walk_pair(net, ctx, steps):
    """P50d: both caches fed the SAME next token, so the only difference is the projection."""
    ref = WindowCache(net, qkv_batched=False).seed(ctx)
    bat = WindowCache(net, qkv_batched=True).seed(ctx)
    ctx = list(ctx)
    worst, worst_step, argmax_flips = 0.0, -1, 0
    for s in range(steps):
        zr = np.asarray(cached_logits_h(net, ref)[0], dtype=np.float64)
        zb = np.asarray(cached_logits_h(net, bat)[0], dtype=np.float64)
        d = float(np.abs(zr - zb).max())
        if d > worst:
            worst, worst_step = d, s
        if int(np.argmax(zr)) != int(np.argmax(zb)):
            argmax_flips += 1
        nxt = int(np.argmax(zr))              # the shipped arm decides the path both follow
        ctx = (ctx + [nxt])[-net.W:]
        ref.push(nxt)
        bat.push(nxt)
    return worst, worst_step, argmax_flips


def default_regression(net, ctx, steps):
    """P50g: the SAME walk kv_decode_test.py does, on the DEFAULT cache - it must hit the stored numbers."""
    cache = WindowCache(net, fold=True).seed(ctx)
    ctx = list(ctx)
    worst, worst_step = 0.0, -1
    for s in range(steps):
        z_ref = step_logits(net, ctx, None)
        z_c = step_logits(net, ctx, cache)
        d = float(np.abs(z_ref - z_c).max())
        if d > worst:
            worst, worst_step = d, s
        nxt = int(np.argmax(z_ref))
        ctx = (ctx + [nxt])[-net.W:]
        cache.push(nxt)
    return worst, worst_step


def loop_cost(net, stream, batched, reps=3):
    """P50f: ms per step of the REAL cached step (cached_logits_h + push), median of `reps`."""
    W = int(net.W)
    ts = []
    for _ in range(reps):
        c = WindowCache(net, qkv_batched=batched).seed(stream[:W])
        for t in stream[W:W + 50]:            # warm, not timed
            cached_logits_h(net, c)
            c.push(t)
        c = WindowCache(net, qkv_batched=batched).seed(stream[:W])
        t0 = pc()
        for t in stream[W:W + NSTEPS]:
            cached_logits_h(net, c)
            c.push(t)
        ts.append((pc() - t0) / NSTEPS)
    return sorted(ts)[len(ts) // 2] * 1e6     # median us/step


def main():
    print("=" * 100)
    print(f"PLAN ITEM (f3): THE BATCHED [Wq|Wk|Wv] PROJECTION - does it move a TOKEN?  THREADS={THREADS}")
    print(f"pass started {time.strftime('%Y-%m-%d %H:%M:%S')}  numpy {np.__version__}  "
          f"loadavg {tuple(round(x, 2) for x in os.getloadavg())}")
    print("predictions (written first): lens/runs/QKV-BATCH-PREDICTIONS.txt (P50a-P50i)")
    print("(this file is APPENDED to: earlier blocks are earlier passes, kept and never overwritten)")
    print("=" * 100)
    out = {"threads": THREADS, "ships": {}}
    verdicts = {}

    for ship, corpus in SHIPS:
        net = lens.load(os.path.join(ROOT, ship), device="cpu")
        vocab, vfile = load_vocab(net, corpus)
        vmap = {w: i for i, w in enumerate(vocab)}
        name = os.path.basename(ship)
        print(f"\n--- SHIP {name}  d={net.D} window={net.W} concepts={net.K} V={net.V} "
              f"attention=yes  vocab={vfile}")
        rng = np.random.default_rng(20261021)
        stream = list(rng.integers(1, net.V - 2, size=NSTEPS + 200))

        # ------------------------------------------------------------------ P50a array identity
        ident, amax = array_identity(net, stream)
        qkv_d, win_d = window_identity(net, stream)
        print(f"P50a raw array identity over 50 real rows: {ident}   max |batched-separate| {amax:.3e}")
        print(f"     info: the window's Q/K/V max diff {qkv_d:.3e}; the window's OUTPUTS max diff {win_d:.3e}")
        verdicts[f"P50a {name}"] = (ident, f"identical={ident}, max {amax:.2e}")

        # ------------------------------------------------------------------ P50b/P50c token identity
        runs = differ = 0
        first = None
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            for seed in SEEDS:
                ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               use_cache=True)
                bat = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=seed, rule_stop=False,
                               cache_factory=batched_factory)
                d = token_diff(ref["token_ids"], bat["token_ids"])
                runs += 1
                if d:
                    differ += 1
                    if first is None:
                        first = ("sampled", prompt, seed, d, ref["token_ids"][:12], bat["token_ids"][:12])
        greedy_runs = greedy_differ = 0
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            g_ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True,
                             rule_stop=False, use_cache=True)
            g_bat = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, greedy=True,
                             rule_stop=False, cache_factory=batched_factory)
            greedy_runs += 1
            if token_diff(g_ref["token_ids"], g_bat["token_ids"]):
                greedy_differ += 1
                if first is None:
                    first = ("greedy", prompt, 0, 1, g_ref["token_ids"][:12], g_bat["token_ids"][:12])
        total_runs = runs + greedy_runs
        print(f"P50b/c token identity: {total_runs - differ - greedy_differ}/{total_runs} runs identical "
              f"({runs} sampled + {greedy_runs} greedy x {MAXNEW} tokens); differing runs: "
              f"sampled {differ}, greedy {greedy_differ}")
        if first:
            print(f"     FIRST DIFFERENCE: kind={first[0]} prompt={first[1]!r} seed={first[2]} "
                  f"differing={first[3]}")
            print(f"       shipped: {first[4]}")
            print(f"       batched: {first[5]}")
        verdicts[f"P50b/c {name}"] = (differ == 0 and greedy_differ == 0,
                                      f"{total_runs - differ - greedy_differ}/{total_runs} runs")

        # ------------------------------------------------------------------ P50d logit walk
        enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", format_prompt(PROMPTS[3], vocab).lower())]
        if len(enc) < net.W:
            enc = [0] * (net.W - len(enc)) + enc
        w, ws, flips = logit_walk_pair(net, enc[-net.W:], MAXNEW)
        print(f"P50d logit walk over {MAXNEW} steps, same tokens both sides: max |z_batched-z_separate| "
              f"{w:.3e} (step {ws}), argmax flips {flips}")
        verdicts[f"P50d {name}"] = (w < 1e-4 and flips == 0, f"{w:.2e} (step {ws}), flips {flips}")

        # ------------------------------------------------------------------ P50e control
        ctrl_differ = 0
        for prompt in PROMPTS:
            fmt = format_prompt(prompt, vocab)
            ref = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, rule_stop=False, use_cache=True)
            swp = generate(net, vocab, vmap, fmt, max_new=MAXNEW, seed=0, rule_stop=False,
                           cache_factory=swap_factory)
            if token_diff(ref["token_ids"], swp["token_ids"]):
                ctrl_differ += 1
        print(f"P50e CONTROL (same batch, K and V blocks swapped): {ctrl_differ}/{len(PROMPTS)} runs differ "
              f"from the shipped path (bar >= 1)")
        verdicts[f"P50e {name}"] = (ctrl_differ >= 1, f"{ctrl_differ}/{len(PROMPTS)} runs differ")

        # ------------------------------------------------------------------ P50g default path unchanged
        exp, exp_step, exp_file = STORED_P30B[name]
        got, got_step = default_regression(net, enc[-net.W:], MAXNEW)
        same = (got == exp and got_step == exp_step)
        print(f"P50g default path (qkv_batched=False) logit walk: {got!r} at step {got_step} against the "
              f"stored {exp!r} at step {exp_step} ({exp_file}) -> {'EXACT' if same else 'DIFFERENT'}")
        verdicts[f"P50g {name}"] = (same, f"{got:.6e} step {got_step} vs stored {exp:.6e} step {exp_step}")

        # ------------------------------------------------------------------ P50f loop cost
        loop_cost(net, stream, False, reps=1)                  # warm both paths first
        loop_cost(net, stream, True, reps=1)
        sep = loop_cost(net, stream, False)
        bat = loop_cost(net, stream, True)
        print(f"P50f loop cost over {NSTEPS} real steps: separate {sep:.1f} us/step, batched {bat:.1f} "
              f"us/step -> batched/separate {bat / sep:.3f}x ({sep / bat:.3f}x faster)")
        verdicts[f"P50f {name}"] = (bat < sep, f"{bat / sep:.3f}x of the separate arm")

        # ------------------------------------------------------------------ P50h bytes
        c = WindowCache(net, qkv_batched=True).seed(stream[:net.W])
        bpt, tot = c.bytes_per_token(), c.total_bytes()
        proj = 4 * int(net.D) * int(net.D) * 4
        print(f"P50h bytes: cache {bpt} B/token, {tot} B total (W={net.W}); the batch's Wcat is "
              f"{int(net.D) * 3 * int(net.D) * 4} B = the same three projection matrices "
              f"(projection traffic {proj} B/token either way)")
        verdicts[f"P50h {name}"] = (bpt == 16 * int(net.D), f"{bpt} B/token")

        out["ships"][name] = {"d": int(net.D), "window": int(net.W), "concepts": int(net.K),
                              "raw_identical": ident, "raw_max_abs": amax, "qkv_max_abs": qkv_d,
                              "window_outputs_max_abs": win_d,
                              "runs": total_runs, "runs_differing": differ + greedy_differ,
                              "walk_max_abs": w, "walk_step": ws, "walk_argmax_flips": flips,
                              "control_runs_differing": ctrl_differ,
                              "default_walk": got, "default_walk_step": got_step,
                              "stored_walk": exp, "stored_walk_step": exp_step,
                              "loop_us_separate": sep, "loop_us_batched": bat,
                              "loop_ratio": bat / sep, "bytes_per_token": bpt,
                              "vocab_file": vfile}

    print("\n" + "=" * 100)
    print(f"VERDICTS at {THREADS} thread(s) (binary; bars written before the run)")
    for k, (ok, detail) in verdicts.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}   [{detail}]")
    print("=" * 100)
    path = os.path.join(ROOT, f"lens/runs/qkv_batch_t{THREADS}.json")
    json.dump({"verdicts": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in verdicts.items()}, **out},
              open(path, "w"), indent=1)
    print(f"machine-readable numbers: {path}")
    return 0 if all(v[0] for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
