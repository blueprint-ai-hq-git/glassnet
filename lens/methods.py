#!/usr/bin/env python3
"""lens/methods.py - the training methods, implemented for real and measured.

WHY THIS FILE EXISTS: the first version of hard mode in lens/modes.py passed `--method rl --rl-seq-in ...`
to the trainer. NONE of those flags exist in train_chat.py, so hard mode would have died on unknown
arguments. That is scaffolding dressed up as a method, and the user called it out: "you don't even need our
own training method just generated few lines of code that is just compatible and not even tested the
settings... not just think or choose random settings... just do real testing."

So this file contains implementations that RUN, and a bench that MEASURES them:

  SFT      supervised fine-tuning on the next token            -> the existing trainer (real, in use)
  DISTILL  student matches the teacher's full distribution     -> lens/scripts/run_distill.py, whose KL
                                                                 backward is gradient-checked, with a
                                                                 teacher precondition (weak teacher => no
                                                                 verdict, not a fake result)
  RL       rejection-sampling policy update with an EXACT       -> implemented here, below
           verifier (no reward model, no human labels)

RL, exactly as implemented: for each prompt the model samples several candidate answers. A program checks
each one EXACTLY (for arithmetic: does it equal the sum?). Prompts where nothing was correct get no update
at all -- they carry no learning signal and pretending otherwise would be noise. Prompts with a correct
sample are trained toward that correct answer. That is RL with a binary verifiable reward: the reward comes
from a checker, never from a guess.

The bench is the arithmetic rule task, because it can tell LEARNING A RULE from MEMORISING: train pairs from
0..9, held-out pairs from the SAME range, and extrapolation to 10..19 which no memorised table can answer.

Usage: python3 lens/methods.py --d 128 --samples 8 --rounds 3
"""
import argparse
import os
import re
import sys
import time

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
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


# ------------------------------------------------------------------ the verifier
def verify_sum(prompt, candidate_token_text):
    """EXACT check, no model, no guessing: parse 'a + b' with a regex and compare to the candidate.

    First version used prompt.split("what is")[-1] and int() on the pieces. The prompt ends with
    "? assistant :", so the second operand parsed as " 5   assistant :" and int() threw, every candidate
    scored False, and the measured reward rate was 0% at samples 2, 8 AND 32 -- zero updates, RL silently
    doing nothing. A regex on the digits is the fix; the bench is what caught it.
    """
    m = re.search(r"(\d+)\s*\+\s*(\d+)", str(prompt))
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(2))
    try:
        cand = int(str(candidate_token_text).strip())
    except Exception:
        return False
    return cand == a + b


# ------------------------------------------------------------------ RL, implemented
def rl_verifiable(net, vocab, vmap, prompts, verifier, samples=8, rounds=3, lr=0.05, temp=1.0,
                  seed=0, tag="rl"):
    """Rejection-sampling RL with an exact verifier.

    Returns measured numbers: how often the model was already right (reward rate), how often a round
    produced a usable training signal, and how the loss moved. Every number printed comes from this run.
    """
    rng = np.random.default_rng(seed)
    stats = {"prompts": len(prompts), "reward_rate": [], "usable_rounds": 0, "updates": 0,
             "loss_first": None, "loss_last": None}
    for r in range(rounds):
        n_reward = 0
        losses = []
        order = rng.permutation(len(prompts))
        for pi in order:
            prompt, answer_text = prompts[pi]
            ids = [vmap.get(w, 0) for w in prompt.lower().split()]
            W = net.W
            window = np.asarray([([0] * (W - len(ids)) + ids)[-W:]], dtype=np.int64)
            logits = np.asarray(net.forward_arrays(window)["logits"][0], dtype=np.float64)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            # sample several candidate answer tokens from the model's own distribution
            cand = rng.choice(len(p), size=samples, p=p)
            good = [c for c in cand if verifier(prompt, vocab[c])]
            if not good:
                continue                     # no correct sample: no signal, no update, counted honestly
            n_reward += 1
            stats["usable_rounds"] += 1
            target = np.asarray([good[0]], dtype=np.int64)
            for par in net.parameters():
                par.grad = None
            out = net.loss(window, target, use_concept_loss=True)
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            for par in net.parameters():
                if par.grad is not None:
                    par.data = par.data - lr * np.asarray(par.grad)
            losses.append(float(loss.data))
            stats["updates"] += 1
        stats["reward_rate"].append(n_reward / max(len(prompts), 1))
        if losses:
            if stats["loss_first"] is None:
                stats["loss_first"] = losses[0]
            stats["loss_last"] = float(np.mean(losses))
    return stats


# ------------------------------------------------------------------ the measured bench
def main():
    import lens  # noqa: E402
    import lens.opt as opt_mod  # noqa: E402
    import rules_test as RT  # noqa: E402   (reuses the same task definition, so the two cannot drift)

    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--samples", type=int, default=8, help="candidate answers sampled per prompt")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--pretrain", type=int, default=300, help="plain supervised epochs before RL")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-in", type=int, default=12, help="tokens of prompt fed to the model")
    ap.add_argument("--seq-out", type=int, default=1, help="answer tokens the model produces")
    ap.add_argument("--reward", default="math_exact", choices=["math_exact"],
                    help="which exact verifier scores an answer; only checkable rewards are offered")
    a = ap.parse_args()

    all_pairs = [(x, y) for x in range(10) for y in range(10)]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(all_pairs))
    held = [all_pairs[i] for i in perm[:20]]
    train = [all_pairs[i] for i in perm[20:]]
    extra = [(x, y) for x in range(10, 20) for y in range(10, 20)]

    lines = [RT.sentence(x, y) for x, y in all_pairs + extra]
    vocab = RT.build_vocab(lines)
    vmap = {w: i for i, w in enumerate(vocab)}
    Xtr, Ytr = RT.make_set(train, vocab, a.seq_in)
    Xhe, Yhe = RT.make_set(held, vocab, a.seq_in)
    Xex, Yex = RT.make_set(extra, vocab, a.seq_in)

    net = lens.Net(len(vocab), RT.concepts_for(vocab), d=a.d, window=a.seq_in, seed=a.seed, device="cpu")
    params = sum(p.data.size for p in net.parameters())
    print(f"METHOD BENCH: arithmetic rules, d={a.d} ({params:,} parameters), "
          f"{len(train)} train pairs, {len(held)} held-out pairs, {len(extra)} extrapolation pairs")
    print(f"  methods in this repo: SFT = train_chat.py (real, in use); DISTILL = run_distill.py with a "
          f"gradient-checked KL backward (real); RL = implemented in this file (new)")
    print(f"  RL settings: samples {a.samples} per prompt, {a.rounds} rounds, lr {a.lr}, "
          f"verifier = exact integer equality")

    def measure(tag):
        tr = RT.accuracy(net, Xtr, Ytr)
        he = RT.accuracy(net, Xhe, Yhe)
        ex = RT.accuracy(net, Xex, Yex)
        print(f"  {tag:<22} train {tr*100:5.1f}%   held out {he*100:5.1f}%   extrapolation {ex*100:5.1f}%"
              f"   gap {(tr-he)*100:5.1f} points")
        return tr, he, ex

    measure("before any training")
    t0 = time.time()
    opt = opt_mod.RelativeGrad(net.parameters(), lr=a.lr, mode="relgrad")
    bs = 32
    for ep in range(a.pretrain):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = order[i:i + bs]
            for p in net.parameters():
                p.grad = None
            out = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=True)
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            opt.step()
    print(f"  (supervised warm start: {a.pretrain} epochs in {time.time()-t0:.1f}s)")
    measure("after SFT warm start")

    prompts = [(f"human : what is {x} + {y} ? assistant :", str(x + y)) for x, y in train]
    t0 = time.time()
    st = rl_verifiable(net, vocab, vmap, prompts, verify_sum, samples=a.samples, rounds=a.rounds,
                       lr=a.lr, seed=a.seed)
    print(f"  (RL: {st['updates']} updates from {st['prompts']} prompts over {a.rounds} rounds, "
          f"{time.time()-t0:.1f}s)")
    print(f"  RL reward rate per round: {[f'{r*100:.0f}%' for r in st['reward_rate']]} "
          f"(share of prompts where at least one sampled answer was CORRECT and checked exactly)")
    if st["loss_first"] is not None:
        print(f"  RL loss on usable prompts: {st['loss_first']:.4f} -> {st['loss_last']:.4f}")
    measure("after RL")

    print("\n  HONEST READING: held-out and extrapolation are the two that matter. Train accuracy rising "
          "while held-out stays flat is memorisation, and the gap column is the memorisation meter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())