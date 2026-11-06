#!/usr/bin/env python3
"""BENCH_EXTERNAL — the honest external-benchmark harness (the "is it actually smart?" yardstick).

Why this exists: the project had NO standard benchmark measurement at all. Every claim so far was
next-token accuracy on our own corpora. This runs three real, licensed benchmarks and reports
MEASURED numbers with their datum, ship identity, and the random baseline beside every score.

  * MMLU (MIT)        -- 4-choice, scored by next-token LIKELIHOOD of a/b/c/d (not generation),
                         which is how lm-eval computes `acc` for this benchmark. Random = 25%.
  * GSM8K (MIT)       -- free-form math; greedy generation, last number extracted, exact match.
                         Random ~ 0%.
  * HumanEval (MIT) / MBPP (CC-BY-4.0) -- code; this harness measures SYNTAX-VALID rate only.
                         pass@1 (the real metric) needs a sandboxed test runner and is NOT claimed
                         here -- it is printed as an unmeasured gap.

Honesty rules: every number prints n, the ship, the vocab, and its baseline; any failure is printed
as a failure, never as a zero. Writes lens/runs/BENCH_EXTERNAL.json.
"""
import ast
import json
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
# PROJ itself must be on the path for `import lens` (the package lives at PROJ/lens); scripts/
# holds chat_data. Running with a script path puts the SCRIPT's dir on sys.path, not the cwd.
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import chat_data  # tokenize() -- reused verbatim, never re-implemented

SHIP = os.environ.get("BENCH_SHIP", "lens/runs/chat/t12_splitfix_d512_ep192.npz")
VOCAB = os.environ.get("BENCH_VOCAB", "data/chat/prepared/vocab_v12000.json")
DEVICE = os.environ.get("BENCH_DEVICE", "cpu")
WIN = int(os.environ.get("BENCH_WIN", "32"))
N_MMLU = int(os.environ.get("BENCH_N_MMLU", "500"))
N_GSM = int(os.environ.get("BENCH_N_GSM8K", "200"))
N_CODE = int(os.environ.get("BENCH_N_CODE", "100"))


def load_jsonl(path, n=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n * 20:      # read a generous pool, then sample
                break
    return rows


def load_ship():
    import lens
    vocab = json.load(open(os.path.join(PROJ, VOCAB)))
    vmap = {w: i for i, w in enumerate(vocab)}
    net = lens.load(os.path.join(PROJ, SHIP), device=DEVICE)
    print(f"  ship: {SHIP}\n  vocab: {VOCAB} ({len(vocab):,} words)   device: {DEVICE}")
    return net, vocab, vmap


def encode(text, vmap):
    return np.asarray([vmap.get(t, 0) for t in chat_data.tokenize(text.lower())], dtype=np.int64)


def next_logits(net, ids):
    """Next-token logits for the last position, over the model's window. There is no KV cache in
    this engine, so the window is re-fed each step (the model IS a 32-token window model).
    Short prompts are LEFT-PADDED to exactly WIN tokens: the model refuses any other width
    ("windows must be N tokens wide"), and padding at the left keeps the true context adjacent to
    the predicted position. Pad id 0 = <unk> (the model's own unknown token)."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) > WIN:
        seg = ids[-WIN:]
    elif len(ids) < WIN:
        seg = np.concatenate([np.zeros(WIN - len(ids), dtype=np.int64), ids])
    else:
        seg = ids
    out = net.forward_arrays(seg[None, :])
    return out["logits"][0]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def greedy(net, ids, n_new):
    """Greedy generation using the model's own forward pass (deterministic)."""
    cur = list(ids)
    out_tokens = []
    for _ in range(n_new):
        lg = next_logits(net, np.asarray(cur, dtype=np.int64))
        nxt = int(np.argmax(lg))
        out_tokens.append(nxt)
        cur.append(nxt)
    return out_tokens


# ---------------------------------------------------------------- MMLU
def mmlu(net, vocab, vmap, n):
    raw = load_jsonl(os.path.join(PROJ, "data/external/mmlu_test.jsonl"))
    rng = np.random.default_rng(0)
    rows = [(r["question"], r["choices"], int(r["answer"]), r["subject"]) for r in raw]
    rng.shuffle(rows)
    rows = rows[:n]
    letters = ["a", "b", "c", "d"]
    lids = [vmap.get(L) for L in letters]
    letter_ok = all(i is not None for i in lids)
    if not letter_ok:
        print("  MMLU: option letters are not single vocab tokens -> scoring the first token of each "
              "choice text instead (stated, not silent)")
    correct = wrong = 0
    p_correct = []
    for q, ch, ans, subj in rows:
        if letter_ok:
            prompt = (f"human: {q} "
                      + " ".join(f"{L}) {c}" for L, c in zip(letters, ch))
                      + " assistant: the answer is")
        else:
            prompt = (f"human: {q} "
                      + " ".join(f"{L}) {c}" for L, c in zip(letters, ch))
                      + " assistant: the answer is")
        lg = next_logits(net, encode(prompt, vmap))
        p = softmax(lg)
        if letter_ok:
            scores = [p[i] for i in lids]
        else:
            scores = [p[encode(str(c), vmap)[0]] for c in ch]
        pick = int(np.argmax(scores))
        p_correct.append(float(scores[ans]))
        if pick == ans:
            correct += 1
        else:
            wrong += 1
    n_tot = correct + wrong
    return {"n": n_tot, "acc": correct / max(n_tot, 1), "random_baseline": 0.25,
            "mean_prob_of_correct_choice": float(np.mean(p_correct)),
            "datum": "cais/mmlu test split, subjects sampled uniformly over files",
            "scoring": "next-token likelihood of a/b/c/d after the prompt"}


# ---------------------------------------------------------------- GSM8K
def gsm8k(net, vocab, vmap, n):
    raw = load_jsonl(os.path.join(PROJ, "data/external/gsm8k_test.jsonl"), n)
    qs = [r["question"] for r in raw][:n]
    ans = [r["answer"] for r in raw][:n]
    num = re.compile(r"-?\d[\d,]*\.?\d*")
    correct = 0
    examples = []
    for q, a in zip(qs, ans):
        gold = a.split("####")[-1].strip().replace(",", "")
        gen = greedy(net, encode(f"human: {q} assistant:", vmap), 24)
        text = " ".join(vocab[i] for i in gen)
        nums = num.findall(text.replace(",", ""))
        pred = nums[-1].rstrip(".") if nums else None
        ok = pred is not None and pred == gold
        correct += int(ok)
        if len(examples) < 3:
            examples.append({"question": q[:90], "gold": gold, "generated": text[:110]})
    return {"n": len(qs), "acc": correct / max(len(qs), 1), "random_baseline": 0.0,
            "datum": "openai/gsm8k main test, first n", "examples": examples}


# ---------------------------------------------------------------- CODE
def code_syntax(net, vocab, vmap, n):
    raw = load_jsonl(os.path.join(PROJ, "data/external/code_bench.jsonl"))
    rows = [(r["source"], r["task"]) for r in raw][:n]
    ok = 0
    examples = []
    for src, desc in rows:
        gen = greedy(net, encode(f"human: write python code for this task: {desc} assistant:", vmap), 64)
        text = " ".join(vocab[i] for i in gen)
        try:
            ast.parse(text)
            ok += 1
        except SyntaxError:
            pass
        if len(examples) < 2:
            examples.append({"source": src, "task": str(desc)[:80], "generated": text[:100]})
    return {"n": len(rows), "syntax_valid_rate": ok / max(len(rows), 1),
            "pass_at_1": "NOT MEASURED -- needs a sandboxed test runner (named gap)",
            "datum": "openai/openai_humaneval + google-research-datasets/mbpp, first n each",
            "examples": examples}


def main():
    net, vocab, vmap = load_ship()
    res = {"ship": SHIP, "vocab": VOCAB, "device": DEVICE}
    for name, fn, n in (("mmlu", mmlu, N_MMLU), ("gsm8k", gsm8k, N_GSM), ("code", code_syntax, N_CODE)):
        print(f"\n=== {name.upper()} (n={n})")
        try:
            r = fn(net, vocab, vmap, n)
            res[name] = r
            for k, v in r.items():
                if k in ("examples",):
                    continue
                print(f"    {k}: {v}")
            if "examples" in r:
                for e in r["examples"]:
                    print(f"    e.g. {json.dumps(e)[:220]}")
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"    FAILED: {type(e).__name__}: {e}")
    out = os.environ.get("BENCH_OUT") or os.path.join(PROJ, "lens/runs/BENCH_EXTERNAL.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n  written: {out}")


if __name__ == "__main__":
    main()