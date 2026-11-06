"""code_data.py — turn licensed code into training windows AND a checkable test.

WHY THIS IS THE RIGHT TARGET FOR A SMALL MODEL
----------------------------------------------
"Knows everything" is not reachable on this machine (the ceiling measurement says so). Code is
different: it is structured, deterministic, and low-entropy. A small model can learn real patterns
like "a function that returns the sum of two numbers" without world knowledge.

AND IT GIVES A PROVABLE TEST. The owner's rule: "a CHECKABLE task, where a wrong answer is provably
wrong". For code that means the answer can be EXECUTED. A generated function either runs and
produces the right value, or it does not. No interpretation, no judge model, no opinion.

SOURCES (licence checked before download; all on the allowed list)
  ise-uiuc/Magicoder-OSS-Instruct-75K   MIT   problem -> solution, with a language tag
  codeparrot/apps                       MIT   problem -> solution + input_output test cases

FORMAT
------
The model is trained on plain text, so the training text is the problem followed by the solution.
At evaluation the prompt is the problem, and whatever the model writes is executed.
"""
import json
import os
import re

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CODE = os.path.join(PROJ, "data", "code")
CACHE = os.path.join(CODE, "prepared")
os.makedirs(CACHE, exist_ok=True)

VOCAB_CAP = int(os.environ.get("CD_VOCAB", "6000"))
VAL_FRAC = 0.02
SEED = 0
MAX_CHARS = int(os.environ.get("CODE_MAX_CHARS", "3000"))

# WHAT GOES IN A TRAINING DOCUMENT.
#   "both" (the original) = problem text + "\n\n" + solution
#   "code"               = the SOLUTION ONLY
# The diagnosis that produced this switch, measured on the 19.5M-token corpus built with "both":
#     most common token   '`'     6.08%   <- a markdown backtick
#     second most common  '<unk>' 5.74%   <- 1,119,649 unknown tokens, where identifiers go
#     third most common   'the'   5.55%   <- English prose
#     43% of documents open with the problem statement
# The model writes prose about code because half its training text IS prose, and English words crowd
# the 6,000 vocabulary slots that code identifiers need. This switch is the ONE change.
DOC_MODE = os.environ.get("CODE_DOC", "both")
if DOC_MODE not in ("both", "code", "clean"):
    raise SystemExit(f"CODE_DOC must be 'both', 'code' or 'clean', got {DOC_MODE!r}")

# only these languages: the checker runs python, so training should be dominated by python
KEEP_LANGS = {"python", "py", "python3"}


def load_pairs(limit=None):
    """Yield (problem, solution, lang) from the licensed sources."""
    out = 0
    mp = os.path.join(CODE, "magicoder_oss_instruct.jsonl")
    if os.path.exists(mp):
        with open(mp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                p, s = d.get("problem") or "", d.get("solution") or ""
                if not p.strip() or not s.strip():
                    continue
                yield p[:MAX_CHARS], s[:MAX_CHARS], (d.get("lang") or "unknown").lower()
                out += 1
                if limit and out >= limit:
                    return
    ap = os.path.join(CODE, "apps_train.jsonl")
    if os.path.exists(ap):
        with open(ap) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = d.get("question") or ""
                sols = d.get("solutions") or []
                if not q.strip() or not sols:
                    continue
                yield q[:MAX_CHARS], sols[0][:MAX_CHARS], "python"
                out += 1
                if limit and out >= limit:
                    return


def python_pairs(limit=None):
    """Only Python pairs -- the language the checker can execute."""
    n = 0
    for p, s, lang in load_pairs(limit=limit):
        if lang in KEEP_LANGS or lang == "unknown":
            yield p, s
            n += 1


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\sA-Za-z_0-9]")


def tokenize(text):
    return TOKEN_RE.findall(text)


def extract_code(solution):
    """Return ONLY the content of the FIRST fenced code block, or the whole text if there is no fence.

    Rules, all visible:
      * a fence opens on ``` (optionally followed by a language name)
      * a fence closes on the next ```
      * if no fence is found, fall back to the raw text (better than dropping the example)
    """
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*?)```", solution, re.S)
    if m:
        return m.group(1)
    m = re.search(r"```[a-zA-Z0-9_+-]*\s(.*?)```", solution, re.S)
    if m:
        return m.group(1)
    # no fence: drop any stray backticks and take the text as-is
    return solution


def doc_text(p, s):
    """The text of one training document. THIS IS THE SWITCH that DOC_MODE controls.

    CORRECTION (2026-10-10): DOC_MODE="clean" used to fall through to the "both" branch, so a
    CODE_DOC=clean run built a corpus byte-identical to "both" — the tag said clean while the
    content was fences+prose. The only reason the v32000 clean corpus was ever actually clean is
    that build_code_clean.py wrote its own cache directly. verify_clean_cache.py caught it as
    "wrong corpus — do not train"; datum: v6000_clean == v6000_both ids: True (19,505,365 tokens).
    """
    if DOC_MODE == "clean":
        return extract_code(s)
    return s if DOC_MODE == "code" else (p + "\n\n" + s)


def build(max_pairs=int(os.environ.get("CODE_MAX_PAIRS", "60000"))):
    tag = f"v{VOCAB_CAP}_{DOC_MODE}"
    ci = os.path.join(CACHE, f"code_ids_{tag}.npy")
    cv = os.path.join(CACHE, f"code_vocab_{tag}.json")
    cb = os.path.join(CACHE, f"code_bounds_{tag}.npy")
    cs = os.path.join(CACHE, f"code_split_{tag}.npy")
    cm = os.path.join(CACHE, f"code_meta_{tag}.json")
    if all(os.path.exists(p) for p in (ci, cv, cb, cs, cm)):
        print(f"  cached: vocab {len(json.load(open(cv))):,}")
        return {"ids": np.load(ci), "vocab": json.load(open(cv)),
                "bounds": np.load(cb).tolist(), "split": np.load(cs).tolist(),
                **json.load(open(cm))}

    print("  loading python pairs from the licensed sources...")
    pairs = list(python_pairs(limit=max_pairs * 2))[:max_pairs]
    print(f"  {len(pairs):,} python problem/solution pairs")
    if not pairs:
        raise SystemExit("no code pairs found -- check data/code/")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())

    from collections import Counter
    freq = Counter()
    for i, (p, s) in enumerate(pairs):
        if i in val_idx:
            continue
        freq.update(tokenize(doc_text(p, s).lower()))
    vocab = ["<unk>"] + [w for w, _ in freq.most_common(VOCAB_CAP - 1)]
    vmap = {w: i for i, w in enumerate(vocab)}
    print(f"  vocab {len(vocab):,} built from TRAIN text only  (DOC_MODE={DOC_MODE})")

    all_ids, bounds, split = [], [], []
    for i, (p, s) in enumerate(pairs):
        text = doc_text(p, s)
        enc = [vmap.get(t, 0) for t in tokenize(text.lower())]
        if len(enc) < 32:
            continue
        bounds.append(len(all_ids))
        all_ids.extend(enc)
        split.append(1 if i in val_idx else 0)
    ids = np.asarray(all_ids, dtype=np.int64)
    bounds = np.asarray(bounds, dtype=np.int64)
    split = np.asarray(split, dtype=np.int8)
    meta = {"n_pairs": len(bounds), "vocab_cap": VOCAB_CAP}
    np.save(ci, ids)
    np.save(cb, bounds)
    np.save(cs, split)
    json.dump(vocab, open(cv, "w"))
    json.dump(meta, open(cm, "w"))
    print(f"  encoded {len(ids):,} tokens over {len(bounds):,} pairs")
    return {"ids": ids, "vocab": vocab, "bounds": bounds.tolist(), "split": split.tolist(), **meta}


def make_windows(ids, bounds, split, window, which, stride=None):
    if stride is None:
        stride = max(1, window // 2)
    X, Y = [], []
    for j in range(len(bounds) - 1):
        if split[j] != which:
            continue
        a, b = bounds[j], bounds[j + 1]
        seg = ids[a:b]
        for s in range(0, len(seg) - window - 1, stride):
            X.append(seg[s:s + window])
            Y.append(seg[s + window])
    return np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)


if __name__ == "__main__":
    d = build()
    v = d["vocab"]
    print(f"  total tokens: {len(d['ids']):,}   pairs: {d['n_pairs']:,}   vocab: {len(v):,}")
    print("  sample: " + " ".join(v[i] for i in d["ids"][:60]))
