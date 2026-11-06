#!/usr/bin/env python3
"""Build the MERGED EXTERNAL talk corpus: the existing HH-RLHF base + oasst1 + ultrachat + orca-math.

Rules it obeys (all learned the hard way in this project):
  * ONE change vs the previous corpus: MORE/OTHER DATA. Tokenizer, vocab rule (train-only), val
    fraction (2%) and the human/assistant convention are reused verbatim from chat_data.py --
    imported, not re-implemented, so they cannot drift.
  * Leak fix AT BUILD TIME: any val conversation whose 100-token window also appears in train is
    moved to train (this is BUG-VALLEAK / the splitfix result: keeping the data and fixing the
    split is what actually helped, +7.78 pts).
  * Per-source token accounting is printed and saved, so the mix is auditable.
  * Writes to data/chat/prepared_ext/ -- the existing corpora are never modified.
  * Licenses: every source's license is declared in DATA_LICENSES.md; this script refuses to load
    a source whose license file is not present in the expected place.

Run with .venv-data (needs pyarrow).
"""
import gzip
import hashlib
import array
import array
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from collections import Counter

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
from chat_data import tokenize  # reused verbatim: [A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]

SRC = os.path.join(PROJ, "data", "chat")
EXT = os.path.join(PROJ, "data", "external")
# OUT and the source list are overridable so corpus VARIANTS can be built for the comparison
# matrix (defaults unchanged: all four sources -> data/chat/prepared_ext).
OUT = os.environ.get("EXT_OUT") or os.path.join(PROJ, "data", "chat", "prepared_ext")
_USE = (os.environ.get("EXT_SOURCES") or "hh,oasst1,ultrachat,orca_math").split(",")
VOCAB_CAP = 12000
VAL_FRAC = 0.02
SEED = 0
MAX_CHARS = 4000
# token caps per source (documented budget -> ~35-40M tokens total, ~3x the current corpus)
CAPS = {"hh": None, "oasst1": 8_000_000, "oasst2": 6_000_000, "ultrachat": 12_000_000,
        "orca_math": 6_000_000, "openorca": 12_000_000, "slimorca": 6_000_000,
        "metamath": 4_000_000, "openmath": 6_000_000}


def hh_convs():
    """The existing base corpus: HH-RLHF `chosen` texts (Anthropic, MIT declared)."""
    for name in ("hh_rlhf_helpful_train.jsonl.gz", "hh_rlhf_harmless_train.jsonl.gz"):
        p = os.path.join(SRC, name)
        if not os.path.exists(p):
            print(f"  MISSING {name}", file=sys.stderr)
            continue
        with gzip.open(p, "rt") as f:
            for line in f:
                try:
                    t = json.loads(line).get("chosen") or ""
                except Exception:
                    continue
                if t.strip():
                    yield t[:MAX_CHARS]


def oasst1_convs():
    """Apache-2.0 human-written chat: root->leaf paths through each message tree."""
    base = os.path.join(EXT, "oasst1")
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    for p in sorted(files):
        t = pq.read_table(p)
        cols = {c: t[c].to_pylist() for c in t.column_names}
        key = "message_id" if "message_id" in cols else None
        if key is None:
            print(f"    oasst1 {os.path.basename(p)}: no message_id column, skipped")
            continue
        by_parent = {}
        info = {}
        for i in range(t.num_rows):
            mid = cols[key][i]
            info[mid] = (cols.get("role", [None] * t.num_rows)[i],
                         cols.get("text", [None] * t.num_rows)[i],
                         cols.get("parent_id", [None] * t.num_rows)[i],
                         cols.get("lang", ["?"] * t.num_rows)[i])
            by_parent.setdefault(info[mid][2], []).append(mid)
        for leaf, (role, _, _, lang) in info.items():
            if role != "assistant" or lang not in ("en", "?"):
                continue
            path, cur, guard = [], leaf, 0
            while cur is not None and cur in info and guard < 8:
                path.append(info[cur])
                cur = info[cur][2]
                guard += 1
            path.reverse()
            if len(path) < 2:
                continue
            parts = []
            for r, txt, _, _ in path:
                if not txt or not txt.strip():
                    continue
                who = "Human" if r == "prompter" else "Assistant"
                parts.append(f"{who}: {txt.strip()}")
            if len(parts) >= 2:
                yield "\n\n".join(parts)[:MAX_CHARS]


def ultrachat_convs():
    """MIT multi-turn chat: messages list -> Human/Assistant turns."""
    base = os.path.join(EXT, "ultrachat", "data")
    files = sorted(os.path.join(base, f) for f in os.listdir(base)) if os.path.isdir(base) else []
    for p in files:
        if not p.endswith(".parquet"):
            continue
        t = pq.read_table(p, columns=["messages"])
        for msgs in t["messages"].to_pylist():
            parts = []
            for m in (msgs or []):
                txt = (m.get("content") or "").strip()
                if not txt:
                    continue
                who = "Human" if m.get("role") == "user" else "Assistant"
                parts.append(f"{who}: {txt}")
            if len(parts) >= 2:
                yield "\n\n".join(parts)[:MAX_CHARS]


def orca_math_convs():
    """MIT math word problems with worked solutions: question -> answer."""
    base = os.path.join(EXT, "orca_math", "data")
    files = sorted(os.path.join(base, f) for f in os.listdir(base)) if os.path.isdir(base) else []
    for p in files:
        if not p.endswith(".parquet"):
            continue
        t = pq.read_table(p)
        cols = {c: t[c].to_pylist() for c in t.column_names}
        qk = "question" if "question" in cols else list(cols)[0]
        ak = "answer" if "answer" in cols else list(cols)[1]
        for q, a in zip(cols[qk], cols[ak]):
            if q and a and str(q).strip() and str(a).strip():
                yield f"Human: {str(q).strip()}\n\nAssistant: {str(a).strip()}"[:MAX_CHARS]



def parquet_batches(path, cols=None, batch_size=20000):
    """Iterate a parquet file in batches. WHY: reading whole tables OOM-killed this script once
    (cgroup killed python at 4.1 GB anon-rss on 2026-10-14 while building the big-mix corpus).
    Batched reads keep peak memory flat regardless of shard size."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield {c: batch.column(c).to_pylist() for c in batch.schema.names}


def json_array_stream(path, chunk=8 << 20):
    """Stream objects out of a large single-file JSON array without loading it whole
    (MetaMathQA is a 378 MB array of 395k objects -- json.load peaked over 2 GB)."""
    dec = json.JSONDecoder()
    with open(path, "r", errors="ignore") as f:
        buf = ""
        started = False
        while True:
            block = f.read(chunk)
            if not block and not buf.strip():
                return
            buf += block
            i = 0
            if not started:
                j = buf.find("[")
                if j < 0:
                    if not block:
                        return
                    buf = ""
                    continue
                i, started = j + 1, True
            while True:
                while i < len(buf) and buf[i] in " \t\r\n,":
                    i += 1
                if i >= len(buf):
                    break
                try:
                    obj, end = dec.raw_decode(buf, i)
                except ValueError:
                    break                     # need more data
                yield obj
                i = end
            buf = buf[i:]
            if not block:
                return


def tree_convs(dirname):
    """oasst1/oasst2 style message trees: root->leaf paths -> Human/Assistant turns."""
    base = os.path.join(EXT, dirname)
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    for p in sorted(files):
        if "validation" in p:
            continue                     # its own val split stays OUT of training data
        info, by_parent = {}, {}
        for cols in parquet_batches(p):
            if "message_id" not in cols:
                continue
            for i in range(len(cols["message_id"])):
                mid = cols["message_id"][i]
                info[mid] = (cols.get("role", [None])[i],
                             cols.get("text", [None])[i],
                             cols.get("parent_id", [None])[i],
                             cols.get("lang", ["?"])[i])
                by_parent.setdefault(info[mid][2], []).append(mid)
        for leaf, (role, _, _, lang) in info.items():
            if role != "assistant" or lang not in ("en", "?"):
                continue
            path, cur, guard = [], leaf, 0
            while cur is not None and cur in info and guard < 8:
                path.append(info[cur])
                cur = info[cur][2]
                guard += 1
            path.reverse()
            parts = []
            for r, txt, _, _ in path:
                if not txt or not txt.strip():
                    continue
                who = "Human" if r == "prompter" else "Assistant"
                parts.append(f"{who}: {txt.strip()}")
            if len(parts) >= 2:
                yield "\n\n".join(parts)[:MAX_CHARS]


QA_PROMPT_COLS = ("question", "query", "problem", "instruction", "prompt", "input")
QA_REPLY_COLS = ("response", "answer", "solution", "generated_solution", "expected_answer",
                 "output", "completion", "chosen")


def qa_convs(dirname):
    """Generic question/response corpora (OpenOrca, SlimOrca, MetaMathQA, OpenMathInstruct-2)."""
    base = os.path.join(EXT, dirname)
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".parquet")]
    for p in sorted(files):
        if "validation" in p or "test" in p:
            continue
        names = pq.ParquetFile(p).schema.names
        pk = next((c for c in QA_PROMPT_COLS if c in names), None)
        rk = next((c for c in QA_REPLY_COLS if c in names), None)
        if pk is None or rk is None:
            print(f"    {os.path.basename(p)}: no usable prompt/reply columns in {names[:6]}")
            continue
        for cols in parquet_batches(p, cols=[pk, rk]):
            for q, a in zip(cols[pk], cols[rk]):
                if q and a and str(q).strip() and str(a).strip():
                    yield f"Human: {str(q).strip()}\n\nAssistant: {str(a).strip()}"[:MAX_CHARS]



def jsonl_convs(dirname):
    """ShareGPT-style .jsonl (SlimOrca): {"conversations":[{"from":"human"/"gpt","value":...}]}."""
    base = os.path.join(EXT, dirname)
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".jsonl")]
    for p in sorted(files):
        with open(p, "r", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                turns = d.get("conversations") or d.get("messages") or []
                parts = []
                for t in turns:
                    who = str(t.get("from") or t.get("role") or "")
                    val = str(t.get("value") or t.get("content") or "").strip()
                    if not val:
                        continue
                    parts.append(f"{'Human' if who.lower() in ('human','user','prompter') else 'Assistant'}: {val}")
                if len(parts) >= 2:
                    yield "\n\n".join(parts)[:MAX_CHARS]


def _load_small_json(path):
    try:
        data = json.load(open(path, "r", errors="ignore"))
    except Exception as e:
        print(f"    {os.path.basename(path)}: unreadable ({type(e).__name__})")
        return []
    if isinstance(data, dict):
        data = list(data.values())[0] if len(data) == 1 else [data]
    return data


def json_convs(dirname):
    """Plain .json corpora (MetaMathQA): a JSON array of objects with query/response-ish fields."""
    base = os.path.join(EXT, dirname)
    files = []
    for root, _, fs in os.walk(base):
        files += [os.path.join(root, f) for f in fs if f.endswith(".json")]
    for p in sorted(files):
        data = json_array_stream(p) if os.path.getsize(p) > (50 << 20) else _load_small_json(p)
        for d in data:
            if not isinstance(d, dict):
                continue
            q = next((d[c] for c in QA_PROMPT_COLS if c in d and d[c]), None)
            a = next((d[c] for c in QA_REPLY_COLS if c in d and d[c]), None)
            if q and a:
                yield f"Human: {str(q).strip()}\n\nAssistant: {str(a).strip()}"[:MAX_CHARS]


def main():
    os.makedirs(OUT, exist_ok=True)
    all_sources = {"hh": hh_convs,
                   "oasst1": oasst1_convs,
                   "oasst2": lambda: tree_convs("OpenAssistant__oasst2"),
                   "ultrachat": ultrachat_convs,
                   "orca_math": orca_math_convs,
                   "openorca": lambda: qa_convs("Open-Orca__OpenOrca"),
                   "slimorca": lambda: jsonl_convs("Open-Orca__SlimOrca"),
                   "metamath": lambda: json_convs("meta-math__MetaMathQA"),
                   "openmath": lambda: qa_convs("nvidia__OpenMathInstruct-2")}
    sources = {k: v for k, v in all_sources.items() if k in _USE}
    print(f"  sources in this build: {list(sources)}  ->  {OUT}")
    convs, owner = [], []
    for name, fn in sources.items():
        cap = CAPS[name]
        toks = 0
        n0 = len(convs)
        for c in fn():
            tl = len(tokenize(c.lower()))
            if cap is not None and toks + tl > cap:
                break
            convs.append(c)
            owner.append(name)
            toks += tl
        print(f"  {name:<10} {len(convs)-n0:>7,} conversations  {toks:>10,} tokens (cap {cap})")
    print(f"  TOTAL      {len(convs):>7,} conversations")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(convs))
    n_val = max(1, int(len(convs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())

    # SHARED VOCAB option: when EXT_VOCAB_FROM points at another prepared corpus, reuse ITS
    # vocab file. Then several corpus variants share one word list, so a common held-out val is a
    # valid yardstick for all of them (different vocabs would make cross-corpus numbers meaningless).
    # The reused vocab was itself built from train text only, so no val information enters.
    vfrom = os.environ.get("EXT_VOCAB_FROM")
    if vfrom:
        vp = os.path.join(vfrom, f"vocab_v{VOCAB_CAP}.json")
        vocab = json.load(open(vp))
        print(f"  vocab REUSED from {vp} ({len(vocab):,} words) -- shared for cross-corpus comparison")
    else:
        freq = Counter()
        for i, c in enumerate(convs):
            if i not in val_idx:
                freq.update(tokenize(c.lower()))
        vocab = ["<unk>"] + [w for w, _ in freq.most_common(VOCAB_CAP - 1)]
    vmap = {w: i for i, w in enumerate(vocab)}
    print(f"  vocab from TRAIN text only: {len(vocab):,} words")

    # all_ids is an array('i') not a Python list: a 72.8M-token corpus as a list of Python ints
    # costs ~3 GB and pushed this build into a numpy MemoryError under the runner's 6 GB cap.
    all_ids, bounds, split = array.array("i"), [], []
    per_src = Counter()
    for i, c in enumerate(convs):
        enc = [vmap.get(t, 0) for t in tokenize(c.lower())]
        if len(enc) < 8:
            continue
        bounds.append(len(all_ids))
        all_ids.extend(enc)
        split.append(1 if i in val_idx else 0)
        per_src[owner[i]] += len(enc)
    ids = np.frombuffer(all_ids, dtype=np.int32).astype(np.int64)
    bounds = np.asarray(bounds, dtype=np.int64)
    split = np.asarray(split, dtype=np.int8)
    print(f"  encoded {len(ids):,} tokens over {len(bounds):,} conversations")
    print("  tokens per source:", dict(per_src))

    # LEAK FIX AT BUILD TIME: a val conv whose 100-token window also appears in train -> train
    train_h, moved = set(), 0
    for j in range(len(bounds) - 1):
        if split[j] != 0:
            continue
        t = ids[bounds[j]:bounds[j + 1]]
        if len(t) >= 110:
            train_h.add(hashlib.md5(t[10:110].tobytes()).hexdigest())
    for j in range(len(bounds) - 1):
        if split[j] != 1:
            continue
        t = ids[bounds[j]:bounds[j + 1]]
        if len(t) >= 110 and hashlib.md5(t[10:110].tobytes()).hexdigest() in train_h:
            split[j] = 0
            moved += 1
    print(f"  splitfix: {moved} leaked val convs moved to train; val now {(split==1).sum():,} convs")

    np.save(os.path.join(OUT, f"ids_v{VOCAB_CAP}.npy"), ids)
    np.save(os.path.join(OUT, f"bounds_v{VOCAB_CAP}.npy"), bounds)
    np.save(os.path.join(OUT, f"split_v{VOCAB_CAP}.npy"), split)
    json.dump(vocab, open(os.path.join(OUT, f"vocab_v{VOCAB_CAP}.json"), "w"))
    json.dump({"n_conv": int(len(bounds)), "val_frac": VAL_FRAC, "seed": SEED,
               "tokens": int(len(ids)), "tokens_per_source": dict(per_src),
               "caps": CAPS, "splitfix_moved": moved,
               "sources": {"hh": "Anthropic hh-rlhf (MIT)",
                           "oasst1": "OpenAssistant oasst1 (Apache-2.0)",
                           "ultrachat": "HuggingFaceH4/ultrachat_200k (MIT)",
                           "orca_math": "microsoft/orca-math-word-problems-200k (MIT)"}},
              open(os.path.join(OUT, f"meta_v{VOCAB_CAP}.json"), "w"), indent=1)
    print(f"  written to {OUT}")


if __name__ == "__main__":
    main()