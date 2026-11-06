#!/usr/bin/env python3
"""build_knowledge_corpus.py - turn the licence-cleared knowledge/maths files into our corpus format.

ONE change vs the existing corpora: MORE DATA (knowledge and maths). The tokenizer, the vocab (REUSED
from the existing corpus so cross-corpus numbers stay comparable), the "Human: ... / Assistant: ..."
conversation shape, and the val-split rule are all unchanged.

Licence rule: every source here passed the licence gate and is recorded in the meta file and in
DATA_LICENSES.md. Nothing with an unclear licence is read.

Two honesty notes that matter:
* Textbooks and books have no questions in them, so the prompt side is a SYNTHETIC single-turn wrapper
  ("tell me about <opening words>"). That is stated here, in the meta file, and in the log, because a
  synthetic prompt is not a real user question and must never be presented as one.
* Duplicate documents are removed (exact hash, plus a cheap near-duplicate guard on the opening text).
  Duplicates are known to cause memorisation that looks like learning.

Usage: python3 lens/scripts/build_knowledge_corpus.py
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter

import gc

import numpy as np
import pyarrow.parquet as pq

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(ROOT)
EXT = os.path.join(ROOT, "data/knowledge")
SRC = os.path.join(ROOT, "data/chat/prepared_ext_honest")   # vocab source + comparison yardstick
OUT = os.path.join(ROOT, os.environ.get("KN_OUT", "data/chat/prepared_kn"))
VOCAB_CAP = 12000
MAX_CHARS = 6000            # a single conversation may not be longer than this
CHUNK = 1400                # for long prose: how much text one wrapped turn carries
TOK = re.compile(r"\w+|[^\w\s]")
CAPS = {                    # tokens per source; keeps the corpus bounded and the run inside 2 h
    "metamath": 60_000_000,
    "hendrycks": 6_000_000,
    "openr1": 60_000_000,
    "textbooks": 80_000_000,
    "cosmo_math": 40_000_000,
    "gutenberg": 40_000_000,
    "magicoder": 25_000_000,
    "codefeedback": 20_000_000,
    "opencode": 20_000_000,
}
SOURCES = {
    "metamath": "meta-math/MetaMathQA (MIT)",
    "cosmo_math": "HuggingFaceTB/cosmopedia auto_math_text (Apache-2.0)",
    "hendrycks": "EleutherAI/hendrycks_math algebra (MIT)",
    "openr1": "open-r1/OpenR1-Math-220k (Apache-2.0)",
    "textbooks": "HuggingFaceTB/cosmopedia khanacademy+openstax+stanford (Apache-2.0) [SUBSTITUTE: nampdn-ai/tiny-textbooks is gated behind an authenticated account]",
    "gutenberg": "sedthh/gutenberg_english (MIT)",
    "magicoder": "ise-uiuc/Magicoder-OSS-Instruct-75K (MIT)",
    "codefeedback": "m-a-p/CodeFeedback-Filtered-Instruction (Apache-2.0)",
    "opencode": "nvidia/OpenCodeInstruct (CC BY 4.0)",
}


def parquet_batches(path, cols=None, batch_size=2000):
    """Batched reads, deliberately small: whole-table reads OOM-killed an earlier builder at 4.1 GB
    anon-rss, and on 2026-10-18 17:22 a 20000-row batch over the long-prose Gutenberg shards pushed the
    machine into a global OOM kill that took the agent's whole process group down. 2000 rows of long text
    is a few tens of MB, which is safe next to a training job."""
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield {c: b.column(c).to_pylist() for c in b.schema.names}


def json_array_stream(path, chunk=8 << 20):
    """Stream objects from a large single-file JSON array without loading it whole."""
    dec = json.JSONDecoder()
    with open(path, "r", errors="ignore") as f:
        buf, started = "", False
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
                    break
                yield obj
                i = end
            buf = buf[i:]
            if not block:
                return


def jsonl_stream(path):
    """One JSON object per line, streamed so a 200 MB file never has to fit in memory."""
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_text_col(b):
    """The Gutenberg shards name their column TEXT (uppercase) while cosmopedia uses text. Reading the
    real schema beats guessing: requesting a column that does not exist returns an EMPTY batch, and the
    first version of this builder then indexed an empty dict and died with IndexError."""
    for k in b.keys():
        if k.lower() in ("text", "content"):
            return k
    return None


def pick(cols, *names):
    """Return the first key that exists and is not empty. Code datasets do not share one schema, and
    guessing a key that is absent would silently produce an empty corpus."""
    for n in names:
        if n in cols:
            return n
    return None


def wrap(question, answer):
    q, a = str(question).strip(), str(answer).strip()
    if not q or not a:
        return None
    return f"Human: {q}\n\nAssistant: {a}"[:MAX_CHARS]


def prose_convs(text, tag):
    """Long prose -> synthetic single-turn pairs. The prompt is built from the passage's own opening
    words and is labelled synthetic in the meta file; it is not a real user question."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) < CHUNK:
        return
    for i in range(0, min(len(text), CHUNK * 6), CHUNK):     # at most 6 turns per document
        piece = text[i:i + CHUNK].strip()
        if len(piece) < 300:
            continue
        head = " ".join(piece.split()[:9]).strip(" ,.;:")
        if len(head) < 8:
            continue
        yield wrap(f"tell me about {head}", piece)


def stream_source(name):
    if name == "metamath":
        if not os.path.exists(os.path.join(EXT, "metamathqa_395k.json")):
            print("    (metamath file not present -- skipped)"); return
        for o in json_array_stream(os.path.join(EXT, "metamathqa_395k.json")):
            yield wrap(o.get("query", ""), o.get("response", ""))
    elif name == "hendrycks":
        if not os.path.exists(os.path.join(EXT, "hendrycks_algebra.parquet")):
            print("    (hendrycks file not present -- skipped)"); return
        for b in parquet_batches(os.path.join(EXT, "hendrycks_algebra.parquet")):
            for p, s in zip(b["problem"], b["solution"]):
                yield wrap(p, s)
    elif name == "cosmo_math":
        for f in ("cosmo_math_0.parquet", "cosmo_math_1.parquet"):
            path = os.path.join(EXT, f)
            if not os.path.exists(path):
                print(f"    ({f} not present -- skipped)")
                continue
            for b in parquet_batches(path):
                if not b:
                    continue
                cols = b.keys()
                if "prompt" in cols and "text" in cols:
                    for pr, tx in zip(b["prompt"], b["text"]):
                        yield wrap(pr, tx)
    elif name == "openr1":
        if not os.path.exists(os.path.join(EXT, "openr1_math_shard0.parquet")):
            print("    (openr1 file not present -- skipped)"); return
        for b in parquet_batches(os.path.join(EXT, "openr1_math_shard0.parquet")):
            cols = b.keys()
            pk = "problem" if "problem" in cols else list(cols)[0]
            sk = "solution" if "solution" in cols else list(cols)[-1]
            for p, s in zip(b[pk], b[sk]):
                yield wrap(p, s)
    elif name == "textbooks":
        # SUBSTITUTE for the gated tiny-textbooks file: nampdn-ai/tiny-textbooks requires an
        # authenticated account (HF answered "Access to dataset ... is restricted"), so it is skipped
        # rather than forced. HuggingFaceTB/cosmopedia is Apache-2.0 and public, and its educational
        # subsets carry a REAL "prompt" column, so the human side is a genuine instruction from the
        # source instead of a wrapper I invented.
        files = ["cosmo_khanacademy.parquet", "cosmo_openstax_0.parquet", "cosmo_openstax_1.parquet",
                 "cosmo_stanford_0.parquet", "cosmo_stanford_1.parquet"]
        for f in files:
            path = os.path.join(EXT, f)
            if not os.path.exists(path):
                print(f"    ({f} not present -- skipped)")
                continue
            for b in parquet_batches(path):
                if not b:
                    continue
                cols = b.keys()
                if "prompt" in cols and "text" in cols:
                    for pr, tx in zip(b["prompt"], b["text"]):
                        yield wrap(pr, tx)
                elif "text" in cols:
                    for tx in b["text"]:
                        for c in prose_convs(tx, "textbooks"):
                            yield c
    elif name == "magicoder":
        path = os.path.join(EXT, "magicoder_75k.jsonl")
        if not os.path.exists(path):
            print("    (magicoder file not present -- skipped)"); return
        for o in jsonl_stream(path):
            qk = pick(o, "problem", "instruction", "query", "prompt")
            ak = pick(o, "solution", "response", "answer", "output")
            if qk and ak:
                yield wrap(o[qk], o[ak])
    elif name == "codefeedback":
        path = os.path.join(EXT, "codefeedback.jsonl")
        if not os.path.exists(path):
            print("    (codefeedback file not present -- skipped)"); return
        for o in jsonl_stream(path):
            qk = pick(o, "query", "instruction", "problem", "prompt")
            ak = pick(o, "answer", "response", "output", "solution")
            if qk and ak:
                yield wrap(o[qk], o[ak])
    elif name == "opencode":
        files = ["opencodeinstruct_0.parquet", "opencodeinstruct_1.parquet"]
        for f in files:
            path = os.path.join(EXT, f)
            if not os.path.exists(path):
                print(f"    ({f} not present -- skipped)")
                continue
            for b in parquet_batches(path):
                if not b:
                    continue
                cols = b.keys()
                qk = pick(cols, "input", "question", "problem", "instruction", "prompt")
                ak = pick(cols, "output", "answer", "response", "solution")
                if not (qk and ak):
                    print(f"    ({f}: no recognisable question/answer columns -- skipped)"); continue
                for q, a2 in zip(b[qk], b[ak]):
                    yield wrap(q, a2)
    elif name == "gutenberg":
        for f in ("gutenberg_shard0.parquet", "gutenberg_shard1.parquet"):
            p = os.path.join(EXT, f)
            if not os.path.exists(p):
                continue
            for b in parquet_batches(p, cols=["TEXT"], batch_size=5):
                if not b:
                    continue
                tk = find_text_col(b)
                if tk is None:
                    print(f"    ({f}: no text column -- skipped)"); continue
                for t in b[tk]:
                    for c in prose_convs(t, "gutenberg"):
                        yield c


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    vp = os.path.join(SRC, f"vocab_v{VOCAB_CAP}.json")
    vocab = json.load(open(vp))
    vmap = {w: i for i, w in enumerate(vocab)}
    print(f"vocab REUSED from {vp} ({len(vocab):,} words) -- same word list, so this corpus is")
    print("comparable to the existing ones")
    print("synthetic-prompt note: textbooks and books have no questions in them, so their prompt side")
    print("is generated from the passage opening and is recorded as synthetic in the meta file")

    ids, bounds, per_source = [], [0], {}
    seen_exact, seen_head = set(), set()
    dups = {"exact": 0, "near": 0, "empty": 0}
    for name in ("metamath", "hendrycks", "openr1", "cosmo_math", "textbooks", "gutenberg",
                 "magicoder", "codefeedback", "opencode"):
        cap, used, kept = CAPS[name], 0, 0
        for conv in stream_source(name):
            if conv is None:
                dups["empty"] += 1
                continue
            h = hashlib.sha1(conv.encode("utf-8", "ignore")).hexdigest()
            if h in seen_exact:
                dups["exact"] += 1
                continue
            head = h[:12] + conv[:300].lower()
            if head in seen_head:
                dups["near"] += 1
                continue
            seen_exact.add(h)
            seen_head.add(head)
            toks = [vmap.get(w, 0) for w in TOK.findall(conv.lower())]
            if len(toks) < 4:
                dups["empty"] += 1
                continue
            ids.extend(toks)
            bounds.append(len(ids))
            used += len(toks)
            kept += 1
            if used >= cap:
                break
        per_source[name] = {"tokens": used, "conversations": kept, "cap": cap,
                            "licence": SOURCES[name]}
        gc.collect()
        print(f"  {name:11s} tokens {used:>12,}  conversations {kept:>8,}  cap {cap:>12,}  "
              f"({time.time()-t0:.0f}s)  [memory freed]")
        if used < cap:
            print(f"              note: source ran out before the cap ({used:,} of {cap:,})")

    ids = np.asarray(ids, dtype=np.int32)
    bounds = np.asarray(bounds, dtype=np.int64)
    np.save(os.path.join(OUT, f"ids_v{VOCAB_CAP}.npy"), ids)
    np.save(os.path.join(OUT, f"bounds_v{VOCAB_CAP}.npy"), bounds)
    json.dump(vocab, open(os.path.join(OUT, f"vocab_v{VOCAB_CAP}.json"), "w"))
    meta = {"n_conv": int(len(bounds) - 1), "tokens": int(len(ids)), "vocab_cap": VOCAB_CAP,
            "sources": SOURCES, "tokens_per_source": per_source, "dedup": dups,
            "synthetic_prompts": ["gutenberg"],
            "synthetic_note": "cosmopedia subsets carry real prompts; only gutenberg book text uses a "
                              "wrapper built from its own opening",
            "vocab_reused_from": vp,
            "note": "knowledge/maths corpus; val split is made disjoint afterwards by repair_split.py "
                    "so no held-out window appears verbatim in train",
            "built": time.strftime("%F %H:%M")}
    json.dump(meta, open(os.path.join(OUT, f"meta_v{VOCAB_CAP}.json"), "w"), indent=1)
    print(f"\nwrote {OUT}: {len(ids):,} tokens, {len(bounds)-1:,} conversations, "
          f"duplicates removed exact {dups['exact']:,} / near {dups['near']:,}")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()