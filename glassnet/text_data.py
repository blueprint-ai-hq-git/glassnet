"""glassnet.text_data — Banking77 (cc-by-4.0) tiny text task, from scratch.

Pipeline (all own code, no sklearn): download -> csv parse -> lowercase
tokenize -> vocab from TRAIN only -> log-count bag-of-words (dim 2048).
Splits: the author CSV carries no official split, so we carve stratified
train 80 / val 10 / test 10 (seed 1234) — TEST quarantined. Vocab + concept
normalisation see TRAIN only (no leakage).
"""
import os
import re
import csv
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "banking77", "banking77.csv")
CACHE = os.path.join(DATA_DIR, "loop_banking77.npz")
SPLIT_SEED = 1234
VOCAB_SIZE = 2048
MIN_DOCS = 3                      # token must appear in >=3 train docs to enter vocab

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def _log(*a):
    msg = " ".join(str(x) for x in a)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", "console.log"), "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def _download(urls, dest, expect_bytes_hint=0):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "glassnet-r3"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if expect_bytes_hint and len(data) < expect_bytes_hint // 2:
                _log(f"[banking77] {url}: too small ({len(data)} B) — next")
                continue
            with open(dest, "wb") as f:
                f.write(data)
            _log(f"[banking77] downloaded {len(data)} B <- {url}")
            return True
        except Exception as e:
            _log(f"[banking77] {url} failed: {e}")
    return False


def _load_rows():
    """Official parquet split from HF (train 10,003 / test 3,080) when
    pyarrow is available; CSV from the author repo as fallback."""
    parq = [os.path.join(DATA_DIR, "banking77", f) for f in
            ("train.parquet", "test.parquet")]
    if not all(os.path.exists(p) for p in parq):
        ok = _download([
            "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/train.parquet",
        ], parq[0], expect_bytes_hint=100_000)
        if ok:
            ok2 = _download([
                "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/data/test.parquet",
            ], parq[1], expect_bytes_hint=50_000)
            if not ok2:
                raise RuntimeError("banking77 test parquet download failed")
    if all(os.path.exists(p) for p in parq):
        import pyarrow.parquet as pq
        texts, labels, splits = [], [], []
        for path, tag in zip(parq, ("train", "test")):
            t = pq.read_table(path)
            d = t.to_pydict()
            col = "text" if "text" in d else list(d)[0]
            lb = "label" if "label" in d else list(d)[1]
            for txt, lab in zip(d[col], d[lb]):
                texts.append(str(txt))
                labels.append(str(lab))
                splits.append(tag)
        return texts, labels, splits
    if not os.path.exists(CSV_PATH):
        ok = _download([
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        ], CSV_PATH, expect_bytes_hint=900_000)
        if not ok:
            raise RuntimeError("banking77 download failed on all URLs")
        ok2 = _download([
            "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        ], os.path.join(DATA_DIR, "banking77", "test.csv"),
            expect_bytes_hint=300_000)
        if not ok2:
            raise RuntimeError("banking77 test csv download failed")
    if os.path.exists(os.path.join(DATA_DIR, "banking77", "test.csv")):
        texts, labels, splits = [], [], []
        for path, tag in ((CSV_PATH, "train"),
                          (os.path.join(DATA_DIR, "banking77", "test.csv"),
                           "test")):
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and len(header) >= 2 and \
                        header[0].strip().lower() not in ("text", "category",
                                                          "label"):
                    # headerless file -> first row is data
                    if header:
                        row0 = header
                    else:
                        row0 = None
                else:
                    row0 = None
                rows = ([row0] if header and len(header) >= 2 and
                        header[0].strip().lower() not in
                        ("text", "category", "label") else []) + list(reader)
            for r in rows:
                if len(r) >= 2:
                    texts.append(r[0])
                    labels.append(r[1])
                    splits.append(tag)
        return texts, labels, splits
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) >= 2]
    h = {name.lower().strip(): i for i, name in enumerate(header)}
    tcol = next((h[k] for k in ("text", "utterance") if k in h), 0)
    lcol = next((h[k] for k in ("category", "label", "intent") if k in h), 1)
    scol = next((h[k] for k in ("split", "type", "train", "test") if k in h), None)
    texts, labels, splits = [], [], []
    for r in rows:
        texts.append(r[tcol])
        labels.append(r[lcol])
        splits.append(r[scol] if scol is not None else "")
    return texts, labels, splits


def load_vocab():
    """Feature-column order of the Banking77 bag-of-words (frequency-sorted,
    TRAIN only). Rebuilds the cache if it predates vocab persistence."""
    if not os.path.exists(CACHE):
        return None
    z = np.load(CACHE, allow_pickle=True)
    if "vocab" not in z.files:
        load(force=True)            # deterministic rebuild, now with vocab
        z = np.load(CACHE, allow_pickle=True)
    return list(z["vocab"])


def load(force=False):
    """Returns (Xtr, ytr, Xval, yval, Xte, yte, texts_tr, texts_val, texts_te,
    label_names) — stratified 80/10/10, vocab + stats from TRAIN only."""
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE, allow_pickle=True)
        return (z["Xtr"], z["ytr"], z["Xval"], z["yval"], z["Xte"], z["yte"],
                z["texts_tr"], z["texts_val"], z["texts_te"], z["label_names"])

    texts, labels, splits = _load_rows()
    uniq = sorted(set(labels))
    y_all = np.array([uniq.index(lb) for lb in labels], dtype=np.int64)
    if len(uniq) != 77:
        _log(f"[banking77] WARNING: {len(uniq)} unique labels (expected 77)")
    _log(f"[banking77] {len(texts)} rows, {len(uniq)} intents")

    # official split if the CSV marks train/test
    has_official = any(s in ("train", "test") for s in splits)
    n = len(texts)
    if has_official:
        tr_idx = [i for i, s in enumerate(splits) if s == "train"]
        te_idx = [i for i, s in enumerate(splits) if s == "test"]
        # carve val from train (stratified 10%)
        rng = np.random.default_rng(SPLIT_SEED)
        va_idx = []
        rem = []
        for c in range(len(uniq)):
            ci = [i for i in tr_idx if y_all[i] == c]
            p = rng.permutation(len(ci))
            va_idx += [ci[j] for j in p[: len(ci) // 10]]
            rem += [ci[j] for j in p[len(ci) // 10:]]
        tr_idx, va_idx = rem, va_idx
    else:
        rng = np.random.default_rng(SPLIT_SEED)
        tr_idx, va_idx, te_idx = [], [], []
        for c in range(len(uniq)):
            ci = [i for i in range(n) if y_all[i] == c]
            p = rng.permutation(len(ci))
            ntr, nva = int(0.8 * len(ci)), int(0.1 * len(ci))
            tr_idx += [ci[j] for j in p[:ntr]]
            va_idx += [ci[j] for j in p[ntr:ntr + nva]]
            te_idx += [ci[j] for j in p[ntr + nva:]]
    _log(f"[banking77] split: train {len(tr_idx)} / val {len(va_idx)} / "
         f"test {len(te_idx)} (official markers: {has_official})")

    # vocab from TRAIN only
    tr_tokens = [tokenize(texts[i]) for i in tr_idx]
    from collections import Counter
    cnt = Counter()
    seen = set()
    for tk in tr_tokens:
        for t in set(tk):
            seen.add(t)
    for t in seen:
        c = sum(1 for tk in tr_tokens if t in tk)
        if c >= MIN_DOCS:
            cnt[t] = c
    vocab = [t for t, _ in cnt.most_common(VOCAB_SIZE)]
    if len(vocab) < VOCAB_SIZE:
        _log(f"[banking77] WARNING: vocab {len(vocab)} < {VOCAB_SIZE}")
    vmap = {t: i for i, t in enumerate(vocab)}

    def bow(idx_list):
        X = np.zeros((len(idx_list), len(vocab)), dtype=np.float32)
        for i, ix in enumerate(idx_list):
            for t in tokenize(texts[ix]):
                j = vmap.get(t)
                if j is not None:
                    X[i, j] += 1.0
        return np.log1p(X)                    # log-count features

    Xtr, Xval, Xte = bow(tr_idx), bow(va_idx), bow(te_idx)
    ytr = y_all[tr_idx].astype(np.int64)
    yval = y_all[va_idx].astype(np.int64)
    yte = y_all[te_idx].astype(np.int64)
    t_tr = [texts[i] for i in tr_idx]
    t_va = [texts[i] for i in va_idx]
    t_te = [texts[i] for i in te_idx]

    np.savez(CACHE, Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval, Xte=Xte, yte=yte,
             texts_tr=np.array(t_tr, dtype=object),
             texts_val=np.array(t_va, dtype=object),
             texts_te=np.array(t_te, dtype=object),
             label_names=np.array(uniq, dtype=object),
             vocab=np.array(vocab, dtype=object))
    _log(f"[banking77] cached -> {CACHE} (dim {Xtr.shape[1]})")
    return (Xtr, ytr, Xval, yval, Xte, yte, t_tr, t_va, t_te, uniq)