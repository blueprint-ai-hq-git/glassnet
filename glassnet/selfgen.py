"""glassnet/selfgen.py — generate text with our OWN model, then turn it into training data.

Nothing here downloads anything or uses outside weights. The teacher is a model we trained from
scratch; the "data" it produces is its own continuation of real contexts.

The pieces:
    generate(net, seeds, n_new, temperature)  -> token sequences (N, W + n_new)
    to_windows(seqs, win)                     -> (X, y) window/next-token pairs
    select(X, y, conf, rule, threshold)       -> the pre-registered selection
    corpus_stats(X, y, vocab)                 -> diversity + repetition, so "repetitive" is a
                                                 measurement, not a worry
"""
import numpy as np


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def generate(net, seeds, n_new=32, temperature=1.0, rng=None, batch=256):
    """Autoregressively continue each seed window with the model's own samples.

    seeds : (N, W) int array of real contexts
    returns (N, W + n_new) int array
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    W = seeds.shape[1]
    out = np.array(seeds, dtype=np.int64)
    for _ in range(n_new):
        # only the last W tokens are the model's input window -- passing the whole growing
        # sequence is a shape bug (it asked the concept layer for W*D of a longer row)
        logits = np.asarray(net.forward(out[:, -W:])["logits"].data) / max(temperature, 1e-6)
        nxt = np.empty((len(out),), dtype=np.int64)
        for i in range(len(out)):
            p = _softmax(logits[i])
            nxt[i] = int(rng.choice(len(p), p=p))
        out = np.concatenate([out, nxt[:, None]], axis=1)
    return out


def to_windows(seqs, win=8):
    """Slice generated sequences into (context -> next token) pairs, the same shape the models
    already train on. Only positions that were actually GENERATED are used as targets, so the
    student never trains on the real seed as if the teacher had produced it."""
    X, y = [], []
    for s in seqs:
        for i in range(len(s) - win):
            if i + win < len(s):
                X.append(s[i:i + win])
                y.append(s[i + win])
    return np.asarray(X, dtype=np.int64), np.asarray(y, dtype=np.int64)


def teacher_confidence(net, X, y, batch=512):
    """The teacher's own probability for the true next token -- used by the `conf` selection."""
    out = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), batch):
        logits = np.asarray(net.forward(X[s:s + batch])["logits"].data)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        out[s:s + batch] = p[np.arange(len(p)), y[s:s + batch]]
    return out


def select(X, y, conf, rule="all", threshold=0.5, quantile=0.5):
    """The pre-registered selection rules from TARGET-F.md section 2. Nothing else is applied.

    `conf` uses the ORIGINAL fixed threshold (>= 0.5). The smoke test showed it admits 0 of 288
    generated windows on this model, so TARGET-F.md carries a dated amendment to the quantile
    rule below -- recorded before any arm was run, not after.
    """
    if rule == "all":
        return X, y, np.ones(len(X), dtype=bool)
    if rule == "conf":
        keep = conf >= threshold
        return X[keep], y[keep], keep
    if rule == "conf50":
        cut = np.quantile(conf, 1.0 - quantile)
        keep = conf >= cut
        return X[keep], y[keep], keep
    raise ValueError(f"unknown selection rule {rule!r}")


def corpus_stats(X, y, vocab, real_X=None):
    """Measurable corpus health, so the collapse warning is a number rather than a feeling."""
    flat = np.concatenate([X.ravel(), y])
    vals, counts = np.unique(flat, return_counts=True)
    top = np.argsort(-counts)[:10]
    stats = {
        "n_windows": int(len(X)),
        "n_distinct_tokens": int(len(vals)),
        "top10_token_share": float(counts[top].sum() / len(flat)),
        "top10_tokens": [vocab[int(vals[i])] if vals[i] < len(vocab) else "<PAD>" for i in top],
        "target_entropy": float(-np.sum((counts / counts.sum()) * np.log(counts / counts.sum()))),
        "mean_abs_delta_between_windows": float(
            np.abs(np.diff(X[:, -1].astype(np.int64))).mean()) if len(X) > 1 else 0.0,
    }
    if real_X is not None:
        rflat = real_X.ravel()
        rv, rc = np.unique(rflat, return_counts=True)
        stats["real_n_distinct_tokens"] = int(len(rv))
        stats["real_target_entropy"] = float(-np.sum((rc / rc.sum()) * np.log(rc / rc.sum())))
        stats["real_top10_token_share"] = float(
            np.sort(rc)[::-1][:10].sum() / len(rflat))
    return stats
