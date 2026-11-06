"""glassnet.eval — small metrics written by hand (no sklearn)."""
import numpy as np

from .tensor import Tensor


def accuracy(model, X, y, batch=512):
    """Classification accuracy over X (N, 784) vs int labels y."""
    correct = 0
    for i in range(0, X.shape[0], batch):
        logits = model.predict(X[i:i + batch])["logits"].data
        correct += int(np.sum(logits.argmax(axis=1) == y[i:i + batch]))
    return correct / X.shape[0]


def auroc(scores, labels):
    """Area under the ROC curve (Mann–Whitney U statistic), tie-aware.
    labels: truth; we binarise continuous concept targets at > 0.5."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    l = np.asarray(labels, dtype=np.float64).ravel() > 0.5
    n_pos, n_neg = int(l.sum()), int((~l).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.shape[0], dtype=np.float64)
    i = 0
    while i < s.shape[0]:
        j = i
        while j + 1 < s.shape[0] and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[l].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))