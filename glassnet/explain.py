"""glassnet/explain.py — exact explanations for the readable model.

WHY THIS IS EXACT, NOT A STORY
------------------------------
In this architecture the path from concepts to the answer is a single linear layer:

    logits = h @ W_answer + b_answer          (K concepts -> V words)

So the logit of any word is exactly the sum of one term per concept:

    logit[v] = b_answer[v] + sum_c ( h[c] * W_answer[c, v] )

Each term `h[c] * W_answer[c, v]` is therefore that concept's **exact contribution** to that
word's score. There is no approximation, no gradient surrogate, no attention to interpret.
`explain()` asserts the sum matches the model's own forward pass, and reports the residual --
if it were ever non-zero, the explanation would be wrong and this function says so.

Public API
----------
    explain(net, window, vocab, names, top_k=5)  -> dict
    format_explanation(exp)                      -> str (plain English)
"""
import numpy as np


def _word(vocab, i):
    return vocab[i] if 0 <= i < len(vocab) else "<PAD>"


def explain(net, window, vocab, names, top_k=5, firing=0.5, verify=True):
    """Explain one prediction.

    window : 1-D array of token ids (the input context)
    names  : list of concept names, one per concept column
    Returns a dict with the prediction, the concepts that fired, each fired concept's exact
    contribution to the predicted word, and (with verify=True) the reconstruction residual.
    """
    ids = np.asarray(window)[None, :]
    out = net.forward(ids)
    h = np.asarray(out["concepts"].data)[0]          # (K,)
    logits = np.asarray(out["logits"].data)[0]       # (V,)

    W = net.answer.W.data                            # (K, V)
    b = net.answer.b.data                            # (V,)

    pred = int(np.argmax(logits))
    contrib = h * W[:, pred]                         # (K,) exact contribution per concept

    order = np.argsort(-np.abs(contrib))
    fired = [c for c in order if h[c] > firing][:top_k]
    top = [c for c in order[:top_k]]

    exp = {
        "window_words": [_word(vocab, i) for i in window],
        "predicted_word": _word(vocab, pred),
        "predicted_id": pred,
        "predicted_logit": float(logits[pred]),
        "bias_term": float(b[pred]),
        "concepts": [
            {"col": int(c), "name": names[c] if c < len(names) else f"col{c}",
             "activation": float(h[c]),
             "contribution": float(contrib[c]),
             "fired": bool(h[c] > firing),
             "weight_to_predicted": float(W[c, pred])}
            for c in (fired if fired else top)
        ],
        "n_fired": int((h > firing).sum()),
        "all_activations": {names[c] if c < len(names) else f"col{c}": float(h[c])
                            for c in range(len(h))},
    }

    if verify:
        recon = float(b[pred] + np.sum(contrib))
        exp["reconstruction"] = recon
        exp["residual"] = float(abs(recon - logits[pred]))
        exp["exact"] = bool(exp["residual"] < 1e-4)
    return exp


def format_explanation(exp, width=78):
    """Plain English, readable on a phone."""
    L = []
    L.append("-" * width)
    L.append(f"INPUT      : {' '.join(exp['window_words'])}")
    L.append(f"IT SAYS    : '{exp['predicted_word']}'")
    if exp.get("exact") is not None:
        L.append(f"EXACTNESS  : contributions + bias rebuild the score to within "
                 f"{exp.get('residual', 0):.2e}  -> "
                 f"{'EXACT' if exp['exact'] else 'NOT EXACT (explanation wrong!)'}")
    L.append(f"FIRED      : {exp['n_fired']} concept(s) above 0.5")
    L.append("")
    L.append(f"  {'concept':<24} {'fires?':>7} {'act.':>7} {'contribution':>13}")
    for c in exp["concepts"]:
        L.append(f"  {c['name']:<24} {('YES' if c['fired'] else 'no'):>7} "
                 f"{c['activation']:>7.3f} {c['contribution']:>+13.4f}")
    L.append(f"  {'(bias, not a concept)':<24} {'':>7} {'':>7} {exp['bias_term']:>+13.4f}")
    L.append("")
    why = [c for c in exp["concepts"] if c["contribution"] > 0]
    against = [c for c in exp["concepts"] if c["contribution"] < 0]
    if why:
        L.append("WHY: because " + ", and ".join(
            f"{c['name']} fired ({c['activation']:.2f}) and pushed '{exp['predicted_word']}' "
            f"up by {c['contribution']:+.3f}" for c in why[:3]))
    if against:
        L.append("AGAINST: " + ", ".join(
            f"{c['name']} pushed it DOWN by {c['contribution']:+.3f}" for c in against[:3]))
    return "\n".join(L)


def concept_word_table(net, vocab, names, top_n=12):
    """Which words does each concept push up most? Readable model -> readable answer."""
    W = net.answer.W.data
    rows = []
    for c in range(W.shape[0]):
        top = np.argsort(-W[c])[:top_n]
        rows.append({"col": c, "name": names[c] if c < len(names) else f"col{c}",
                     "top_words": [_word(vocab, int(i)) for i in top],
                     "max_weight": float(W[c].max()), "min_weight": float(W[c].min())})
    return rows
