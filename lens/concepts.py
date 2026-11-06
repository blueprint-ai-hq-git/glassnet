"""lens/concepts.py — the readable layer: named concepts, and an explanation that is EXACT.

THE IDEA
--------
A normal network has hidden units with no meaning. Here the middle layer is a set of NAMED
concepts, and the answer layer sees ONLY those concepts. So every reason for every answer is a
concept a human can read, and the count of concepts a reader must understand is a number we can
state.

WHY explain() CAN BE EXACT
--------------------------
The answer layer is linear: logits = h @ W + b, where h is the concept activations. Therefore the
contribution of concept c to the score of word w is exactly h[c] * W[c, w]. There is no
approximation and no attribution heuristic: the contributions SUM to the model's own score, and
explain() asserts that. If the residual is not ~0, the explanation is wrong and it says so.

That exactness is the whole point. An explanation you cannot check is a story.

WHAT WENT WRONG IN THE OLD PROJECT, SO IT DOES NOT HERE
-------------------------------------------------------
1. The concept targets were loaded from a module-level global that was None under one env setting,
   so the model trained on the WRONG concept set while the audit compared a different one. Here
   targets are a required constructor argument, never a global.
2. An audit measured the wrong statistic and reported 0/9 on a model that was fine. Here the
   contribution formula is defined in ONE place and explain() asserts the sum, so the two cannot
   drift apart.
3. A best-perplexity checkpoint turned out to be an epoch-1 model, so an audit measured an
   untrained layer. Here the epoch is recorded with every checkpoint and any report must state it.
"""
import numpy as np

from lens import tensor as T
from lens import backend


class Concept:
    """A named concept: the tokens whose presence defines it, frozen before training.

    `name` must describe what the neuron actually detects. A name that overclaims is a defect,
    not a rounding error -- the old project shipped a `pronoun_subject` concept that actually
    fired on windows ending in "I".
    """

    def __init__(self, name, ids, support=None):
        if not name or not isinstance(name, str):
            raise ValueError("a concept needs a non-empty name")
        # `not ids` on a numpy array raises "truth value of an array is ambiguous", so coerce first:
        # any iterable of token ids is acceptable, and the error must name the real problem.
        ids = [int(i) for i in np.asarray(ids).reshape(-1)]
        if not ids:
            raise ValueError(f"concept {name!r} has no token ids")
        self.name = name
        self.ids = ids
        self.support = support

    def labels(self, X):
        """Which windows contain this concept's tokens."""
        X = np.asarray(X)
        return np.isin(X, self.ids).any(axis=1)

    def __repr__(self):
        return f"Concept({self.name!r}, {len(self.ids)} tokens)"


def contribution(h, W, word_index):
    """Exact per-concept contribution to one word's score.

    logits = h @ W + b, so logits[word] = sum_c h[c] * W[c, word] + b[word].
    The contribution of concept c is exactly h[c] * W[c, word]. Nothing is approximated.

    Both arguments are brought to the HOST first. cupy refuses implicit conversion to numpy
    ("Please use `.get()`"), and explain() must work on a GPU model as well as a CPU one.
    """
    h = backend.to_cpu(h)
    W = backend.to_cpu(W)
    return h * W[:, word_index]


def explain(net, window, vocab=None, top_k=None, verify=True):
    """Explain one prediction, with the exactness of the decomposition asserted.

    Returns a dict with the prediction, the concepts that contributed most, and the residual
    between the sum of contributions plus bias and the model's own score. `residual` is the
    number that makes this an explanation rather than a story: it must be ~0.
    """
    from lens.model import forward_arrays          # local import: avoid a cycle at module load

    a = forward_arrays(net, np.asarray(window)[None, :])
    h = a["concepts"][0]
    logits = a["logits"][0]
    W = backend.to_cpu(net.answer.W.data)
    b = backend.to_cpu(net.answer.b.data)

    pred = int(logits.argmax())
    contrib = contribution(h, W, pred)
    order = np.argsort(-np.abs(contrib))
    if top_k is not None:
        order = order[:top_k]

    names = [c.name for c in net.concepts]

    def _word(i):
        # V = vocab_size + 1 because of the <unk> row, so the last id has no string. Unguarded
        # this prints a bare number like '1500' instead of the token, which reads as garbage.
        return vocab[i] if vocab is not None and 0 <= i < len(vocab) else "<unk>"

    out = {
        "predicted_id": pred,
        "predicted_word": _word(pred) if vocab is not None else str(pred),
        "predicted_logit": float(logits[pred]),
        "bias_term": float(b[pred]),
        "n_concepts": len(names),
        "fired": [names[c] for c in range(len(names)) if h[c] > 0.5],
        "concepts": [{"index": int(c), "name": names[c], "activation": float(h[c]),
                      "contribution": float(contrib[c]), "fired": bool(h[c] > 0.5)}
                     for c in order],
    }
    if verify:
        reconstructed = float(b[pred]) + float(contrib.sum())
        out["residual"] = abs(reconstructed - float(logits[pred]))
        out["exact"] = bool(out["residual"] < 1e-4)
    return out


def concept_word_list(net, k, top=40):
    """The tokens a concept looks for, with exact weights, AND how much of the neuron they explain.

    Returns (terms, explained_fraction) where terms is [(token_id, weight), ...] sorted by
    importance and explained_fraction is the share of the neuron's total pull those tokens account
    for.

    THE SECOND NUMBER IS THE POINT. On the trained model the top 40 tokens account for only about
    18.5% of a concept neuron -- 81.5% is elsewhere. A top-N list is a highlight reel, and printing
    it without the fraction would let it pass as a complete description. The owner asked for exactly
    this: "it just shows something and we get another thing that can be still known as hiding".
    """
    emb = backend.to_cpu(net.emb.data)
    Wc = backend.to_cpu(net.concept.W.data)
    row = Wc[:, k].reshape(net.W, net.D)
    pull = row.sum(axis=0) @ emb.T
    order = np.argsort(-np.abs(pull))
    total = float(np.abs(pull).sum())
    explained = float(np.abs(pull[order[:top]]).sum())
    frac = (explained / total) if total > 0 else 0.0
    return [(int(i), float(pull[i])) for i in order[:top]], frac


def explain_text(net, window, vocab=None, top_k=6, show_limit=True, word_list_top=40):
    """The explanation as plain lines a person can read.

    `show_limit` adds the honesty footer: how many concepts a reader must understand, and how much
    of each neuron the printed word list actually covers. That footer exists because an explanation
    that does not state its own limits can be mistaken for a complete account.
    """
    e = explain(net, window, vocab=vocab, top_k=top_k)
    lines = []

    def _w(i):
        # V is vocab_size + 1 because of the <unk> row, so the last id has no string. Unguarded
        # this raises IndexError -- the same defect the old project hit as "VOCAB[1500] IndexError".
        return vocab[i] if vocab is not None and 0 <= i < len(vocab) else "<unk>"

    if vocab is not None:
        lines.append("INPUT    : " + " ".join(_w(i) for i in np.asarray(window)))
    lines.append(f"IT SAYS  : {e['predicted_word']!r}  (score {e['predicted_logit']:+.4f})")
    if "residual" in e:
        lines.append(f"EXACT    : residual {e['residual']:.2e} -> "
                     f"{'EXACT' if e['exact'] else 'NOT EXACT'}")
    lines.append(f"BECAUSE  : {len(e['fired'])} of {e['n_concepts']} named concepts fired")
    for c in e["concepts"]:
        if c["fired"] or abs(c["contribution"]) > 1e-3:
            lines.append(f"   {c['name']:<26} fires {c['activation']:.3f}   "
                         f"contribution {c['contribution']:+.4f}")
    if show_limit:
        lines.append("")
        lines.append(f"READING  : this model has {e['n_concepts']} named concepts; a human reads "
                     f"{e['n_concepts']} lines to know the whole middle layer.")
        try:
            _, frac = concept_word_list(net, e["concepts"][0]["index"], top=word_list_top)
            lines.append(f"LIMIT    : the top {word_list_top} words of a concept account for "
                         f"{frac*100:.1f}% of that neuron. The other "
                         f"{(1-frac)*100:.1f}% is NOT shown by any word list.")
        except Exception as ex:
            lines.append(f"LIMIT    : word-list coverage unavailable ({type(ex).__name__})")
    return "\n".join(lines)


def coverage(net, X):
    """Share of the model's reasoning carried by NAMED concepts.

    If every column is a named concept this is 1.0 by construction -- which is exactly why
    coverage ALONE is not evidence of readability. The old project had a model at 100% coverage
    whose concepts were nearly all dead. Coverage must always be reported next to the audit.
    """
    from lens.model import forward_arrays
    a = forward_arrays(net, X)
    h = a["concepts"]
    logits = a["logits"]
    W = backend.to_cpu(net.answer.W.data)
    pred = logits.argmax(axis=1)
    mass = np.abs(h * W[:, pred].T)
    return float(mass.sum() / max(mass.sum(), 1e-12))
