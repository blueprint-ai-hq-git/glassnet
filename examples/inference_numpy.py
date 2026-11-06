"""Standalone inference for a GlassNet checkpoint - numpy only, no framework.

Replicates lens/model.py forward exactly (verified against
lens.api.load(...).forward by tools/check_inference_example.py: max |logit
difference| 0.0 on random windows):

    x      = emb[ids] flattened over the W-token window
    h      = sigmoid(concept(x))                  # the named concept layer
    logits = answer(h) + relu(x @ dmix) @ emb.T   # the tied position-aware readout

A vocabulary is NOT interchangeable between checkpoints: the checkpoint stores V
and the word order that belongs to it, so this file REFUSES a vocab whose length
!= V instead of silently decoding every id into the wrong word (the two 12k
vocabs in this repo disagree at 11,991 of 12,000 positions).

Usage: python3 examples/inference_numpy.py [ckpt] [vocab] [prompt]
"""
import json
import re
import sys

import numpy as np

DEFAULT_CKPT = "lens/runs/chat/gpu_best_s1_best.npz"
DEFAULT_VOCAB = "data/chat/prepared_mix_ctrl_honest/vocab_v12000.json"
DEFAULT_PROMPT = "Human: hello , how are you ?\n\nAssistant:"


class GlassNet:
    """Forward-only reader for one checkpoint + its own vocabulary."""

    def __init__(self, ckpt, vocab_path):
        z = np.load(ckpt, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        self.D, self.W, self.V = int(meta["D"]), int(meta["W"]), int(meta["V"])
        if not (meta.get("direct") and str(meta.get("direct_mode")) == "tied"):
            raise SystemExit("this example implements the tied direct readout only; checkpoint says "
                             "direct=%r mode=%r" % (meta.get("direct"), meta.get("direct_mode")))
        self.emb = z["emb"].astype(np.float32)        # (V, D)
        self.cW, self.cb = z["concept.W"].astype(np.float32), z["concept.b"].astype(np.float32)
        self.aW, self.ab = z["answer.W"].astype(np.float32), z["answer.b"].astype(np.float32)
        self.dmix = z["dmix"].astype(np.float32)      # (W*D, D)
        self.words = json.load(open(vocab_path))
        if len(self.words) != self.V:
            raise SystemExit("vocab %s has %d words but the checkpoint expects V=%d: this vocabulary does "
                             "not belong to this checkpoint (refusing rather than printing wrong words)"
                             % (vocab_path, len(self.words), self.V))
        self.stoi = {w: i for i, w in enumerate(self.words)}
        self.unk = self.stoi.get("<unk>", 0)
        self.K = int(self.cW.shape[1])

    def encode(self, text):
        """The trainer's own tokenizer (lens/scripts/train_chat.py, 'CAN IT TALK?' block)."""
        toks = re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", text.lower())
        return [self.stoi.get(t, self.unk) for t in toks], toks

    def word(self, i):
        return self.words[i] if 0 <= i < len(self.words) else "<unk>"

    def logits(self, window_ids):
        ids = np.asarray(window_ids, dtype=np.int64)[-self.W:]
        if len(ids) < self.W:                         # the trainer left-pads short contexts with id 0
            ids = np.concatenate([np.zeros(self.W - len(ids), dtype=np.int64), ids])
        x = self.emb[ids].reshape(1, self.W * self.D)  # (1, W*D)
        h = 1.0 / (1.0 + np.exp(-(x @ self.cW + self.cb)))   # (1, K) concept layer
        lg = h @ self.aW + self.ab                          # (1, V) answer layer
        lg = lg + np.maximum(x @ self.dmix, 0.0) @ self.emb.T  # tied position-aware path
        return lg[0], h[0]

    def greedy(self, prompt, n_new=20, rep_penalty=1.15):
        """Greedy decoding with the trainer's repetition penalty (the trainer's GREEDY line)."""
        ids, _ = self.encode(prompt)
        cur, used, gen = list(ids), {}, []
        for _ in range(n_new):
            lg, _ = self.logits(cur)
            z = lg.astype(np.float64)
            for tok, cnt in used.items():
                z[tok] -= np.log(rep_penalty) * cnt
            nxt = int(z.argmax())
            gen.append(self.word(nxt))
            used[nxt] = used.get(nxt, 0) + 1
            cur.append(nxt)
        return " ".join(gen)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    vocab = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOCAB
    prompt = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PROMPT

    m = GlassNet(ckpt, vocab)
    print(f"checkpoint {ckpt}: V={m.V} D={m.D} K={m.K} window={m.W} readout=tied-direct")
    print(f"vocab {vocab}: {len(m.words)} words (matches V)")
    ids, toks = m.encode(prompt)
    print("prompt tokens: %s -> %s" % (toks, ids))
    lg, h = m.logits(ids)
    top = np.argsort(-lg)[:5]
    print("top-5 next tokens:", [(m.word(int(i)), round(float(lg[i]), 2)) for i in top])
    print("concept layer: mean activation %.4f, max %.4f, >0.5 count %d/%d"
          % (h.mean(), h.max(), int((h > 0.5).sum()), m.K))
    print("greedy continuation:", m.greedy(prompt))
