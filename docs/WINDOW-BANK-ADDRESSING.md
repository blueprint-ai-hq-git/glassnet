

---

# THE WINDOW BANK'S LIMIT IS ADDRESSING, NOT RETENTION - MEASURED, NOT ARGUED (2026-11-19 15:10)

Before building a delayed-recall probe for the readable window bank (the sharp instrument that separates "memory works" from "memory does nothing" by ~97
points), I read its write and read paths. The read is:

```python
q = h @ p["Wq"]                      # query from the current state
k = h @ p["Wk"]                      # key from the state at write time
s = np.einsum("bkh,bh->bk", Km, q) / np.sqrt(H)
a = softmax(s)                        # over the K slots that are filled
r = a @ C                             # C holds REAL TOKEN EMBEDDINGS
out = h + tanh(r @ p["Wo"])
```

**There is no position or age term anywhere in the read.** A slot's index `j = t % K` exists in the implementation but is invisible to the attention. So the
only way to select a specific slot is content similarity between the current query state and that slot's key.

## Why that matters, with numbers

Retention is not the problem - the ring buffer replaces a slot WHOLLY (`C[j] = emb[toks[t]]`) and nothing decays, so a fact survives exactly K-1 steps. The
question is whether a blend-dominated read can return the right token. Measured on the **real trained embedding matrix** from `win3kwinoff_K0_ep3000.npz`
(V=12002, D=128), sampling K distinct tokens and asking whether the nearest vocabulary entry to their mean is one of the K:

```
K= 2   mean-of-K nearest token is one of the K in 100.0% of trials | dist 0.7952
K= 4                                           87.5%              | dist 0.9118
K= 8                                           19.5%              | dist 0.9139
K=16                                            5.0%              | dist 0.8943
    typical token-to-token distance: 1.60
```

**With a full 16-slot window, a uniform read returns a vector whose nearest real token is one of the 16 only 5% of the time.** So unless attention is sharply
peaked on exactly the right slot, the read is not "noisy", it is **wrong** - and peakedness requires a discriminator that the architecture does not provide.

## The consequence for the probe I was about to run

A delayed-recall probe asks the model to report the token from position 0 after D filler steps. For the window bank to pass, the query at step D must select
slot 0 - but the query is built from filler tokens, and there is nothing in it that says "position 0". **The probe would have failed by construction, and I
would have recorded that as "the window bank cannot hold 16 steps", which would be the wrong conclusion about the right mechanism.** Retention is exact;
addressing is absent.

## The concrete repair this points to

**Give the read a position/age term.** Either add a learned per-slot or per-age key (`s = (Km + age_emb[j]) @ q / sqrt(H)`) or concatenate the age to the
key. That is a small change, it makes arbitrary-delay addressing possible for the first time, and it is the next experiment rather than another probe of a
mechanism that cannot express the answer.

## What the window bank IS good for, stated plainly

Because the key comes from the state at write time, and the query from the current state, the bank naturally retrieves **recent or similar tokens** - which
is exactly what next-token prediction needs, and is why it is not measurably negative in that role. It is a KV-cache-like recent buffer, not a long-delay
memory store, and it should be judged on the job it can actually do until the age key is added.
