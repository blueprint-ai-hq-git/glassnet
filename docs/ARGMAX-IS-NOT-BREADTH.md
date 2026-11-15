# The breadth ceiling was partly my measuring instrument: 364 at argmax, 944 when sampled (2026-11-15)

Two things happened in this block. The interface-width experiment failed its falsifier, and then a check of the breadth METRIC itself
found that most of today's breadth conclusions were measuring confidence rather than vocabulary.

## 1. Interface width: falsifier fired, and the direct path actively hurts

Predictions in `lens/runs/INTERFACE-WIDTH-PREDICTIONS.txt`. Readable 53-concept arm (lam 1.0) plus a low-rank direct path, which is the
one structural difference left against the control (53 interface scalars vs 1024):

```
arm                        effective dims   distinct tokens   final VAL   final ppl   concepts
readable, no direct path      10.09 of 53         364           21.08%      269.30      52/53
+ direct rank 8                5.25 of 53         359           18.75%      421.39      51/53
+ direct rank 32               4.18 of 53         297           18.79%      558.07      51/53
+ direct rank 128              4.08 of 53         347           19.50%      907.10      51/53
direct-path control            2.97 of 36         539           17.73%     54,751          --
```

**I1 FAILED** (rank 8 predicted above 380, got 359; rank 32 above 430, got 297; rank 128 above 470, got 347). **I2 FAILED** (accuracy
was predicted to hold at 21.08%; it fell to 18.75-19.50%). **I3 could not be scored: the explanation-cost step produced no output**, the
`tools/explain_verified.py --ckpt` invocation I flagged as unverified last wake does not accept that argument, so nothing was measured
there. Predictions said **FALSIFIER FIRES if breadth stays at or below 380 at rank 128: it did (347)**.

So interface width is not the constraint either - and adding it costs 1.6-2.3 points of accuracy, up to 67 tokens of the argmax
breadth and a 3.4x worse perplexity. **That is the fifth candidate explanation measured and rejected.**

## 2. The instrument was wrong: argmax diversity is not vocabulary breadth

All of today's breadth numbers used distinct **argmax** predictions. That measures how often a model commits to a rare token, which is a
statement about confidence, not about which words it can produce. Sampling the same distributions on the same 4,000 held-out windows:

```
checkpoint          argmax   sampled T=1.0   sampled T=0.8   sampled T=1.2   mean entropy (nats)
direct-path control   395        1,543            1,086            1,961            5.02
readable lam 1.0      364          944              769            1,120            2.05
readable lam 2.0      422          863              700            1,079            1.88
data itself           971 distinct target types occur in this sample
```

**The readable model produces 944 distinct tokens when sampled, against 364 at argmax - 2.6x more, and close to the 971 target types
the data actually contains.** So the model is not short of vocabulary. What it lacks is willingness to commit: its predictive
distribution is far sharper than the control's (entropy 2.05 vs 5.02 nats), so its argmax collapses onto a small set of safe tokens.

**Correction to earlier documents**: the "~422-token ceiling" framed in `docs/history/BREADTH-IS-SUPERVISION.md` and repeated tonight is a
property of the **argmax metric**, not a hard limit on what the model can say. The supervision dial does still move argmax diversity
(135 -> 422), and that remains a real, measured effect - but it must be described as "willingness to commit to a rare token", not as
"how many words the model knows".

## What survives, what is withdrawn

- **Survives**: readable beats matched control on accuracy at every operating point; the shipped configuration is lam 1.0 at
  21.13% +/- 0.05 across three seeds with 51-52 of 53 concepts honest; supervision strength trades accuracy (about half a point per
  step) for argmax diversity; and the honest concept-name transfer gap is unresolved.
- **Withdrawn**: any claim that the readable model "cannot talk widely" or that its vocabulary is capped near 422 words. On a sampled
  basis it reaches 944 of the 971 token types the held-out data uses.
- **Open**: whether the sampled output is *good* - diverse tokens are not automatically sensible text, and nothing here measured quality
  of generated text. That is the next experiment, and it is the honest one to run because it tests the thing a user would actually see.
