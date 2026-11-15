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


## CORRECTION, an hour later: the 944 tokens were junk breadth

I published "the readable model samples 944 distinct tokens, close to the 971 target types the data contains" as evidence that it is
not short of vocabulary. Then I measured what those tokens actually are, with `tools/chat_generate_compare.py` - generation from
held-out prompts, scored on objective properties:

```
checkpoint / mode        repetition   self_repeat   coverage(top-571 real)   KL(gen||targets)   stop_rate
readable lam 1.0 greedy      0.339        0.229             0.141                  22.5             0.417
readable lam 1.0 sample      0.321        0.260             0.083                  24.0             0.250
direct control  greedy       0.006        0.062             0.042                  24.2             0.000
direct control  sample       0.000        0.021             0.047                  24.2             0.000
```

`coverage` is the share of generated tokens that fall inside the 571-token set covering 90% of real targets. **The readable model's
sampled tokens are 8-14% realistic**, and its text degenerates into symbol runs and repeated turn-end tokens:

```
readable, sample : "<|im_end|><|im_end|>youwantyouransweranswer:########"
readable, greedy : "################"
readable, sample : "oilyflycostlyofeaglesisshelters,soheautomationofx-2}"
control,  sample : "moretooandknowledgeabout1peoplebecauseanalyzinganalyzingbefactory.then"
```

**So the 944-token figure was long-tail junk, not vocabulary.** The argmax set was narrow partly because argmax avoids that junk. The
honest statement is the opposite of what I published an hour ago: on a sampled basis the readable model's extra "vocabulary" is
mostly tokens real text never uses, and its degeneration mode is loops and premature turn-end, while the control degenerates into
repetitive word salad.

What that correction does NOT change: the accuracy comparison, the shipped lam 1.0 configuration, and the explanation exactness all
stand. What it does establish is that **generation quality is the real remaining gap for BOTH models**, and that distinct-token counts
are the wrong instrument for it - coverage, repetition and KL are the right ones, and both models score badly on all three.

The next experiment is therefore about decoding and generation quality, not vocabulary size:
`lens/runs/DECODING-FIX-PREDICTIONS.txt`.
