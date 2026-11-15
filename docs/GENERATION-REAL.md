# Generation re-measured through the real forward: more data helps everything, and the "dissociation" was an artefact (2026-11-15)

Predictions in `lens/runs/GENERATION-REAL-PREDICTIONS.txt`. Every number here comes from `lens.load()` + `net.forward()` - the model
itself - after the hand-built path was shown to differ by 131 logits (`docs/history/INSTRUMENT-WAS-BROKEN.md`). 24 prompts x 24 tokens,
greedy, same windows for every arm.

## The numbers, on the real forward

```
arm                                   repetition   self_repeat   realistic coverage   KL(gen||targets)   VAL accuracy
2M windows, 12 epochs (best accuracy)     0.271        0.159           0.316               17.845          23.47%
600k windows, 18 epochs (shipped)         0.475        0.370           0.260               19.748          21.08%
direct-path control                       0.030        0.036           0.240               19.049          17.73%
```

Samples, verbatim (greedy, 24 tokens):

```
2M/12ep : "thenumberofthenumberofthenumberofthenumberis:**:**"
600k    : "a.<|im_end|>intheworld.<|im_end|>,theother,andthetime.inthefirst,thenumb"
control : "<|im_end|>togetthispeople,theyneedtounderstandhowmuchtobeabletokeep.they"
```

**Every sample is made of real words.** There is no symbol soup: the withdrawn "junk token" narrative was entirely an artefact of the
broken forward.

## Prediction scorecard

- **R1 FAILED**: predicted repetition below 0.20; got **0.271** on the best arm.
- **R2 FAILED**: predicted coverage above 0.50; got **0.316**.
- **R3 FAILED**: predicted KL below 10; got **17.845**.
- **R4 PASSED**: the text is visibly word-like on all three arms, as predicted qualitatively.
- **FALSIFIER NOT TRIGGERED**: repetition came in at 0.271, below the 0.40 that would have confirmed the withdrawn conclusion.

## The finding this reverses, and it is the important one

`docs/history/ACCURACY-VS-GENERATION.md` claimed, from the broken instrument, that **3.3x more data improved accuracy while degrading
generation**. Through the real forward that is **backwards**:

```
                              accuracy    repetition   coverage    KL
600k windows, 18 epochs        21.08%       0.475       0.260     19.748
2M windows, 12 epochs          23.47%       0.271       0.316     17.845

change                        +2.39 pts    -0.204      +0.056     -1.903
```

**More data improved accuracy AND generation on every measure simultaneously.** There is no dissociation, no accuracy-versus-quality
trade-off, and the entire "the objective is misaligned" conclusion is withdrawn - it was a measurement artefact, not a property of the
model.

## What is real, and what is left

- **Real**: the readable model loops more than the control (0.271 vs 0.030) **but produces a distribution much closer to the corpus**
  (coverage 0.316 vs 0.240, KL 17.85 vs 19.05) and is +5.74 points more accurate. So the honest trade is *looping versus corpus
  fidelity*, not *accuracy versus quality*.
- **Real**: more data is the lever that moved both axes at once.
- **Open**: whether more data keeps helping - which is the cheapest experiment available and the next item in the queue.
- **Withdrawn earlier and still withdrawn**: everything measured through the hand-built path, including the breadth curve and its ceiling.

## Four million windows: accuracy and corpus fidelity keep improving, looping does not follow

Predictions in `lens/runs/MORE-DATA-2-PREDICTIONS.txt`. One change against the 2M/12-epoch arm: 4,000,000 training windows.

```
arm                    tokens/param   final VAL   final ppl   repetition   coverage   KL(gen||targets)   concepts
600k windows, 18 ep        1.31        21.08%      269.30       0.475       0.260         19.748         52/53
2M windows, 12 ep          4.36        23.47%      211.49       0.271       0.316         17.845         52/53
4M windows, 12 ep          8.72        24.57%      181.83       0.318       0.339         17.385         52/53
```

- **M1 PASSED**: final validation **24.57%**, above the 23.47% bar and the best accuracy measured all day.
- **M2 FAILED**: repetition rose to **0.318** instead of falling below 0.271.
- **M3 PASSED**: coverage **0.339** and KL **17.385**, both better than the 2M arm.
- **FALSIFIER NOT TRIGGERED**: it required repetition at or above 0.30 - which happened, 0.318 - **and** accuracy at or below 23.47%,
  which did not (24.57%). So the data lever has not saturated on accuracy, but it has stopped helping the looping.

## The shape of it, across three data points

```
                       600k      2M       4M
accuracy               21.08%   23.47%   24.57%     monotone, +3.49 in total
corpus coverage         0.260    0.316    0.339     monotone
KL to real text        19.748   17.845   17.385     monotone
repetition              0.475    0.271    0.318     NON-monotone
```

**Three of four axes improve monotonically with data; looping does not.** That is a cleaner statement of the real situation than anything
measured before the instrument was fixed: more data reliably improves prediction and corpus fidelity, and it moves looping around
(0.475 -> 0.271 -> 0.318) without controlling it.

Sample text from this arm, verbatim: `"thewaytodotheworldoftheworldoftheworldoftheworldoftheworldoftheworldofth"` - real words, still a
loop.

## What this implies for the next step

The data lever is not exhausted (accuracy and fidelity are still climbing) but it is **not a looping lever**. Looping is the one property
that has survived eight changes now, so it needs the training-protocol work - scheduled sampling or an unlikelihood term against repeated
tokens - rather than another token budget. Before that arm is built, the current best configuration needs the seed treatment every other
claim got: **24.57% is one seed**, and this project has twice seen a verdict reverse on a later seed.

## Three seeds at 4M: the configuration is stable, and my "looping got worse" verdict was seed noise

Predictions in `lens/runs/DATA4M-3SEED-PREDICTIONS.txt`. Seeds 1 and 2 at 4M/12ep, generation measured through the real forward.

```
seed   final VAL   final ppl   concepts   repetition   coverage   KL
 0      24.57%      181.83       52/53       0.318       0.339    17.385
 1      24.53%      181.21       50/53       0.252       0.373    16.423
 2      24.64%      180.08       52/53       0.256       0.316    17.825

mean    24.58%      181.04         --        0.275       0.343    17.211
sd       0.055       0.88          --        0.037       0.029     0.71
```

- **S1 PASSED**: every seed inside the predicted 24.17-24.97% band, spread of only **0.055 points**.
- **S2 PASSED**: all seeds far above the 21.50% floor.
- **S3 PASSED**: coverage above 0.30 and KL below 19.0 on every seed.
- **S4 deliberately made no prediction**; repetition came in at 0.318 / 0.252 / 0.256.
- **FALSIFIER NOT TRIGGERED**: validation never fell below 24.17%.

## The correction inside this result

Last wake I reported that repetition at 4M "rose to 0.318 instead of falling" and scored that prediction as failed. **Across three seeds the 4M repetition is 0.275 +/- 0.037, and seed 0's 0.318 is the outlier.** Against the 2M arm's single-seed 0.271, that is no difference at all - so the claim that more data made looping worse was **noise**, and with this noise level any single-seed repetition difference below about 0.08 is uninterpretable.

That is the same class of error as the hand-built forward, one level down: I compared two single measurements and treated the difference as a finding. **From here, every repetition comparison must be multi-seed or explicitly labelled single-seed and provisional.**

## The configuration this establishes

**4M windows / 12 epochs is the best measured configuration in the project**: 24.58% +/- 0.06 validation, perplexity 181.04, corpus coverage 0.343 +/- 0.03, KL 17.21, 50-52 of 53 concepts honest, and no accuracy regression across seeds. Against the previously shipped 600k/18-epoch model at 21.08% that is **+3.50 points**, with coverage up from 0.260 to 0.343.

Looping at this configuration is **0.275 +/- 0.037** - real, repeatable, and untouched by data. It is now the only defect left.
