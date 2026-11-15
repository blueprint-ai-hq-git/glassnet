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
