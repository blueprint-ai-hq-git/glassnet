# Matched 18-vs-18: the readable chat model wins on accuracy, perplexity, and explanation - and still loses on vocabulary

Date: 2026-11-14 20:58. The two arms are now MATCHED: `nodirect1024long_s0` (readable, direct pathway off) and `direct1024long_s0`
(direct on), both 18 epochs, d1024, seed 0, 600k windows of the full 294,225,090-token corpus, identical in every other respect.
Predictions for the control were written in `lens/runs/DIRECT18-PREDICTIONS.txt` before it ran. Logs
`logs/exposure/{nodirect1024long,direct1024long}_s0.log`.

## The control collapses; the readable arm climbs

```
epoch          1     3     6     9    12    15    18
readable VAL  10.02 15.82 18.64 19.95*  ...  21.45 21.48     (*9-epoch value at same step)
control  VAL  17.12 20.00 21.07 18.53 18.16 17.89 17.73
control  ppl    318   263   377  1370  7439 29943 54751
```

**The direct-path control memorises**: training accuracy reaches 70.29% (loss 0.2997) while validation accuracy falls from its epoch-6
peak of 21.07% to **17.73%** and validation perplexity explodes from 263 to **54,751.20** - a 208x blow-up. The readable arm improves
monotonically and sets its held-out best at the **last** epoch.

## All three endpoints, matched

```
endpoint                          readable    control    advantage
final epoch VAL accuracy           21.48%     17.73%    +3.75  readable
last-3-epoch mean VAL accuracy     21.41%     17.71%    +3.70  readable
held-out best (by position)        23.67%     23.43%    +0.24  readable
final VAL perplexity              253.02   54,751.20    readable 216x better
next-token, best checkpoint        23.15%     22.05%    +1.10  readable
distinct tokens, best checkpoint       78        226      -148  control
concept path carries the answer     83.4%       3.8%    readable by construction
concept-only accuracy              22.75%      2.45%    readable 9.3x
```

**On every accuracy endpoint and on perplexity the readable model is now ahead of its matched control, and it is the only one of the
two whose named concepts actually explain the answer (83.4% agreement versus 3.8%).** The single remaining deficit is vocabulary
breadth: **78 distinct tokens against the control's 226.**

## Predictions, scored

Control-side predictions, all three held:

- **P1 HOLDS**: predicted final VAL at or below 18.57% and perplexity at or above 987.76; got **17.73%** and **54,751.20**.
- **P2 HOLDS**: predicted its held-out best would not improve on 23.46% by more than 0.5; got **23.43%**.
- **P3 HOLDS**: predicted training accuracy well past 55.46% while validation degrades; training reached **70.29%** against 17.73%
  validation.

So the falsifier did **not** trigger: the 9-epoch control had not been stopped early in a way that flattered the readable arm. **The
readable arm's advantage at matched 18 epochs is real.**

## What this does and does not establish

- **Establishes**: at matched settings and matched epochs, forcing the model to answer through its concept vector costs nothing on
  accuracy, produces 216x better validation perplexity, and is the only configuration whose explanation covers its decision.
- **Does not establish** that the readable model is better in general: **it speaks in 78 distinct words against 226**, which is a real
  and shippable-relevant deficit, and it is **one seed**.
- **Does not remove** the concept-name finding from `HONESTY-TRADEOFF-READABLE.md`: the readable model's explanations are exact, but
  its concept neurons are *less* aligned with their names than the direct-path model's (held-out synonyms 0.5676 vs 0.7178).
- The control's best checkpoint (epoch 6) is what the 226-token figure comes from, which is the fair choice for a vocabulary
  comparison; the final-epoch control is worse on everything.

## The one open lever

Diversity, 78 against 226. The concept layer is half-saturated (13 of 36 concepts fire on a median window), and the widening design
with 24 measured-surviving candidates (36 -> 60 concepts) is prepared and waiting on the owner:
`docs/history/CONCEPT-WIDENING-DESIGN.md`.
