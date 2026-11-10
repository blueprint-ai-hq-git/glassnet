# A decode guard fixes the looping but not the meaning (2026-11-10)

The symbol attractor was the top open item: the readable model scores 31.50% next-token accuracy at 24 tokens of
context and still collapses into `*`, `#` and backtick runs within a few steps. Two readings of that:

  (a) the model's ranking is fine and unconstrained greedy decoding walks into a self-reinforcing loop, so a
      decode-time guard fixes the text;
  (b) the model genuinely prefers symbol tokens, so a guard only paints over a broken distribution.

Measured with `tools/decode_guard.py` on `gpu_d512_s0_best.npz`, 12 prompts, 24 tokens each, no training:

```
guard      symbol share   longest symbol run   distinct 4-grams
greedy          67.7%            15.0               39.3%
norepeat        57.6%             5.0              100.0%
symrun          50.7%             2.3               69.0%
both            46.5%             2.4              100.0%
```

## What the numbers say
The guards do exactly what they were built to do. `norepeat` takes distinct 4-grams from 39.3% to **100%** -
the looping is completely gone - and `symrun` cuts the longest symbol run from 15 tokens to 2.3.

And the text is still meaningless:

```
hello , how are you ?    -> these these complex these factors , we have the following of the world of the world
what is your name ?      -> * * i * * i = * * i ' s * * i '
the capital of france is -> the ` ` s the solution , the ` ` python class is : ` `
i feel sad because       -> the ` ` s the ` ` function the ` ` the ` class ` `
```

**Reading (b) is the correct one.** Even with repeats banned and symbol runs capped, the model still spends
**46.5%** of its emitted tokens on symbols, against 27.7% in the training corpus - about 1.7x the rate it was
shown. The preference is in the weights, not in the decoder. A guard removes the visible loop and leaves the
model wanting the same tokens; it converts one failure (repetition) into a less obvious one (grammatical-looking
noise salted with backticks).

## Consequence
Decoding tricks are off the list as a fix for this. The next lever has to act on the weights or the data:
1. the matched-split retrain now running is the first arm whose validation composition is honest, so it is the
   first place a real change in behaviour can be measured;
2. after that, class weighting or a masking penalty on pure-symbol tokens during training, measured against the
   symbol share of generation, not against accuracy alone;
3. the frequency baseline stays beside every number: a model that emits symbols at 1.7x the corpus rate is not
   reading the data, whatever its accuracy says.
