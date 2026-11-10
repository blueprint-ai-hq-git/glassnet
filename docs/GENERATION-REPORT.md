# What the architecture that learns actually says (2026-11-10)

The readable concept/answer model measurably learns conditional structure - 11.00% next-token accuracy with
one token of context, 31.50% with 24, against a 3.99% frequency baseline (`ARCH-COMPARISON.md`). So the
question that decides the direction is whether that turns into usable text. Generated with
`tools/generate_arch.py` from `gpu_d512_s0_best.npz`.

## Greedy (temperature 0)
```
hello , how are you ?     -> these these complex these factors , we have the following of the world of the world of the
what is your name ?       -> * * * * * * * * * * * * * * * * * *
the capital of france is  -> the ` ` ` ` ` ` ` ` ` ` ` ` ` class class class
two plus two equals       -> chapter chapter , such as the equation , we can set the following equation as follows : *
i feel sad because        -> the ` ` ` ` ` ` ` ` ` ` ` ` ` class ` class `
the sea is                -> the ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` class
```

## Sampled (temperature 0.8)
```
hello , how are you ?     -> through the first , including valuable point on the bits between the origin . * * step 2
what is your name ?       -> by these how that does these learned into slope expressions to practice its complex ; n , it
the capital of france is  -> 8 * chapter 1 . * * * * * * * * * * * * *
two plus two equals       -> states , which chapter 2 . * * 2 * * * * * * * * *
i feel sad because        -> it transaction by - levels nicotine concepts and saving back to handle t brand government addressing . however
the sea is                -> thai remain . # # # # omega the answer is : # # # #
```

## Verdict
Better than the recurrent core, which collapsed to one identical string for every prompt, but **still not
talk**, and it must not be described as talk:

1. **Local grammar appears:** "including valuable point on the bits between the origin", "which chapter 2",
   "we can set the following equation as follows", "however". Function-word structure is real.
2. **Global meaning does not:** content words are semantically unrelated to the prompt ("nicotine concepts" for
   sadness, "thai remain" for the sea).
3. **No factual recall:** "the capital of france is" produces no country.
4. **A symbol attractor dominates:** `*`, `#` and a backtick token take over within a few steps under both
   decodings. The corpus contains a lot of markdown, and the model has learned that these tokens are cheap.
5. The prompt's own words are echoed or ignored rather than answered ("the sea is" then "thai ... the answer
   is :").

## Consequence
Two architectures, one failure mode in the decoder: both end up looping on frequent symbol tokens. That makes
the symbol attractor the next thing to measure and fix, because it caps what any generation test can show:
a model can be at 31.50% next-token accuracy and still be unusable if greedy decoding walks into `*` and never
leaves. This is now the top open item, ahead of further architecture work.
