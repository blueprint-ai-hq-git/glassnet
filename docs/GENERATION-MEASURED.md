# Generation, measured properly: the readable model loops, but its words are MORE realistic than the control's (2026-11-15)

This is the third correction of the day and the first that moves in the readable model's favour. It exists because my previous two
readings used too few prompts and the wrong emphasis. Full grid: `lens/runs/DECODING-GRID.txt`, predictions written beforehand in
`lens/runs/DECODING-FIX-PREDICTIONS.txt`, tool `tools/chat_generate_compare.py`.

## The grid: 24 prompts x 24 generated tokens, two checkpoints, four decoders

```
checkpoint                    setting                                             repetition   self_repeat   realistic coverage   KL(gen||targets)   stop_rate
direct-path control           A greedy, no controls                                  0.040        0.091            0.123              22.05           0.000
direct-path control           B sample T=0.8                                         0.000        0.026            0.156              21.09           0.000
direct-path control           C B + top-k 50 + rep 1.3                               0.000        0.003            0.148              21.07           0.042
direct-path control           D C + no-repeat-3gram + end banned 8 steps            0.000        0.003            0.146              21.11           0.083
readable lam 1.0              A greedy, no controls                                  0.555        0.518            0.351              18.11           0.292
readable lam 1.0              B sample T=0.8                                         0.513        0.495            0.280              19.55           0.250
readable lam 1.0              C B + top-k 50 + rep 1.3                               0.498        0.487            0.203              21.33           0.250
readable lam 1.0              D C + no-repeat-3gram + end banned 8 steps            0.000        0.159            0.229              19.99           0.250
```

`realistic coverage` is the share of generated tokens inside the 571-token set that covers 90% of real targets. `KL` compares the
generated unigram distribution against the real target distribution.

## What this actually says - three findings, in order of importance

1. **The readable model's generated words are MORE realistic than the control's, on every decoder**: coverage 0.351 vs 0.123 at greedy,
   and 0.229 vs 0.146 even at the most constrained decoder. Its unigram distribution is also much closer to real text (**KL 18.11 vs
   22.05**). So the "junk breadth" reading I published an hour ago was wrong in emphasis: the tail exists, but the body of what it
   generates is more text-like than the control's.
2. **Its defect is LOOPING, and looping is a decoder problem**: repetition 0.555 under greedy, and the controls take it to **0.000**
   while coverage only falls 0.351 -> 0.229. The control barely loops at all (0.040 -> 0.000).
3. **Both models generate badly in absolute terms** - coverage of 0.12-0.35 against the 0.5+ a working model should reach, and KL of
   18-22 nats. Neither is close to usable text, and the honest statement is that generation quality, not vocabulary count, is where
   this project stands.

## Prediction scorecard

- **X1 PASSED**: under setting D repetition goes to **0.000** on the readable model (predicted below 0.05) and stays at **0.000** on
  the control (predicted below 0.02).
- **X2 FAILED narrowly**: predicted D would lift readable coverage only to 0.10-0.20; it reached **0.229**, just above the band.
- **X3 held as written**: the model with the better distributional numbers (coverage 0.351, KL 18.11) is also the one with the better
  accuracy (21.13% vs 17.73%), so the two instruments agree about which model knows the corpus better.
- **FALSIFIER NOT TRIGGERED**: coverage at D was 0.229, below the 0.30 that would have meant the decoder was the whole story.

## Method note, because two of my three readings today were sample-size artefacts

The earlier "8-14% realistic" figure came from **12 prompts**, this one from **24**. The wider sample moved coverage from 0.141 to 0.351
on the same checkpoint and decoder family. Generation metrics on this model are sensitive to prompt count at these sizes, so every
generation claim from now on states its prompt count alongside the number. That is the same lesson as the argmax-versus-sampled
correction, one level down: the instrument has to be specified, not just named.

## Where this leaves the work

The readable model is not vocabulary-poor and its words are not junk - it **loops**, and its absolute generation quality is low because it
is under-trained: the trainer itself prints "tokens per parameter: 1.31 (UNDER-TRAINED)" at this configuration. So the next arm is the
obvious one that has not been tried: **more data at the same recipe** (`lens/runs/MORE-DATA-PREDICTIONS.txt`).
