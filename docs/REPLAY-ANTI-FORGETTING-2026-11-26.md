# 2026-11-26 - replay's value is ANTI-FORGETTING, not accuracy: it fails to transfer twice on the real model

Source: lens/runs/OLD-MODEL-REPLAY.json (overwritten by the fair run), journals of units glassnet-oldreplay2 and
glassnet-oldreplay3. Model: direct1024_d512, d=1024, 36 named concepts, direct route, tied embeddings, 47.5M parameters.

Two independent tests on the real model, both three seeds per arm, both from identical starting weights:

    test A - NARROW pool (6,000 windows from 400,000 tokens), 1,500 steps
      before          6.7324 nats   12.83%
      no_replay       6.5954        12.41%   (12.43, 12.40, 12.40)
      replay          6.5951        12.45%   (12.45, 12.40, 12.50)
      replay advantage  +0.04 points = NOISE

    test B - FAIR pool (200,000 windows from 20,000,000 tokens), 3,000 steps
      before          6.9352 nats   12.20%
      no_replay       6.7396        12.27%   (12.28, 12.28, 12.25)
      replay          6.7363        12.24%   (12.20, 12.22, 12.30)
      replay advantage  -0.03 points = NOISE, loss +0.0033 nats = NOISE

## What the two tests together say

On the small copy of this architecture (855k parameters, 8,000 vocabulary, 40,000 windows) replay was worth **+0.93 accuracy
points** at 3.5 sigma. On the real 47.5M model it is worth **+0.04 and then -0.03**, i.e. nothing, twice, with the second test
given twelve times the training pool after I found the first one was data-starved.

The one rule that explains all three results: **replay pays when the data pool is narrow relative to the training budget, and pays
nothing when fresh windows are plentiful.** In the small copy the pool was small, so revisiting old windows added information the
fresh draws could not supply. On the real model with 200,000 windows and only 3,000 steps, every batch was already fresh, so a
reservoir adds nothing.

That is consistent with the one place replay is already proved in this project: **forgetting**. Without replay a model lost 92.9
points of an old skill (0.8% retention); with a 1,024-window buffer it kept 88.4%. Replay protects what is already learned. It
does not make a model learn better when it is not under forgetting pressure.

## So the status of each brain mechanism is now

    rehearsal (replay)   KEEP, for anti-forgetting only - not an accuracy lever on a wide pool
    named concepts       NEUTRAL for accuracy, kept only as a readability feature
    coverage strata      DROP - costs on its own and +1.29 nats together with replay
    collapse metric      KEEP as a diagnostic - it caught models predicting 7 and 23 distinct tokens

## What this rules out

Any further attempt to buy accuracy on the real model by adjusting these three mechanisms. The measured accuracy levers in this
project are distinct data volume and training steps; the mechanisms are not one of them. Next work should spend its budget on
tokens, not on mechanism tuning.
