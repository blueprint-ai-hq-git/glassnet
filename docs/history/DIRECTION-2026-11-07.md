# Direction set by the maintainer, 2026-11-07 - the promise we have been under-delivering

The data-scale ladder is producing better numbers (12.18 -> 17.43 -> 20.81 -> 23.74 -> 23.92 held-out,
seed 0), and that is not the gap. The gap is that the model cannot yet do the things the project
promises. This file is the correction, in the maintainer's words.

## 1. Long context is parked. Make the 32 tokens hold, completely.
No longer-window work for now. The window is 32 tokens and it stays 32 until the model uses it fully.
- Measure held-out accuracy as a function of available context length (1, 2, 4, 8, 16, 24, 32 tokens)
  by rebuilding held-out windows at each length with `chat_data.make_windows`. Prediction first:
  accuracy should rise with context; if it is flat, the window is not being used and no future
  "unlimited context" claim can rest on it.
- The target is not "a bigger window", it is "100% use of the window we have".

## 2. Memorise what must be memorised, generalise what must generalise - and show which is which.
A model that does not know what numbers are called is not failing to generalise, it is missing an
arbitrary fact that has to be stored. The project must separate the two, visibly:
- an inventory of facts that are memorisation by nature (names, spellings, token strings)
- an inventory of behaviours that are generalisation by nature (arithmetic carried out, rules applied
  to unseen operands)
- a test per inventory item, scored separately, so "it memorised this" and "it computed this" never get
  reported as the same thing again
- this replaces the single blended accuracy number as the headline for capability

## 3. Normal-model features, built transparently.
The features an ordinary model has should exist here, but ours: training loop, optimiser, KV cache,
concept layer, explain(), stop token, decoding - each readable, each with a test, each with its
numbers published. Nothing borrowed from a framework, nothing hidden.

## 4. Learn faster. 50x is the target, and speed is already measurable.
Removing our gates measured 7.7x at 500k and 8.75x at 2M on identical data, with better accuracy
(+1.24 and +1.83 pp). The next speed work is batch shape and backend, measured the same way: same
config, one change, wall-clock per epoch and accuracy published side by side.
- target stated honestly: 50x over the gated baseline (558 s/epoch at 500k) is 11 s/epoch
- every speed claim ships with the accuracy it cost or gained, never speed alone

## 5. Fully decode it.
A single command that loads a checkpoint and prints, for a given prompt, the tokens, the active
concepts by name, the rules they fired, the class scores and the chosen answer. That is the demo that
makes the project legible to a stranger, and it is what the public page should link to.

## What this changes about the public story
The public page must lead with what the model *cannot* do yet, then this plan. The reason people did
not come is not the numbers - it is that there was nothing to see and nothing promised to watch.
