# Training methods: what we use, what we are inventing

Written 2026-10-18 because the user asked directly: *"what training methods are you using, we have to invent
our own training methods for distillation, RL, normal SFT, teacher model training, and everything is
learning"* - plus two hard requirements: **one setting that runs from small up to 10B without the model
going dumb**, and **a memorisation safety net that training finds by itself**.

## 1. What we use TODAY (nothing hidden, nothing claimed beyond this)

| piece | what it is | where |
|---|---|---|
| objective | next-token cross-entropy, masked to assistant tokens | `lens/scripts/train_chat.py` |
| concept loss | each named unit is trained to match its concept label | `lens/model.py` |
| sparsity | optional L1 penalty on mean(|concept activation|) | `--lam-act` |
| turn-end | the stop token is trained instead of being masked out | `--train-turn-end` |
| 2-token head | second head predicts the token two steps ahead | `--mtp 2` |
| optimiser | plain SGD with a hand-set learning rate | `train_chat.py` |

**No distillation. No RL. No teacher model. No SFT stage.** One flat objective. That is the honest state,
and it is the weakness the user identified.

## 2. Why a hand-set learning rate can never be "one setting to 10B"

The same rate that works at 1M parameters is too large at 100M (the step exceeds what the curvature
allows) and too small at 100k (it crawls). A hand-set rate is therefore a per-size arm - exactly what the
user objects to. `lens/opt.py` replaces it with **scale-invariant** rules that read the step size from the
gradient itself:

* `relgrad`: `w <- w - lr * g / (||g|| + eps)` - step depends on direction, not magnitude
* `rmsprop`: `w <- w - lr * g / (sqrt(EMA(g^2)) + eps)` - per-weight scale from recent gradients

with `lr` meaning "how far to move in weight space", so it transfers across sizes. Smoke test (rules task,
d=64, 600 epochs, our relgrad): **it trains** - 50.0% train at lr 0.02, 70.0% at lr 0.1 - and does not
blow up, at two different rates, on a task where plain SGD needed per-size tuning.

## 3. The memorisation safety net (training finds it by itself)

`lens/opt.MemorisationGuard`, wired into `train_chat.py` as `--max-gap`. It computes, every epoch,

    gap = train_accuracy - held_out_accuracy

and stops the run when the gap crosses the limit, reporting whether the gap was still RISING (memorisation
in progress) or merely high and flat (a model that fits its data better than the held-out set). It needs no
human to notice. Calibrated against a known case from this project: at 192 epochs one run reached train
43.19% against honest 4.62% - a 38-point gap that looked like progress and was memorisation.

## 4. The finding that decides the next change

A controlled rules test (`lens/scripts/rules_test.py`) trains on 80 addition pairs, then measures **held-out
pairs from the same range** and **extrapolation to 10..19**:

| run | train | held out | extrapolation | verdict |
|---|---|---|---|---|
| d=64, rms lr 0.05, 600 ep | 65.0% | 15.0% | 0.0% | partial, 50-point gap |
| d=64, relgrad lr 0.02 | 50.0% | 5.0% | 0.0% | partial, 45-point gap |
| d=64, relgrad lr 0.1 | 70.0% | 5.0% | 0.0% | partial, 65-point gap |

**The optimiser is not the bottleneck.** The concept layer is a bag of token-presence tests over a
flattened window, so it cannot BIND "3 is the first operand" and "5 is the second" - and addition is a rule
about exactly that binding. So the model memorises what it can and generalises nothing.

**Therefore the next change is representational, not optimisational: slot/position binding in the concept
layer** (named units that reference a slot, e.g. `first_operand_is_7`, `second_operand_is_5`, verifiable
from the ids). One change, its own prediction, re-run the same rules test.

## 5. Our own methods still to build (declared, not implemented, not claimed)

1. **Self-distillation from our own best checkpoint** - the model's own earlier, better self as teacher; no
   external weights, so it stays inside the from-scratch rule.
2. **RL on verifiable rewards** - maths and code answers can be checked exactly by a program, so no reward
   model is needed and no human labels: sample several answers, keep the ones that check out, train on
   those (rejection-sampling RL / GRPO-style), with the rule-learning test as the reward's honesty check.
3. **Curriculum driven by the model's own error rate** - feed the model the examples it currently fails,
   measured per concept and per topic, instead of a fixed shuffle.
4. **SFT on the model's own successful traces** - once generation ever succeeds (a maths answer that
   checks out, code that parses), train on those traces; this is the loop that compounds.
5. **Teacher-model training** - a bigger size in our own ladder teaches a smaller one (our own weights, our
   own architecture, so no pretrained-weights rule is broken).

Each one is a separate experiment with a prediction written before it runs, and each must pass the
memorisation net and the random-draw gates before it counts as working.

## 6. NO SEQUENCE LIMIT (measured, both directions)

The user's requirement: *"we have to set the sequential token limit for output and the data input ... because
of the RAM GB ... we have to find a way so it works, we don't need any limit, it just works."*

Why it is possible for us and not for a transformer: attention keeps every token, so its memory grows with
sequence length and it must be given a hard context window. Our architecture summarises a fixed-size window
into a fixed number of named concepts, so a chunk costs a fixed amount of memory and arbitrarily long input
can be streamed through a bounded running summary (`lens/stream.py`, `StreamingEncoder`).

INPUT, measured (ship sb_S0_d256_noattn_ep96, window 32, 41 concepts):

| tokens fed | chunks | buffer length | summary size | peak RSS |
|---|---|---|---|---|
| 1,000 | 62 | 0 | 41 | 75.6 MB |
| 10,000 | 625 | 0 | 41 | 75.6 MB |
| 100,000 | 6,250 | 0 | 41 | 75.6 MB |
| 1,000,000 | 62,500 | 0 | 41 | 75.6 MB |

A 1,000x longer input moved memory by **0.0 MB**. OUTPUT, measured with the stop rules off so nothing stops
it early: 200 / 1,000 / 3,000 tokens generated, memory 377.8 MB flat (**+0.0 MB** from 1,000 to 3,000).

HONEST LIMIT, stated so nobody over-claims: the summary is a fixed-size running state (decay + max), so very
early input is COMPRESSED rather than retained word-for-word. "No token limit" means no memory limit and no
hard cap - it does NOT mean perfect recall of a million-token input. Only the first claim is made.
