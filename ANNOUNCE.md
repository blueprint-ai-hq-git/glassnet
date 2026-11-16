# Announcement pack

Everything below is copy-paste ready. Every number is measured and traceable to a file in this repo. Nothing here is a projection.

**Do not claim**: that this beats large models, that concept names are meaningful, or that the readable architecture is more accurate than a plain one
at equal budget - it is not (matched comparison: -1.53 +/- 0.44 points).

**Do claim**: it is built from scratch, every neuron has a name, `explain()` is exact, and ten pre-registered tests were published as failures.

---

## One-line pitch

> A tiny language model you can read: every neuron has a name, `explain()` returns the exact arithmetic behind each answer, and none of it uses
> PyTorch or autograd.

## Show HN title

> Show HN: A language model with named concept neurons, built from scratch in NumPy

## X / short post (under 280)

> Built a 14M-param language model from scratch in NumPy. No PyTorch, no autograd.
>
> Every middle neuron has a human name. explain() returns the exact sum behind each answer: residual 2.4e-06.
>
> It reaches 24.6% next-token accuracy. And its named neurons do NOT mean what their names say - I published 10 falsified tests proving it.

## Long post (r/LocalLLaMA, r/MachineLearning)

**I built a language model where every neuron has a name. Here is what that got me - and what it did not.**

I wanted a model I could actually read. Not attention maps, not "the model looked here". A model where a middle neuron is labelled `is_a_request`
and a rule layer connects two or three of those labels, and `explain()` returns the arithmetic instead of a narrative.

**What is built.** Own tensors, forward pass, backward pass, losses, optimiser, KV cache. No PyTorch, no autograd library, no pretrained weights. Every
gradient is checked against finite differences before it is trusted. 53 named concepts for chat, 18 for images. `explain()` is exact: max residual
2.4e-06, concept share 1.000000 - the explanation is the computation, not a story about it.

**What it achieves.** 24.58% +/- 0.055 next-token accuracy on 750k conversations, over three seeds. Perplexity 181.04. That is a small model and I am
not going to pretend otherwise.

**What it does not achieve - and this is the part worth reading.** A concept neuron labelled `is_a_request` detects `is_a_request` in its input at
AUROC 0.9997. But the weight row that concept uses to answer holds **0.08% of its own words' probability mass**. In plain terms: the neuron knows
what it is looking at, and still does not prefer saying it.

I spent today trying to fix that. Ten mechanisms. Free answer rows, structurally derived rows, priced corrections, two independent hard caps, pure
structural rows, row-content supervision, next-token concept supervision, and a 15x weight sweep on the one term that showed a pulse.

The hard cap is the cleanest demonstration: it did exactly what it was designed to do, flipping the correction's energy share from 0.9999 to 0.2087 and
making the structural term the dominant carrier of every answer row. **Own-word mass went from 0.0008 to 0.0016.** Ten mechanisms span correction
shares from 0.0000 to 0.9999 and own-word mass never leaves 0.0007-0.0019.

And the 15x sweep: on a frozen model the row-content penalty raises own-word mass from 0.1207 to 0.1237 in one gradient step. Train *with* it for six
epochs and the model sits at 0.0007 at weight 2, 8 and 30 alike. Training erases the term's effect rather than resisting it.

**One real win, published with its cost.** Labelling a concept only when its own word is the token being predicted gives **+1.54 points** of accuracy,
paired on seed, 3 of 3 seeds positive, best ppl 158.85. It also takes vocabulary coverage from 0.292 down to 0.167, at two different strengths. My
pre-registered promotion rule required both accuracy and coverage. Accuracy cleared it, coverage did not, so **it is not shipped** - it is published as
a measured gain that is not promotable.

**Why publish the failures.** Because a transparency project that hides its negative results is not a transparency project. Every test above had its
prediction written to a file before the run, and when a falsifier fired I published the firing. There are 195 write-ups in `docs/`, 826 result files,
and the shipped configuration is the one that survived, not the one that looked best.

**Repo**: github.com/blueprint-ai-hq-git/glassnet - MIT, no pretrained weights, datasets licensed and listed in `DATA_LICENSES.md`.

If you work on interpretability: the open question is how to make a concept's answer row favour that concept's own vocabulary. Ten things do not work.
I would like to know what an eleventh would be.

---

## Where to post, and what to expect

- **r/LocalLLaMA** - long post fits the culture; lead with the negative result, not the accuracy
- **r/MachineLearning** - strict self-promotion rules; post as a discussion of the negative finding with the repo in the body, and check the
  self-promo policy first
- **Hacker News, Show HN** - the "built from scratch in NumPy" angle plus exact `explain()` is the hook
- **X** - short post above; the falsified-tests line travels further than the accuracy line

Nobody can guarantee reach. What makes this spreadable is that the negative result is unusually specific and honest, and that every number has a file
behind it.
