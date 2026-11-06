# GlassNet — beginner's guide (read me first)

GlassNet is a neural network that you can **read**. Every middle neuron has a
name, every decision comes with a receipt, and no number is hidden from you.
It is built completely from scratch — our own math, our own training, no
borrowed black boxes.

## Quick start (the whole library in 3 sentences)

```python
from glassnet.api import GlassNet

model = GlassNet("fashion-mnist")   # 1. make the model
model.train()                        # 2. train it (auto-pilot does the tuning)
print(model.explain(18))             # 3. "why did you say that?"
```

That's it. The auto-pilot watches training and fixes problems by itself; it
writes down every action it takes. The first `train()` downloads
Fashion-MNIST once (about 30 MB, MIT licensed — see DATA_LICENSES.md).

## What actually happens inside `train()`

1. **Concept layer** — 14 name-tagged neurons, each forced to track one
   measurable pixel concept: symmetry, tall_shape, quadrant inks, edges…
   (exact formulas in CONCEPTS.md)
2. **Sparse rule layer** — 30 rules; each rule uses at most 3 concepts.
   Example rule: *"not wide AND has horizontal edges AND very tall → vote
   trouser"*.
3. **Answer layer** — every clothing class adds up the votes of its rules.

Layer C only ever reads B, B only ever reads A, and A's names are audited:
after training we measure whether each neuron *really* fires on its named
concept, and a fake "lie neuron" test proves the audit would catch a liar.

## How to read an explanation

```
I said 1 (trouser) with probability 94.2%
because of these rules:
  rule #17 fires 78% pushes +1.958
      wide_shape: w -1.62 off (a=0.13)
      horizontal_edges: w +1.40 ON (a=0.59)
      tall_shape: w +0.65 ON (a=1.00)
```

- `w` = how strongly this rule listens to that concept (negative = the rule
  fires when the concept is *low*).
- `ON/off` = whether the concept is active for THIS image (`a` = activation).
- `pushes +1.958` = how hard this rule voted for the predicted class.

So that example says: *"I said trouser because rule #17 fired — the image is
very tall, not wide, with horizontal edges."*

Want a picture instead? `model.explain(18, save="why.png")` saves a figure
with the image, the concept bars, the rule votes and the class probabilities.

## Your own images

```python
model = GlassNet()                 # the API is the same
X, y = your_images_numpy, your_labels_numpy   # (N,28,28) or (N,784), [0,1]
model.train((X, y))                # it auto-splits 80/20 train/test
```

## The data judge

```python
model.train_judge()                # ~10 seconds, hand-labeled good/bad pile
scores = model.judge(X)            # >= 0.5 suspicious, < 0.5 fine
```

The judge is trained on 1,200 good clothes vs 1,200 deliberately ruined ones
(8 ways of ruining). A cheap cleaner runs first and removes duplicates,
blank rows and broken pixels. Measured on data it never saw: ~80% held-out,
~71% on corruption types it never even trained on (chance = 50%).

## Honesty rules (this project's law)

- Every number in this project is measured and written to `runs/` — nothing
  is estimated, nothing is hidden.
- Every secret number inside the auto-pilot's brain is listed in
  `HIDDEN_DEFAULTS.md`.
- The system refuses to train unless its gradient checks pass.

## Troubleshooting

- `Need internet on first run` — the dataset downloads once, then is cached.
- Slow? It's CPU-only by design (float32, small nets). One full training
  run takes about a minute.
- Errors? Every script prints an honest traceback — paste it to the owner.
- Where are the numbers? `runs/phase1..5/` — every log, checkpoint, CSV and
  report from every phase of the build.