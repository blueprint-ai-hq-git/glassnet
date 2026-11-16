
## Data-driven check of the co-factor: it is real but smaller than my hand list suggested, and the sharper fact is own-word mass of 0.2%

Predictions in `lens/runs/NUISANCE-PREDICTIONS.txt`, written before this ran. The previous document measured the code/maths co-factor
with a **hand-written** keyword list; before any training change is aimed at it, the definition had to come from the corpus instead of
from my vocabulary.

## Method

Training conversations split by symbol density: the top quartile (184,041 conversations) is the code/maths slice, the bottom quartile
(186,834) is prose. Every token is scored by its code-versus-prose frequency ratio, and the top 2,000 become the nuisance set. Then each
concept's promoted mass - its absolute row of `answer.W`, normalised - is measured against that set and against its own trained words.

The data-driven set is recognisably code: `param`, `endl`, `getelementbyid`, `valueerror`, `cout`, `iostream`, `len`, `const`, `frac`,
`href`, `elif`, `arr`, `bool`, `str`, `datetime`, `ndarray`, `filename`, `def`.

## Result, and a correction to my own published number

```
hand list tokens present in vocab                 : 51
of those, inside the data-driven nuisance set     : 47  (92%)
median share of a concept's promoted mass on the
  DATA-DRIVEN nuisance set                        : 0.204
median share on the concept's OWN trained words   : 0.002
```

- **C4 PASSED**: the two definitions agree at **92%** - my hand list was not idiosyncratic.
- **C5 FAILED**: I predicted the median nuisance share would exceed **0.30**; it is **0.204**.
- **FALSIFIER NOT TRIGGERED**: overlap is far above the 40% that would have forced a re-derivation.

**Correction**: `docs/history/CONCEPT-CONTENT.md` reported a median nuisance share of **0.417** using the hand list applied to each
concept's *top-12 promoted tokens*. That is a different quantity from the share of the whole row, and the whole-row figure - the honest
one for "how much of this neuron's push is code" - is **0.204**. The earlier number measured the composition of the top-12 list, not the
neuron's mass. Both are now stated with their definitions so neither can be quoted as the other.

## The sharper fact this run produced

**A concept's promoted mass on its own trained words is 0.002 - two tenths of one percent.** Across all 53 concepts, the words a concept
was trained to detect account for a rounding error of the force it actually exerts on the output. Roughly 20% of its mass is code-flavoured
and the remaining ~80% is spread over the rest of the vocabulary.

That reframes the readability question again, and more sharply than before:

- `explain()` is **arithmetically exact** (2.9e-06) - verified.
- A concept **carries the whole decision** (concept share 1.000000) - verified.
- But the direction that concept pushes is **a broad, mildly code-flavoured direction in which its own words are 0.2% of the mass.**

So the honest description of the shipped model is: **"the answer is provably a weighted sum of 53 named directions, and each named
direction is a diffuse vocabulary-wide push rather than a concentrated detector of its name."** The name is a handle, not a description of
the direction's content. Any release note that says a concept "detects questions" is over-claiming; "the model's answer is exactly
attributable to 53 named handles, whose content is diffuse" is what the measurements support.

## The worst offenders, for the fix to target

```
cp_code_refs        nuisance 0.300   own words 0.006
mentions_a_number   nuisance 0.275   own words 0.012
cp_commands         nuisance 0.263   own words 0.001
is_explanatory      nuisance 0.252   own words 0.002
cp_shape            nuisance 0.249   own words 0.003
```

These are the concepts whose rows lean most on code vocabulary, and `cp_code_refs` being top is at least self-consistent - it is named for
code and its row is the most code-flavoured. The fix is unchanged and still evidence-backed: paraphrase supervision, a contrastive penalty
on the nuisance set, or supervision on a code-free slice. What changed is the target's size, which is 0.204 rather than 0.417.
