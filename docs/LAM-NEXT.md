
## The accuracy/breadth trade is not a dial: halving supervision strength raises accuracy to 26.52% and still costs coverage (2026-11-16)

Prediction file `lens/runs/LAM-NEXT-PREDICTIONS.txt` and its promotion criterion were written before either arm ran. Numbers from
`lens/runs/LAM-NEXT.txt` and `lens/runs/LAM05-DECODE.txt` (12 prompts, 40 generated tokens).

```
arm                        VAL acc   VAL ppl   coverage T=0.7   repetition T=0.7   KL
next-target lam 1.0        26.12%    169.67        0.173             0.002        20.19
next-target lam 0.5        26.52%    158.85        0.167             0.083        20.65
shipped (window labels)    24.58%    181.04        0.292             0.027        17.73
```

## Every number moved the wrong way for promotion

- **accuracy and perplexity improved**: 26.52% and 158.85 are the best this project has measured on both
- **coverage fell again**: 0.167, below the lam 1.0 arm's 0.173 and far below the shipped 0.292
- **repetition got worse**: 0.083 against 0.002 at lam 1.0 and 0.027 shipped
- **KL got worse**: 20.65 against 17.73 shipped

The promotion criterion fixed in advance required **coverage >= 0.27 AND VAL >= 25.2**. Accuracy clears it with margin. Coverage fails at 0.167.

## The falsifier fired, and it answers the question the curve was built to answer

I wrote: *"if coverage stays below 0.22 at BOTH strengths, the breadth cost is not a strength dial - it is carried by the label change itself - and
promoting accuracy-with-breadth requires a mechanism that changes WHAT is supervised per position rather than how hard."*

Two strengths, 0.173 and 0.167, both under 0.22. **The trade is not a dial.** Supervision strength sets accuracy (26.12 -> 26.52 when halved) and
leaves breadth roughly where it was. Lowering the price buys a little accuracy and no breadth at all, which is the opposite of the middle setting I
hoped for.

**So this direction closes.** Next-token concept supervision is a real and repeatable accuracy gain - +1.54 paired on three seeds at lam 1.0, and
higher still at lam 0.5 - and it cannot be shipped, because what it buys in accuracy it takes out of vocabulary breadth at every strength tested.

## What is now established about the two halves

- **accuracy** responds to the supervision target and to its strength
- **breadth** responds to neither, in this family
- **concept-row meaning** responds to neither, across ten mechanisms

Three quantities, one lever that moves only the first. That is the cleaned-up statement of this project's central difficulty, and it is more precise
than anything available this morning.
