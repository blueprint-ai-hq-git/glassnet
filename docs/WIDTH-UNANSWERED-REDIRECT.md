

---

# THE WIDTH QUESTION IS UNANSWERED, AND THE REASON REDIRECTS IT (2026-11-22 16:18)

Two runs, both refused by the no-learning guard. That is the correct outcome, and it teaches something more useful than a width number.

## What was run

`lens/width_trained.py` trains the shape the repair needs: in `lens/api.py` the answer head is `Dense(K, V)` applied to the concepts, with
`logits = self.answer(h)` and no bypass, so **the concepts are the only route to the answer**. Task: next-token prediction on real corpus
windows (vocabulary capped at 2,000 tokens). Widths K = 8, 16, 36, 72.

```
windows 22000 (train 17600, test 4400) | window 16 | most-frequent-answer baseline 16.32%
      K     params    test acc  vs baseline
      8     154200      16.32%     +0.00
     16     178400      16.34%     +0.02
     36     238900      16.36%     +0.05
     72     347800      16.36%     +0.05
OUTCOME: INVALID_INSTRUMENT - nothing learned, so no width verdict can be issued
```

The 300-window smoke run produced **16.30% for all four widths**, which is what prompted the guard; the full run reproduced it at 22,000
windows. **A flat line at the baseline is not evidence that width does not matter. It is evidence that this model did not learn this task.**

## What that actually means

**A layer of 8-72 word-presence concepts cannot predict real-text next tokens at all, at this budget.** The concepts "does token X appear
in this window" carry almost no information about which of 2,000 tokens comes next. So the width question was being asked of a task where
the readable layer could never succeed for a reason that has nothing to do with width: **the concept vocabulary is wrong for the job, not
too narrow.**

That is the same failure the whole day kept finding in a different costume: **the readable layer describes the sentence instead of
deciding the answer**, and adding more of the same descriptions does not change that.

## Where the honest repair question now points

There is already a case in this project where the readable route **does** carry the answer perfectly: the written-working models
(`lens/explain_emit.py`), where the model writes the intermediate state as tokens and the explanation is its own output -
**100.0% on one, two and three items, verified against perturbed inputs**. The difference between those and the concept layer is not
width:

| | written-working models | concept layer on chat text |
|---|---|---|
| what carries the answer | tokens the model writes | 36 fixed presence concepts |
| reliability | **100.0%** on selection/comparison | concept route reproduces the answer **5.5-18%** |
| who chooses what is written | **the model** | a fixed list chosen by me |

**The next real experiment is therefore not "more concepts" but "let the model choose and write the concepts it uses"** - the same move
that made arithmetic and the two-item choice work. Until a model writes its own intermediate concepts, the count of mine is a number
about my list, not about the model.
