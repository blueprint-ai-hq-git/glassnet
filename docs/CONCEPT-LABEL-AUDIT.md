# The readability tax is not just accuracy: concepts stop tracking their own labels (2026-11-10)

The readable-only arm scored 20.30% against 25.52% (a 5.22 pp tax, `READABLE-ARM-VERDICT.md`) but also passed fewer
label-honesty gates. This names the failures instead of counting them, using the project's own gate
(`lens.api.audit`, AUROC bar 0.55, firing floor 2%) on the same 800 matched validation windows for both models.

```
                        readable-only    copy-path model
concepts passing           30/36            36/36
mean AUROC                 0.6886           0.8695       (drop 0.1810)
mean fire rate             47.4%            28.3%
coverage                    1.0000           1.0000
```

(The trainer's own 87,035-window sample says 28/36 and 35/36 for the same two checkpoints; same ordering, different
sample size. Both are real, neither is a correction of the other.)

## The six failures are exclusive to the readable model
```
concept                readable AUROC   copy-path AUROC
mentions_body             0.4275           0.7319
mentions_a_name           0.4662           0.9010
mentions_sport            0.4774           0.7676
mentions_a_quantity       0.5435           0.7801
mentions_family           0.5437           0.9039
is_a_list                 0.5471           0.7550
```
**Failing in both models: none. Failing in the copy-path model: none.** So the concept FORMULAS are not wrong for this
corpus - every one of the 36 passes when the concept layer is only asked to describe. The same formulas fail when the
concept layer has to carry the whole prediction.

## Why
Look at the two columns that move together: mean AUROC falls 0.1810 while mean fire rate rises from 28.3% to 47.4%.
The concepts in the readable model fire almost half the time regardless of their label, and discriminate less. That is
what a layer looks like when it is being used as a general compressed code rather than as 36 separate descriptions.

The individual numbers say the same thing. `mentions_self` goes 0.9366 -> 0.7439 while its firing rate goes 12.5% ->
39.9%; `mentions_the_user` goes 0.9508 -> 0.7709 and fires 26.5% -> 46.5%. A concept that fires on 40-47% of windows
is no longer reporting the presence of one thing, it is contributing a shared code that happens to be indexed by that
name.

## What this means
1. **The price of readability is bigger than the 5.22 pp of accuracy.** It also costs 0.18 of mean AUROC on the very
   thing the project promises: that a concept means what its name says.
2. **This is a functional conflict, not a supervision deficit.** Every formula passes in the descriptive model, so
   feeding the same formulas harder cannot be the whole answer - the loss is competing with the readout's need for a
   compact code. The queued arm (`--lam-concept 1.0`, `READABLE-CONC-PREDICTIONS.md`) tests exactly this, and its P1
   (coverage rises to 33/36) is now the optimistic reading of a conflict rather than a simple fix.
3. **The honest design question for this project is now explicit:** either the concepts are 36 honest descriptions and
   something else decides the answer (25.52% with 36/36 honest, explanations covering 4-8% of decisions), or the
   concepts decide (20.30%, 30/36 honest, explanations that are true by construction). The middle ground that was
   supposed to give both has not appeared in any measurement so far.
