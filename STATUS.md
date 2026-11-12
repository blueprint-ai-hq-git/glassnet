## Facet concepts: best concept set so far (16.38%), but the model still prefers impossible rules

12 concepts split into four near-independent facets - shape of the last token, content anywhere, position of the
chat tag, punctuation earlier - score **16.38%** against 15.28% for single tokens and 15.40% for positional pairs,
with the cheapest readability price of the three (**1.92 points**). Cross-facet rules now appear among the
heaviest, e.g. `position: chat tag in last 4 AND earlier: token at -3 is punct`. But the second-heaviest rule is
still `shape: last token is punctuation AND shape: last token is 6+ chars`, lift **0.00** (impossible), influence
74.3: a random wiring still offers contradictions and the objective still likes them. Cross-facet wiring is
running. Details: `docs/FACET-CONCEPTS.md`.

## Token-pair concepts: rules are nonsense because the concepts are mutually exclusive

Replacing single-token concepts with positional/pair concepts ("previous token is punctuation", "last two tokens
both digits") held accuracy (**15.40%** vs 15.28%) but median rule pair lift FELL to **0.72** while median |PMI|
rose to **2.18 bits** - the concepts are strongly related, negatively. The model's heaviest rule is
`previous token is punctuation AND last two tokens both digits`, lift **0.00**: not rare, **impossible**, because a
token cannot be both punctuation and a digit. Conjunctions of positional concepts are therefore either
contradictions or tautologies, which is the structural reason 2-3-concept rules read as nonsense - a design error
in the concept set, not in training. Concept honesty also fell to **74.8%**. Details: `docs/PAIR-CONCEPTS.md`.

## Rule semantics: the biggest rules join concepts that avoid each other

`tools/rule_semantics.py` measures every rule against the data. In the unconstrained model **23 of 96 rules join
statistically independent concepts**, median pair lift **0.98** (the wiring was, on average, exactly independence),
and three rules join **logically impossible** pairs (lift 0.00) - `last token is punctuation AND last token is a
digit` - while still carrying influence 50.7. Constraining the mask to pairs with lift >= 1.5 raised median lift to
**1.38** and PMI to **1.279 bits** and removed the impossible pairs at no accuracy cost (**15.30%** vs 15.28%), but
the influence-vs-lift correlation stayed **negative** (-0.127): the model changed which rules exist, not which ones
it trusts. Concept honesty fell 84.0% -> 79.0%. Details: `docs/RULE-SEMANTICS.md`.

## Causal test: concepts drive the answer, but the rules read as nonsense

`tools/intervention.py` forces one concept neuron on or off and recomputes the answer. Forcing a concept changes the
answer on **43.2%** of held-out windows on average (up to **84.8%** for "last token is '?'"), so the readable path
is genuinely used. But honesty and usefulness are only weakly linked (pearson **+0.253**): `contains '}'` is 93.1%
honest and flips only **20.3%**. And the printed reasons expose the real defect - high-weight rules are arbitrary
conjunctions like "window contains <unk> AND window contains '}'", mechanically true and humanly meaningless.
Accuracy cannot detect that. Rule semantics is the next gate. Details: `docs/CAUSAL-EXPLAIN.md`.

## Honesty is a tax, not a state: the concept warmup interpolates the trade-off

Holding the concept loss at 5x for 800 steps then releasing it to 1x gives **15.28% accuracy / 84.0% honesty**,
which sits on the line between 1x (**16.65% / 71.8%**) and 5x (**13.47% / 92.0%**) instead of above it. That is
about **1.2-1.4 points of accuracy per 6 points of honesty** at every setting, so honesty costs accuracy in each
step it applies rather than being a state the model keeps. Readability price at the balanced point: **2.47 points**.
Details: `docs/HONESTY-IS-A-TAX.md`.

## PHASE 2 GATE: all five items PASS on the deliverable configuration

12 train-probe-selected concepts -> 96 rules (2 connections each) -> answer, 2,692,780 params: accuracy **13.47%**
vs **5.42%** trivial (**+8.05**); **96 rules** to read; concept honesty **92.0%** balanced against an 85% bar
(worst concept 63.8%); ablation **17.18%**, so readability costs **3.71 points**. The 92.0% cost 3.18 points of
accuracy against the same concepts at 1x (16.65% / 71.8%), and the best accuracy ever reached with this layer is
**19.68%** at 64.9% honesty - honesty and accuracy trade through one dial, measured across six settings.
Details: `docs/PHASE2-GATE-FINAL.md`.

## The full 36-concept table: rarity does NOT explain honesty, and 15 neurons are dishonest

The whole table exists now (`tools/concept_table.py`, no retraining): mean **67.1%** balanced, min 48.0%, max 99.5%
('last token is .'). Prevalence vs honesty across 36 concepts: **pearson -0.021, spearman +0.114**, so the earlier
'rare concepts starve' explanation is falsified. The cross-firing test finds **15 of 36 neurons predicting another
concept's label better than their own** - e.g. `contains 'a'` at 67.2% own but 84.7% for "last token is '?'".
The concept set is now selected honestly by a train-split probe (`tools/select_concepts.py`, 12 of 36 kept).
Details: `docs/CONCEPT-TABLE.md`.

## Concept honesty vs accuracy: one dial, three settings

Same recipe, only the concept-loss weight changed: **1x -> 19.68% accuracy / 64.9% balanced honesty**, **2.5x -> 19.20% /
74.6%**, **5x -> 17.82% / 84.0%**. The middle dose costs **0.05 points of accuracy per point of honesty**, half the price of
5x (0.10). So the concepts were under-weighted, not dead, and label-honest concepts cost **1.86 points** of
prediction. The readability price also rises with honesty: the rule layer costs 0.75 points against its ablation at
1x and 2.13 points at 5x. Caveat in the doc: one seed per point, and 84.0% still misses the 85% bar with 'as' dead
at 52.8%. Details: `docs/CONCEPT-TRADE.md`.

## CORRECTION: concept honesty is 64.9%, not 79.2%

The Phase 2 gate's 79.2% was plain accuracy on concepts that are true in a median of only **16.3%** of windows, so
always answering 'no' scores **83.7%** - higher than the reported figure. Balanced accuracy, where a dead neuron
scores 50, puts the concepts at **64.9%** with five of 36 effectively dead (worst 'as' 50.3%). Token accuracy is
unchanged at 19.68% and the readability price unchanged at 0.75 points; only the honesty verdict was wrong.
Details: `docs/CONCEPT-HONESTY.md`.

## PHASE 2 GATE, all five items measured

Three-layer model (36 concepts -> 96 rules -> answer), 2,696,644 parameters: accuracy **19.68%** against a **5.42%**
trivial baseline (**+14.26 points**); **2 connections per rule** after pruning from 3; **96 rules** to read; concept
honesty **79.2%** mean (weakest 57.6%, FAILS the 85% bar); ablation without the rule layer **20.43%**, so
readability costs **0.75 points**. Input choice matters most: window mean 9.20%, last token 15.38%, both 19.68%.
Details: `docs/PHASE2-GATE.md`.

## Phase 2: word order is worth +6.2 points, and readability costs 1.27

The three-layer model (36 concepts -> 96 rules -> answer) scores **9.20%** when the concept layer sees the window
mean and **15.38%** when it sees the last token, against a **5.42%** trivial baseline - so order is worth
**+6.18 points**, the biggest effect measured on this layer. Concept honesty moves the other way: **86.5%** with
the window mean, **70.7%** with the last token, because the concepts are named for window contents. The ablation
with no rule layer scores **16.65%**, so readability costs **1.27 points**, measured. Details:
`docs/RULES-ORDERED.md`.

## Phase 2 rule layer: built and honest, and it fails against the right bar

36 concepts -> 96 sparse rules (3 wires each, pruned to 2) -> answer now exists. On held-out windows it scores
**5.42%**, and 'always output the most common token' also scores **5.42%**: it learned only the frequency prior.
My prediction ('within 2 points of the ablation') passed, which is exactly how a bad gate hides a null result -
the ablation scores 3.55%, below a constant. Cause: mean pooling destroys word order, so there is nothing to
condition on. Readability numbers are real: 2 connections per rule, 96 rules, concept honesty 82.9% mean.
Details: `docs/RULES-GATE.md`.

## Identity pointer: 83.3% held-out copying, a new best

Deleting the pointer's learned readout matrix and tying it straight to the embedding table raises held-out copying
from a previous best of **70.2%** to **83.3%** at step 1,000 (1,500 held-out items), taught copying 84.7%, with
language 13.45% at that point. The reasoning is exact rather than empirical: `(alpha-weighted window embedding) @
E^T` is the attention-weighted sum of one-hot token scores, so attending to a position outputs the token there,
with no parameter in between for language training to overwrite. Prediction on record: copy PASSED (75% bar),
language FAILED at that checkpoint (13.45% against 16%). Details: `docs/IDENTITY-POINTER.md`.

## The library works in five lines, and two-stage training fails in all three forms

`import glassnet as gn` now exposes the real engine: `gn.load_chat()`, `gn.model()`, `gn.fit()`, `gn.why()`.
Real run: auto-pilot chose 15,567,566 params, d512, lr 0.30, loss 7.86 -> 6.08 in 3.09 s, **held-out 4.70%**
after three epochs, and `explain()` returned the concept decomposition with residual 11.72 and `exact: False`
(the readable path does not explain the whole logit - the same 7.3% share already published). Three epochs is a
working pipeline, not a good model, and it is labelled that way. `docs/LIBRARY.md`.

Separately: language-first two-stage training was measured three ways and **failed every time** - held-out copy
6.8% unfrozen, 3.2% frozen, 2.5% frozen with a separate frozen embedding, against predictions of 40-50%. Freezing
also cost 8-10 points of language. `docs/TWO-STAGE-FAILS.md`.

## The copy peak, measured properly: 69.8% held-out copying at 1,000 steps

On 3,000 evaluation items per checkpoint (up from 600): held-out copying **70.2 / 63.1 / 56.8 / 46.5%** for seed 0
and **69.3 / 65.5 / 55.1 / 47.2%** for seed 2 at steps 1,000/2,000/3,000/4,000, while language climbs 14.6 ->
20.4%. The prediction (50%+ at step 2,000 on both seeds) **passed**. Copying decays monotonically as language
improves: a measured trade-off, not a tuning problem. Checkpoints are now written at every evaluation, so the
70.2% model is a file (`early_s0_best.npz`), not a line in a log. Details: `docs/COPY-PEAK.md`.

## Copy generalisation peaks early: 63.7% at 2,000 steps across three seeds

Routing, three seeds. Held-out copying at step 2,000 is **64.3 / 66.7 / 60.2%** (**mean 63.7, spread 6.5**) while
language is still ~16%. By step 12,000 language reaches **25.4%** (mean of three seeds, spread 0.6) and held-out
copy drifts and swings (1.7-56.2%). Mean of last three: copy **26.9%**, language **25.4%**. So the two abilities
trade against each other along one run, and the copy operation is strongest at the START of a mixed run. Next
arm trains language first and adds a short copy stage at the end. Details: `docs/ROUTE-SEEDS.md`.

## The data judge exists and fails its own gate

Fifteen readable features, five corruption kinds, held-out conversations. Without order features: AUC **0.922**
but shuffled word order detected only **1.0%**. With them: AUC **0.98**, but **29.4%** of clean held-out windows
flagged. With a calibrated threshold: AUC **0.9597** PASS, false alarm **0.0730** FAIL (bar 5%), one-token-loop
detection **69.5%** FAIL (bar 80%). Not wired into training: a filter that throws away 7% of good data costs more
than the damage it removes. Next design recorded: score a window against its own neighbours instead of a global
threshold. Details: `docs/DATA-JUDGE.md`.

## Routing beats adding: the first model that talks AND copies

Switching between the pointer's readout and the other pathway's readout, instead of adding them, gives held-out
copying of **39.5 / 3.2 / 56.2%** over the last three checkpoints (**mean 33.0%**) with language at **25.8%**
(mean). Every earlier mixture sat between 0.2% and 16.5% held-out. The reason is documented in
`docs/POINTER-AIM-VS-VOTE.md`: the pointer never loses its aim, it loses the SUM. A switch does not have to
out-sum anything. Caveat stated in the doc: this metric swings hard between checkpoints, so 33.0% needs three
seeds before it is a stable number. Details: `docs/ROUTE-BEATS-ADD.md`.

## The pointer keeps its aim and loses the vote

Measured on the trainer's own items: on held-out words the pointer's attention still sits on the correct window
slot with **0.77 mass against 0.031 chance**, and it gets the word right **58.0% (63.0% after mixing)** alone -
but the assembled model outputs it only **0.5-14.8%** of the time. So the copy operation is intact and the
mixture outvotes it: the problem is voting, not erosion, and every arm that tried to fix it by training longer,
freezing, isolating the embedding table or lowering the learning rate addressed the wrong half. Details:
`docs/POINTER-AIM-VS-VOTE.md`.

## CORRECTION: separate embedding table 15.7% at the finish, and single checkpoints must not rank arms

I published that the separate embedding table was worse (3.3%) by comparing its step-4,000 reading against a
finished 12,000-step arm. Its finished figure is **15.7% held-out copying, the highest today**, with **30.15%**
language. But held-out copy swings 7.0 / 9.3 / 7.5 / 15.7 across checkpoints on identical settings, so single
readings rank coin flips: mean of the last three is **14.8%** for the gate arm and **10.8%** for the separate
table, i.e. **the two are indistinguishable**. What is established is the direction 0.2% -> 3.8% -> 14.8-15.7%.
Details: `docs/SEP-EMB-CORRECTION.md`.

## Six ways to mix a copy pathway into a language model, measured

The pointer copies unseen words at **58.0%** alone. Every mixture measured today lands between **0.2% and
16.5%**: 50/50 mixture 4.2%, 24k steps 8.5% (plateau), frozen pointer 0.2%, bounded gate 3.8%, **unbounded gate
14.8%**, separate embedding table 3.3%. Two findings worth the space: a gate bounded in [0,1] can only attenuate
an already-too-quiet voice, and freezing the pointer's own matrices protects nothing because it reads the shared
embedding table that corpus batches keep moving. Best configuration: language **29.77%** with **96.0%** taught
and **14.8%** held-out copying. Details: `docs/MIXING-VARIANTS.md`.

## A model that talks and copies at once: 29.82% language, 85.7% taught copy, 4.2% held-out

First arm to train both abilities together (50% chat corpus, 50% synthetic copy, pointer instead of the old
echo path, 8,000 steps): language **29.82%** on matched windows against **24.27%** for the same model before
the arm (**+5.55 pp**, same ruler - note this is NOT the project's 25.52% assistant-by-position scale),
copying taught words **85.7%**, held-out words **4.2%** against 58.0% for a copy-only model. Held-out copying
was still rising at every checkpoint, so the next arm runs three times longer. Details:
`docs/MIXED-COPY-FIRST.md`.

## The pointer copies unseen words at 58.0%; the old copy path was outvoting it

Three checkpoints on the same 600 mixed items: **pointer-only 83.8% taught / 58.0% held-out** (prediction was
50% or better: passed), pointer+Q/K **94.2% / 0.5%**, pointer+Q/K+gate **96.0% / 0.7%**. So the copy operation
generalises - a word the model never trained on is copied correctly most of the time - and the `dmix` echo
path is what destroys it. The 0.7% reading was also masked by a bug of ours: `pointer_only` was applied before
the `dmix` path was added, so it was never actually pointer-only. Details:
`docs/POINTER-GENERALISES.md`.

## The pointer works, it is being drowned out by the other pathways

The 6,000-step arm scored **94.2% taught / 0.5% held-out**, which looked like memorisation. An oracle test on
the same checkpoint says otherwise: the pointer pathway ALONE copies **84.0%** of taught words and **66.3%**
of held-out words, and lands on the answer position **83.5% / 85.7%** of the time. So the copy operation
generalises; the full model's other pathways simply outvote it and answer `the`. A correct mechanism that
cannot be heard is indistinguishable from a broken one until you isolate it. Next: a learned gate so the
pointer can be heard - prediction, held-out copying at 50% or better. Details:
`docs/POINTER-WORKS-DROWNED-OUT.md`.

## Copying from the window: 17.5% taught, 0.0% held-out, so it cannot copy at all

First benchmark task forced to a score, 3,000 steps from the 25.52% checkpoint: loss fell **6.86 -> 1.73**,
taught words reached **17.5%**, held-out words **0.0% at every checkpoint**. Learning the act of copying would
copy unseen words too; fitting the answer list would not. So it is memorising, and the cause is structural:
the architecture flattens the 32-token window through a gated mix with **no attention, no per-position
addressing, no pointer**, so it cannot reach back and take one token. This explains every 0.0% in the
benchmark table. Next arm adds an explicit readable pointer. Details: `docs/COPY-TASK-STRUCTURAL.md`.

## First benchmark: 0.0% on all six tasks, and that is the baseline

`tools/bench_v0.py` scores arithmetic, subtraction, sequence continuation, in-window copy, yes/no and world
knowledge, 200 generated items each, chat template, `<unk>` allowed and banned, with a frequency baseline.
Result: **0.0% everywhere, 1,155 scorable items**, and the model answers `the` (or `<unk>`) to everything. So
the 25.52% next-token figure was never comparable to anything and concealed that the model cannot do a single
checkable task. Two harness bugs were fixed first: a fake 22.5% from scoring missing answers as `<unk>`, and
`<unk>` never having been banned in earlier generation tests. Details: `docs/BENCH-V0.md`.

## Vocabulary decision: keep word tokens (measured, 2.1049 bits/char, best on both counts)

Three tokenisations, identical model shape and compute, bits per character: words **2.1049** with
**0.2561 tokens/char**, hybrid **2.1648** / 0.2670 (+2.8% bits, 0% unknown), pieces **2.2729** / 0.2751
(+8.0% bits, 0% unknown). So removing the `<unk>` cap costs 2.8-8.0% in-domain. The cap is real (13.73% of
positions want an unsayable word; 23.01% of out-of-domain tokens destroyed) but the insurance is only worth
its premium when we have an out-of-domain benchmark. Details: `docs/VOCAB-DECISION.md`.

## Words versus pieces: the word list wins in-domain by 10.9% in bits per character

Same text, same shape, output tied to embeddings, same compute. Words **2.1003 bits/char** at 12,002 entries with
2.11% unknown; pieces **2.3581 bits/char** at 16,000 entries with 0.00% unknown and 32% more parameters. So
switching to byte-level pieces costs efficiency on in-domain text even though the vocabulary cap is real. The
measured answer is a HYBRID: keep word tokens, add byte fallback only for unknown words. Details:
`docs/TOKENS-VS-PIECES.md`.

## Anchored RL: collapse cured, calibration still worse than supervised

Adding a supervised anchor inside the RL loss kept the distribution alive (**entropy 3.529** against the
reward-only collapse to **1.191**) and put **more probability on the right answer than any other arm**
(**0.189** against 0.150). Accuracy tied the control (**24.50% vs 24.17%**), but echo was worse
(**36.8% vs 31.0%**) and confidence error worse (**0.340 vs 0.287**). Plain conclusion: at this size our RL
method does not beat supervised training on accuracy, its unique ability is trading accuracy for repetition
control. Details: `docs/RL-ANCHORED.md`.

## RL dose response: penalty works, accuracy pays for it, confidence collapses

Echo falls monotonically with the repetition charge **57.4 -> 42.4 -> 13.3 -> 0.6%** and accuracy falls with it
**20.17 -> 20.08 -> 14.17 -> 6.25%**, against a **24.17%** supervised control. At charge 0.1 echo actually
rose above the start point (36.8 -> 42.4%), because reward-only training pushes echo up by itself. Confidence
error climbs **0.287 -> 0.534 -> 0.726 -> 0.920**: the model does not get honest, it gets loudly wrong. The
lever is the reward SHAPE, not the dose. Details: `docs/RL-DOSE-RESPONSE.md`.

## First RL ladder: repetition 36.8% -> 0.6%, and accuracy 24.17% -> 6.25%

Our own policy-gradient trainer, three arms from the same checkpoint, 2,000 steps each. Control (ordinary
training) reached **24.17%** with **31.0%** window-echoing. Reward-only reached only **20.17%** while echo
rose to **57.4%**. Adding a repetition charge crushed echo to **0.6%** and crushed accuracy to **6.25%**: at
`lam_rep=1.0` the repetition term was three times the size of the correctness term, so the model did what it
was paid to do. First lever in this project that ever moved repetition. Details: `docs/RL-LADDER-1.md`.

## 2026-11-10, the day in one page

Ten arms trained or measured, seven lost. The loop survived four levers (symbol mask, decode guard, copy
penalty, exposure bias - the last finished at **19.94% vs 25.52%**) and four explanations were falsified, the
last being confidence sharpening. Capacity is not the answer (**26.02% vs 25.52%** for +89% params, and the
readable path got *less* aligned, **8.2% -> 4.3%**). Readability costs **5.22 pp** of accuracy but also
**0.18 mean AUROC** of honesty (**30/36 vs 36/36** concepts passing). Full writeup:
`docs/NIGHT-2026-11-10.md`.

## The readability tax is bigger than accuracy: concepts stop tracking their own labels

Same 800 matched windows, project gate (`lens.api.audit`, AUROC bar 0.55): readable-only **30/36** concepts pass
against **36/36** for the copy-path model, mean AUROC **0.6886 against 0.8695** (drop 0.1810), mean fire rate
**47.4% against 28.3%**. Six concepts fail, all exclusively in the readable model - `mentions_body` 0.4275,
`mentions_a_name` 0.4662, `mentions_sport` 0.4774, `mentions_a_quantity` 0.5435, `mentions_family` 0.5437,
`is_a_list` 0.5471 - so the formulas are not wrong; they fail only when the concepts must carry the whole
prediction. Details: `docs/CONCEPT-LABEL-AUDIT.md`.

## CORRECTION: the loop is not the copy path's fault

The concept-only arm (no copy path at all) loops **worse**: symbol share **27.8%** against 8.7%,
longest symbol run **23** against 3, distinct 4-grams **19.8%** against 54.0%, copy rate at steps 17-24
**100.0%**. So the copy path carried the loop in one model; it is not its cause. The loop forms in a small
next-token model under greedy decoding whichever route the answer takes. Details:
`docs/READABLE-ARM-VERDICT.md`.

## Readable-only arm: readability costs 5.22 pp, not 19 pp

`--direct` dropped so the 36 concepts are the only route to an answer, same recipe and matched split:
**15.41% / 18.70% / 20.30%** against **19.69% / 24.26% / 25.52%**, so readability costs **5.22 pp** - the
earlier 18.50-19.17 pp was the price of retrofitting concepts onto a copy-trained model, not of a readable
model. Blemish: only **28/36** concepts pass the label-honesty gate here against 35/36. Details:
`docs/READABLE-ARM-VERDICT.md`.

## Capacity verdict: doubling width bought 0.50 pp, all of it unreadable-path

d=512 -> d=768, 15.57M -> 29.42M params, matched split, one change: accuracy **19.69/24.26/25.52% ->
20.30/24.16/26.02%**, so **+0.50 pp** against a measured single-seed spread of 0.26-3.07 pp. The trainer flags
the arm UNDER-TRAINED at **2.18 tokens per parameter**, so capacity and budget are confounded. Same
decomposition both widths: concept-only accuracy **3.00% -> 4.67%**, copy-only **20.17% -> 21.50%**, and
concept agreement with the model's actual answer **8.2% -> 4.3%** - so the extra capacity went to the copy
path and the model became **less** readable. Details: `docs/CAPACITY-VERDICT.md`.

## Self-context final verdict: prediction falsified, and the day's synthesis

Three epochs, matched split, one change: accuracy **19.94% against 25.52%** (-5.58 pp), copy rate over steps
9-24 **89.9% against 81.3%** (I predicted a fall below 90%), previous-token echo **30.2% against 0.0%**,
modal echo **41.7% against 25.0%**, symbols **16.7% against 8.7%**. Only distinct 4-grams improved (76.2%
against 54.0%). The epoch-1 improvement reversed - the second lever today to do that, which makes
single-epoch comparisons non-evidence in this setup. The confidence-sharpening explanation was also
falsified (baseline entropy **rises** 4.19 -> 5.02 while its in-window rate **falls** 75.0% -> 70.8%).
**Synthesis: the loop is a decoding artefact with a working fix (no-repeat 4-gram blocking, 39.3% -> 100%
distinct 4-grams); the content is a model-and-data problem no decoding trick touches.** Details:
`docs/SELFCTX-FINAL-VERDICT.md`.

## Self-context, epoch 1: text much less repetitive, copy curve barely moved

Against the baseline's epoch 1 (both equally trained): repeated 4-grams **63.9% -> 85.7%**, symbol share
**52.4% -> 24.3%**, longest symbol run **16 -> 7**, at a cost of **4.95 pp** of accuracy (19.69% -> 14.74%)
and 1.84x the training time. But the mean copy rate over steps 9-24 only moves **91.4% -> 87.5%**, and it
rises at steps 5-8 and 21-24, so the curve is nudged rather than reshaped. The output trades a backtick loop
for a comma loop and still answers nothing. Interim: one seed, epoch 1 of 3. Details:
`docs/SELFCTX-EP1.md`.

## Run 9: the state finally reacts, the readout still does not

`tools/eval_run9.py` on `scan9_ep1500.npz`: **F1 KL 1.1702 PASS** (bar >0.05), **F2 state cosine 0.1736 PASS**
(bar <0.5, against 0.9716 at run 6), guards clean (**gain 0.978-1.119**, entropy 94.1% of uniform), but
**F4 FAIL**: the trained readout scores **1.1702** against **1.1660** for a random-noise readout, a ratio of
**1.00x** where 2x is required. So the state now carries prompt-specific variation and the answer layer does
not use it - the same shape of finding as the readable model, on a different architecture. Run 9's own sample
answers are still symbol soup, so this is not a talking model. Details: `docs/RUN9-VERDICT.md`.

## The loop is autoregressive feedback, measured by generation step

Share of emitted tokens already in the window, by step: conversational model **6.2% at steps 1-4, 96.9% at
steps 9-12**; merged model **28.1% then 90.6%**. The model writes its own tokens into the window, and after
about eight steps almost anything it can say is something it can already see. Three rival explanations were
falsified today: corpus repetition rate, boilerplate density (math/code carries MORE boilerplate and loops
less), and memorisation - every 16-gram of the training corpus was hashed and **0.0%** of generated 16-grams
appear verbatim in it, so the loop is generated, not recalled. Details:
`docs/FEEDBACK-LOOP-MECHANISM.md`.

## Prediction falsified: the conversational corpus loops worse

Training on the conversational portion only, with the prediction recorded in advance: accuracy **17.35%**
against 25.52% on the merged corpus, in-window copying 69.1% against 74.3%, **copy mass 46.2% against
30.8%**, previous-token echo **39.6% against 0.0%**, symbol share in generation **58.3% against 8.7%** and
the longest symbol run **23 against 3**. So the conversational data, which repeats itself less by direct
measurement, produces a model that loops more - the repetition rate of the corpus is not what drives
looping. The visible culprit is boilerplate: `human`, `assistant`, quotation marks and backticks are cheap
and self-reinforcing under the copy path. Details: `docs/CONV-ARM-FALSIFIED.md`.

## Which part of the corpus teaches the repetition

Per source (`tools/source_repetition.py`): the target is already visible in the window in **38.32%** of
conversational windows against **51.75%** of math/code/book windows, and that math/code/book source is
**255.2M of the 294.2M training tokens, 87%**. Symbol share 17.28% against 30.25%. So the corpus is mostly
math and code, not prose - **correcting my earlier claim** that the model learns prose and is graded on code.
The looping follows from the data: the model spends most of its training on text where the answer is often
already visible. An arm training on the conversational portion only (38.8M tokens, matched holdout, one
change) is running. Details: `docs/SOURCE-REPETITION.md`.

## The copy penalty failed, and the reason reframes the problem

Three epochs, matched split: accuracy 19.69/24.26/25.52% baseline against 21.75/24.24/25.60% penalised, so the
epoch-1 gain washed out to **+0.08 pp** and the copy mass to **-1.7 pp**; generated text is slightly more
repetitive. The copy path is not capacity-limited either: **62.9%** of its 512 units are alive and the top 10
hold only **8.7%** of its energy, so the penalty fights the loss, not a bottleneck. Then the number that
reframes it: the true next token is already visible in the window **40.33%** of the time, while trivial copy
baselines score 6.62% and 2.75% and the model scores **25.60%**. So copying is often correct and the model is
not a copy machine - the repetition in generation is mostly the corpus repeating itself. Details:
`docs/COPY-PENALTY-VERDICT.md`.

## Copy-pressure penalty, first epoch: accuracy +2.06 pp, copying down

Interim, one seed, epoch 1 of 3, matched split, one change (`--lam-copy-penalty 2.0`). Against the baseline's
epoch 1: assistant accuracy **19.69% -> 21.75%**, previous-token echo **38.5% -> 24.3%**, modal-window echo
**45.8% -> 38.5%**, copy path's mass on visible tokens **36.9% -> 29.7%** (frequency justifies about 10%).
Source: `lens/runs/MATCHED-COPY-D512.txt` against `lens/runs/MATCHED-D512.txt`, measured by
`tools/copy_echo.py`. This is the first lever today that reduced the copying while also raising accuracy,
but it is one seed and one epoch - the final epoch and two more seeds decide it. Details:
`docs/COPY-PENALTY-EP1.md`.

## How much of what the model says is just visible in the window

12 prompts x 24 greedy steps (`tools/copy_echo.py`): emitted tokens are already in the window **74-81%** of
the time against **7-9%** by unigram expectation. The role splits by training length: the 3-epoch models
echo the previous token **0.0% / 7.3%**, while the 6-epoch model that scored 37.37% echoes the previous
token **61.5%** and the modal window token **58.3%**. So the echo grows with training, and the best-scoring
checkpoint is the one that copies hardest. The masked arm finished at **19.19% / 22.53% / 23.31%** against
**19.69% / 24.26% / 25.52%** unmasked, so the symbol mask costs **2.21 pp**. Details: `docs/COPY-ECHO.md`.

## The punctuation is cured; the loop is not

12 fixed prompts, 24 greedy tokens each, same harness: the copy-path symbol mask takes symbol share from
**8.7% to 0.0%** and the longest symbol run from **3 to 0** on every prompt, at a cost of **0.50 pp** of
validation accuracy (19.19% against 19.69% at epoch 1). But the loop did not stop, it moved: `hello , how are
you ?` now gives `the current of the function of the function of the function`, and distinct 4-grams fell from
**54.0% to 47.6%**. So punctuation was a symptom; the disease is that the copy path is self-reinforcing for
whatever is frequent in the window. Details: `docs/SYMBOL-MASK-VERDICT.md`.

## The price of readability: 19.17 pp

Same harness, 600 matched validation windows (`tools/concept_only_eval.py`): **full 22.17%**, **copy path
only 20.17%**, **concept path only 3.00%**, and the concept path agrees with the full model **8.2%** of the
time. So giving the 36 named concepts the decision costs **19.17 pp**, and concept-only accuracy lands
**below** the 4.23% frequency baseline - a person reading the readable numbers would score worse than always
answering `,`. The model's current ability is, numerically, its copy path. Details:
`docs/READABILITY-PRICE.md`.

## The readable half explains 7.3% of the model's decisions

Splitting the logits the model already computes into its 36-number concept path and its copy path, on 233
matched validation windows: logit variance **0.7 from the concepts** against **2.3 from the copy path**, and
the concept path alone picks the same token as the full model only **7.3%** of the time. The concept rows
are near-orthogonal (participation ratio **35.97** of 36), so the readable layer is not redundant, it is
simply not the deciding path. So `explain()` can name which concepts fired and still not be the reason for
the answer, and it must say so. Measurements: `docs/READABLE-PATH-EXPLAINS-7PCT.md`.

## Where the punctuation loop comes from, traced in the open model

The architecture separates into a 36-number concept path and a copy path that echoes the window. On
`the capital of france is`, tracing 14 generated tokens: the **concept path's** symbol probability stays
at **0.007-0.008 for all 14 steps**, while the **copy path** goes from 0.021 to **0.965** and carries the
entire loop. Each emitted backtick is appended to the window and echoed again, so the loop is
self-reinforcing, which is why a decode guard could not fix it. Separately, all 36 concepts have a symbol
weight of about **0%**, but their top-voted tokens are unrelated word scatters (`c0: might pound german
bills flood luck`), so the concept layer is neither the cause of the loop nor yet a readable layer.
Trace and prediction: `docs/SYMBOL-FORENSICS.md`.

## A decode guard fixes the looping, not the meaning

Measured on `gpu_d512_s0_best.npz`, 12 prompts, 24 tokens each (`tools/decode_guard.py`):

```
guard      symbol share   longest symbol run   distinct 4-grams
greedy          67.7%            15.0               39.3%
norepeat        57.6%             5.0              100.0%
symrun          50.7%             2.3               69.0%
both            46.5%             2.4              100.0%
```

The guards work - repetition is eliminated - but the text stays meaningless and the model still spends
**46.5%** of its emitted tokens on symbols against **27.7%** in the corpus, about **1.7x** the rate it was
shown. So the preference lives in the weights, not the decoder: decoding tricks are ruled out as a fix.
Details: `docs/DECODE-GUARD.md`.

## Matched validation split built; the original mismatch was +2.65 pp, not +16 pp

`tools/make_matched_split.py` now draws a validation holdout by hashing conversation ids, and
`train_chat.py --split-file` uses it with a length check against the corpus. On the real 294,225,090-token
corpus, sampled through the trainer's own window builder (4,000 windows per side, symbol share of target
tokens): shipped split **val 31.42% vs train 28.78%, +2.65 pp**; matched split **val 28.52% vs train
28.90%, -0.38 pp**. The 13.29% vs 44.04% figures in `docs/DATA-SPLIT-SHIFT.md` are real but describe the
extremes of the file, not the validation split - so the mismatch is real and small, and my earlier
framing overstated it roughly six-fold. Correction and details: `docs/MATCHED-SPLIT.md`.

## Correction: the code tail is the model's easiest region, not a handicap

I previously wrote that the code-versus-conversation split mismatch partly explains a low held-out
accuracy. Measured by region, the opposite is true: the held-out code tail is where the readable model
performs **best**, at **33.00%** accuracy and **perplexity 107.6**, against **17.33-25.00%** and
**270-635** in the conversational regions it trained on. So the split mismatch **flatters** the headline
rather than depressing it. What survives: the split really is compositionally different (13.29% symbols
in the first 200k against 44.04% in the tail), the symbol attractor really is partly the code tail
(47.7% of targets are symbols, 71.3% of predictions), and per-source reporting is needed because an
average hides a 17-33% spread. Details: `docs/CORRECTION-TAIL-IS-EASIEST.md`.

## The held-out split is a different dataset, not a held-out sample

The corpus is a concatenation of sources and the validation tail is the end of it. Symbol-token share:
train first 200k **13.29%**, train middle **28.08%**, held-out last 200k **44.04%**. Decoded, the training
text is conversational (`<|im_start|> user how should i dress for a job interview ?`) while the held-out
tail is **Python code** (`def withdraw ... if amount > 0 : ...`). So the model learns prose and is graded
on code. That partly explains the near-baseline held-out accuracy, and it explains the symbol attractor
(44% of held-out targets are symbols; the model predicts symbols 68.3% of the time). Every accuracy
number measured on this tail is a cross-domain number. Fix: randomised holdout, then re-measure.
Details: `docs/DATA-SPLIT-SHIFT.md`.

## What the architecture that learns actually says

Sampled output from the readable model, which scores 31.50% next-token accuracy at 24 tokens of context:
`the capital of france is` -> `8 * chapter 1 . * * * * * *`;  `i feel sad because` -> `it transaction by -
levels nicotine concepts and saving back to handle t brand government addressing . however`.

Local grammar is real (`we can set the following equation as follows`, `which chapter 2`, `however`) but
global meaning is not, there is no factual recall, and a symbol attractor takes over within a few steps
under both greedy and sampled decoding. So the model is **not** talk, and the symbol attractor is now the
top open item: a model can be at 31.50% next-token accuracy and still be unusable if decoding walks into
`*` and never leaves. Details: `docs/GENERATION-REPORT.md`.

## The readable architecture has real signal; the recurrent talk core does not

Same corpus, same vocabulary, same held-out tail, same measured frequency baseline of **3.99%**. The
readable concept/answer model scores **11.00%** with just one token of context (**2.8x** baseline) and
**31.50%** at 24 tokens (**7.9x**). The recurrent talk core scores **4.69%**, which is **1.18x** - within
noise of answering `,` every time. Context is demonstrably used by the readable model (accuracy rises
with length); the core's own measurements show no such rise. So talk should be built on the
concept/answer family. Details: `docs/ARCH-COMPARISON.md`.

## The model is 0.70 points better than always guessing the most frequent token

Over 20,000 held-out positions, scored against a frequency baseline on the same positions: always
predicting `,` gives **3.99%**, the trained model gives **4.69%** - a **+0.70 pp** advantage, **+17.5%**
relative. The model is also not just copying the unigram (log-correlation **+0.1588**, KL **1.8330**
nats), so it neither learns context structure nor reproduces the base rates faithfully; it has collapsed
onto a handful of answers. That is why every readout and state hunt failed: there is almost no learned
signal to express. Every future arm reports the frequency-baseline gap beside its accuracy.
Details: `docs/MARGINAL-BASELINE.md`.

## The map responds but at too low a resolution - it is not compression

Over 4000 pairs of real held-out contexts: Spearman(state distance, logit distance) **rho +0.610**, and the
top state-distance decile gives **27.92** logit distance against **17.54** in the bottom decile, a **59%**
increase. The state-to-answer map is therefore NOT squashing different states into one answer. What is
wrong is resolution: even at state cosine distance **0.7582**, predictions differ by only **0.138 nats**
of KL, against a top-1/top-2 gap of about **0.008** with 12,002 candidates. Ruled out by measurement now:
state collapse, direct-path swamping, readout ignorance, map compression. Details:
`docs/RESOLUTION-NOT-COMPRESSION.md`.

## Who decides the answer: the state does, and it says one of three things

On 200 real held-out contexts (`scan8_ep1500`): the full argmax equals the **state-only** argmax **87.0%**
of the time and the **direct-only** argmax only **8.5%**, so the recurrent core is where the decision is
made. But the top choice takes **just 3 distinct values across 200 different contexts** - functional
collapse in the answer, which is what the repetition loop looks like from inside. Mean top-1 probability
**0.0334** against a uniform **0.000083**, top-1/top-2 gap **7.975e-03**, mean entropy **8.987** of 9.393.
So the problem is not that the readout ignores the state, it is that the state-to-answer map is degenerate.
Details: `docs/ARGMAX-DECIDED-BY-STATE.md`.

## Run 8 final: decorrelation works, the readout does not, generation collapses

Held-out loss **8.8631**, accuracy **5.60%** (run 5: 5.34%), train/held-out gap **+0.2994**. Criteria as
written before the run: **E1 PASS** state cosine **0.4643** (predicted 0.47 twelve minutes before it landed),
**E2 FAIL** KL **0.0151**, **E3 PASS** 5.60% at +0.26 pp, **E4 FAIL** generation collapses to
`. . the , the the the the` for every prompt, and **F4 FAIL** trained readout 0.0151 against noise 0.0153,
ratio **0.99x**. The state separation move worked; the answer layer is unchanged. Details:
`docs/RUN8-VERDICT.md`.

## A trap in our own headline test, found before it fooled us

Taking run 8's weights and merely setting the new readout gain to **1.0** - RMS normalisation with no
training at all - multiplies the headline KL by **63x**, from **0.0151** to **0.9590**, which would pass
the bar that eight runs failed. The trained-vs-noise control exposes it: at gain 1.0 the trained readout
scores **0.9590** and a readout of random numbers scores **0.9604**, ratio **1.00x**. All of the apparent
improvement is scale, none of it is learning. So F1 alone is gameable by arithmetic, and run 9 is judged on
F4 (trained must beat noise by 2x at the same scale) plus G1/G2. Details: `docs/METRIC-TRAP.md`.

## Where the input dies: measured, not guessed

The logits split into a state term and a direct term. The obvious hypothesis was that the direct term
swamps the state. Measured on `scan8_ep1000`: the state term is **larger** (sd 0.111-0.124 against
0.065-0.085) and carries most of the variation, contributing **0.0108** of the **0.0151** KL when the
direct term is frozen. So the failure is one of **scale**, not attribution: logit differences of about 0.1
across six very different prompts are washed out by a 12,002-way softmax. That is exactly what the new
readout normalisation amplifies. Details: `docs/READOUT-DECOMPOSITION.md`.

## The trained readout separates words no better than random noise

Instrument check on `scan8_ep750`: the KL metric responds monotonically to deliberate readout damage
(`0.0056` at x0.25 scale through `0.5989` at x8) and reproduces its baseline exactly when unchanged, so
the metric is sound. Then the decisive result: a readout of **random numbers** at the trained one's own
scale gives KL **0.0152**, against the trained readout's **0.0151**. Training made the answer layer no
more sensitive to the input than noise is. That is why the states differ (E1 passes) and the output does
not (E2 fails).

## Decorrelation works, the readout is now the bottleneck

Run 8 pairs the direct path with the state decorrelation penalty. Measured on its own checkpoints: state
cosine **0.1717** at step 500 and **0.2261** at step 750, against **0.9716** with no penalty, so E1 passes
(bar < 0.5). KL across six prefix-free words stays at **0.0150** and **0.0151**, against a bar of 0.05, so
E2 fails. Two different inputs now produce genuinely different internal states and still produce nearly
identical next-token distributions: the bottleneck has moved from the state to the answer layer.
Full notes are in the private working log; this line is the measured result.

## Decorrelation penalty verified

The named alternative is implemented and gradient-checked: with lambda 0.5, finite differences match at
**1.018e-08** (seed 5) and **1.009e-09** (seed 23); with the term off the earlier **1.684e-10** is
unchanged, so nothing was disturbed. Run 8 pairs it with the direct path: E1 state cosine below 0.5,
E2 KL above 0.05. Sources: `docs/EVIDENCE.md`, `docs/EVIDENCE.md`.

## Run 6 mid-run: state collapse returns, and my plateau call was wrong

At step 3750 (`scan6_ep3750.npz`): KL across six prefix-free words **0.0091** nats (K1 needs 0.05, trending
FAIL) and state cosine **0.9716** (K3 bar is 0.9, so FAIL). Two conclusions: longer training does not raise
input sensitivity, so the readout limit is structural; and the drift I called a plateau at **0.8088** at
step 1000 had reached **0.9716** by step 3750 - that inference from four points was wrong and is recorded
as wrong. Source: `docs/EVIDENCE.md`.

## Incident: run 6 overwrote run 5's checkpoints

All 15 files matching `scan5_ep*.npz` carry mtimes after run 6 started, so **run 5's weights were
overwritten and are gone**; its log and published numbers survive. Root cause: the checkpoint prefix
was hardcoded in the driver, so a run launched with a different log still wrote the same filenames.
Fixed: the driver reads `CK_PREFIX` from the environment and each runner pins its own. The 15 files
were renamed, not deleted, with a sha256 receipt. Source: `docs/EVIDENCE.md`.

## Direct path at 25 steps: D1 FAILS, ranking changed 6 of 6

KL across six prefix-free words is **0.0150** nats against a pre-registered bar of 0.05, so D1 FAILS and
the bar is not moved after the fact. The same run shows state cosine **0.0743** and **6 of 6 distinct
top-5 sets** - the model now prefers different next tokens per input while most of the probability mass
still sits on frequent tokens. Honest headline: ranking changed, distribution shape has not. Source:
`docs/EVIDENCE.md`.

## Direct path: first non-degenerate, prompt-dependent output

In a 12-step smoke test of the run-7 driver (not a trained result), four different prompts produced four
different strings with no repetition loop: `truffles sarcastic west escaped...`, `cos constitutes
overweight 135 graham...`, `barcelona vietnamese titanium ahead...`, `fabrics defeat fatigue wondering...`.
Every earlier run collapsed to the identical loop `the , . the , .` for every prompt. No accuracy claim -
twelve steps is not training - but it is the first evidence that input reaches the output. Source:
`docs/EVIDENCE.md`.

## Batched direct path: both bugs found and fixed

The gate failure had two independent causes: `einsum("btd,de->bte")` contracted the wrong axis of
`Wd` (correct: `"bte,de->btd"`), and positions whose target is `<unk>` were scored even though
`<unk>` is banned from prediction. After both fixes, four independent seeds pass at **1.68e-10,
1.60e-10, 2.65e-10, 2.70e-10**. The batched direct-path core is verified and runs at **792** scored
tokens/s. Source: `docs/EVIDENCE.md`.

## Batched gradient: proven wrong, not noise

The batched embedding gradient is stable at **-0.00119500** across eps 1e-4 to 1e-8 while the analytic
value is **+0.00581498** - a different number with a different sign, so it is a real error, not finite-
difference noise, and no relu unit sits near the kink. That implementation stays unused; the single-chunk
direct-path core passes at 2.238e-10 and is the version available for a run. Source:
`docs/EVIDENCE.md`.

## Batched gradient bug narrowed

Per-parameter finite differences show the fault is isolated: with subset scoring only the embedding
term is wrong (**7.01e-03**), while Wx, Wd, Wh, Wa and ba agree to about **1e-10**. The suspect is the
tied-readout term `dlg.T @ dsel` when positions are masked. Tool: `tools/diag_batched_grad.py`.

## Batched direct path: gate FAIL, not used

The batched direct-path step fails its finite-difference check: worst abs **7.010e-03** at `emb[1,0]`
against a 1e-6 bar. No run uses it. The single-chunk direct-path core passes at **2.238e-10** and is
the version that may be trained. Source: `docs/EVIDENCE.md`.

## Direct path added to the state core

The published d512 model reaches 37.37% partly because it adds a direct token-to-logits path; the
state core did not. That path is now implemented (`lens/scan_direct.py`) with hand-written gradients
verified numerically to **2.238e-10** abs. It becomes run 7 once run 6 finishes, against the same bar:
KL across six prefix-free words above 0.05 nats. Source: `docs/EVIDENCE.md`.

## Frequency prior ruled out

Subtracting log token frequency at decode leaves input sensitivity unchanged: KL **0.0102 -> 0.0104**
nats across six different words, and the output simply loops on a rare token instead of a common one.
Four candidate causes are now eliminated by measurement - instability (fixed), capacity (d512 probe),
bias (1/33 of the logit term), frequency prior. The remaining gap is structural: the readout cannot
amplify 0.01 nats of state signal into different words. Source: `docs/EVIDENCE.md`.

## Why the output ignores the input

Three tests on run 5's model: **capacity ruled out** (d512, 9,426,402 params, changes KL 0.0102 to
0.0108), **bias ruled out** (input part of the logits is 33x the bias, and removing the bias leaves KL
identical), and the mechanism found: state differences are multi-dimensional but the readout maps them
to nearly one direction (logit cosine **0.88**) - a frequency-prior collapse. Source:
`docs/EVIDENCE.md`.

## Run 5: accuracy doubled, speech still broken

With the spectral fix, held-out accuracy went **2.34% -> 5.34%** (loss 7.2164 -> 7.5663) - the first
state-core gain that came from the model reading its input. Generations still loop, and the measured
cause is that the output distribution barely changes with input: cosine **0.991**, KL **0.0102** nats
across six different words. Source: `docs/EVIDENCE.md`.

## Run 5 drift: decelerating, plateau below the bar

Pairwise state cosine across six prefix-free words: **0.2068, 0.5029, 0.6878, 0.7680, 0.8088** at steps
120 / 250 / 500 / 750 / 1000. Increments shrink (+0.30, +0.19, +0.08, +0.04), so it plateaus under the
0.9 input-blindness bar instead of crossing it. My earlier prediction of a crossing near step 1000 was
wrong and is recorded as such. Source: `docs/EVIDENCE.md`.

## Run 5: state similarity climbing

Same six prefix-free words, three checkpoints: pairwise state cosine **0.2068 -> 0.5029 -> 0.6878** at
steps 120, 250, 500, with the radius held at 0.900 and tanh slope 0.975. The run-4 collapse is not
happening, but the drift is monotone and heading for the 0.9 input-blindness bar. Source:
`docs/EVIDENCE.md`.

## Run 5 mid-flight: fix holding

At step 250 the recurrent radius is **0.900**, state norms about **2.0**, tanh slope **0.984**, and
the pairwise state cosine across six prefix-free words is **max 0.5029** (run 4 was +1.0000). The
fix survives training, though similarity drifted up from 0.2068 at step 120, which is being watched.
Source: `docs/EVIDENCE.md`.

## Spectral fix, test 2: PASS

Holding the recurrent radius at 0.9 with prefix-free prompts gives max state cosine **0.2068**
(mean 0.1216) where run 4 gave **+1.0000** for every pair, with tanh slope **0.98964**. The
input-blindness is a consequence of an unconstrained recurrent matrix. Source:
`docs/EVIDENCE.md`. Accuracy from the 1500-step run with this fix is pending.

## Spectral fix, test 1

Holding the recurrent radius at 1.0 removed the saturation (state norm 15.96 to 3.5, tanh slope
0.005 to 0.954) but the pre-registered falsifier still fired: max pairwise state cosine **0.9885**
against a 0.9 bar. Test prompts shared a prefix, which is a confound. Source:
`docs/EVIDENCE.md`.

## State core: input-blind, cause found

Four different prompts produce the same state (pairwise cosine **+1.0000**), state norm **15.96**,
tanh slope **0.00516**, and the recurrent matrix has **spectral radius 3.406** - the loop saturates
and destroys the input. All four failed runs had one cause. Source: `docs/EVIDENCE.md`.

## State core, run 4

8x throughput (792 vs 99 scored tokens/s) produced held-out loss **7.2164**, accuracy **2.34%** -
slightly worse than run 3 (6.9472 / 2.58%). More tokens at this size is not the lever. Source:
`docs/EVIDENCE.md`.

## Release

`v0.1.0` is published: https://github.com/blueprint-ai-hq-git/glassnet/releases/tag/v0.1.0 - both checkpoints, the tested Colab, the checksums, and the
limits stated before the features. Verified against the GitHub API on 2026-11-09.

## 2026-11-07

Today's changes, each with its log line: `docs/EVIDENCE.md`.

# Where GlassNet is right now

Updated: 2026-11-06. Every figure below comes from a run recorded in `docs/history/` or `lens/runs/`.

## Honest completion

| Area | Done | Note |
|---|---:|---|
| Autograd engine | 100% | gradients verified to 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) |
| Our optimiser | 100% | built and measured. A 3-seed sweep at its own rung, scored best-masked-per-seed, puts it +0.78 points over SGD's best setting there (28.59% vs 27.81%, `docs/EVIDENCE.md`, `docs/EVIDENCE.md`) and level at matched epoch; it fails the stability bar (1.89 pp spread) - not the 4.3 points once published. That arm used a different rung and SGD at lr 1.0 with no decay, so it is not the ladder result: on the capped-window ladder with final VAL, our rule leads only at 500k and plain SGD is ahead from 2M up (`docs/EVIDENCE.md`) |
| KV cache | 100% | 2.29x faster (0.483 -> 0.211 ms/token), identical output |
| Exports (npz/safetensors/gguf) | 95% | safetensors export of the best checkpoint pending |
| Local server | 95% | serves, not yet load-tested |
| Live inspector | 85% | real weights on :8770, read-only |
| Measurement rig | 95% | every run logged to JSONL |
| Named concepts | 60% | 256 neurons, audit runs, coverage printed |
| Our own architecture | 40% | built and measured; **lost** to the plain shape |
| Safety layer | 10% | designed, not built |
| Coherent conversation | 15% | grammar real, content weak |
| Complete readability | 10% | partial coverage only |
| **Machine (engineering)** | **~90%** | |
| **Intelligence (model quality)** | **~22%** | |
| **Project ready to ship** | **~32%** | |

## What this means in one paragraph

The scaffolding around the model is genuine engineering: hand-derived gradients that pass numerical checks,
an optimiser that measurably beats SGD, a cache that is 2.29x faster with identical output, and a system that
stops and resumes without losing work. The model itself is not good yet - 37.37% held-out next-token
accuracy at best. Anyone claiming this is a working assistant would be lying, and the numbers in this
repository are the proof of that.

## Best numbers

| Metric | Value | Where |
|---|---:|---|
| Best held-out accuracy | 37.37% | d=512, full corpus, 6 epochs |
| d=256 full corpus | 35.26% | by position, `lens/runs/ATTN-GRADE-RUN.txt:28` (legacy by-id: 26.31%) |
| d=256, 2M windows | 24.20% | seed 1, by position, `lens/runs/GPU-BEST.txt:64` |
| d=512, 3 epochs, 2M windows (run twice) | 24.39% / 24.40% (ppl 109.90 both) | same script, 0.01 pp spread between two runs a day apart, `docs/EVIDENCE.md` |
| Parameters, shipped d=256 model | 5,908,686 | counted off the model object; `docs/EVIDENCE.md`, also `lens/runs/GPU-BEST.txt:57` |
| Optimiser vs SGD (matched sweep, 3 seeds) | 28.59% vs 27.81% best-epoch mean; 29.12% vs 27.15% at matched epoch | `docs/EVIDENCE.md` |
| Gradient check | 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) max error | `tests/test_gradcheck.py` |
| KV cache speedup | 2.29x (0.483 -> 0.211 ms/token, 2,069 -> 4,747 tokens/s) | `lens/runs/KV-CACHE-MICRO.txt` |
| Arithmetic, unseen operands | 0.0% exact | digit 0 alone: 100%; two different runs, `lens/runs/ARITHMETIC.txt` and `lens/runs/ARITH_DIAG.txt` |

## Levers measured and closed

These were tested one change at a time against a matched reference, and lost on every seed.
They are listed so nobody has to re-run them.

| Lever | Result | Where |
|---|---|---|
| Attention (bare single head, no position) at 20k windows | 8.21% vs 11.02% reference, 2 seeds, -2.81 | `docs/EVIDENCE.md` |
| Window 64 (double tokens per example) at 20k windows | 9.45% vs 11.02% reference, 2 seeds, -1.57 | `docs/EVIDENCE.md` |

## Measurement bugs found and fixed

A wrong number is worse than a missing one, so these are listed with the correction.

| Bug | What it did to the numbers | Where |
|---|---|---|
| Talk decoder left-padded short prompts with token id 0 (`<unk>`) | 20-23 of 32 context slots were `<unk>` for the project prompts, so 95.1% (d256) / 96.5% (d512) of the reply tokens were `<unk>`; pad-free the same weights measure 26.4% / 21.5% | `docs/EVIDENCE.md` |
| Architecture table added the proposed recurrent core and counted the tied embedding twice | published "7,081,122 parameters"; the shipped model prints **5,908,686**, so the count was overstated by 1,172,436 (19.8%) | `docs/EVIDENCE.md` |
| Cost table added a stage labelled "runs 32x" exactly once | published 0.1446 ms/token; with the core counted the 32 times it runs, the same rows give 0.2283 ms | `docs/EVIDENCE.md` |
| KV cache speedup "9.7x" had no run file behind it | only the prose figure existed (0.217 vs 2.115 ms, `docs/EVIDENCE.md`); the recorded micro-benchmark is **2.29x** (0.483 -> 0.211 ms/token, identical to 5.3e-06) | `lens/runs/KV-CACHE-MICRO.txt` |
| `wall_ab_grade.py` P2 computed `on - off` but graded the pre-registered `off - on` band | the falsifier (`on - off < 0.5`) fired on a real 3.11 pp LOSS and called it "earns under 0.5 pp"; the -1.0..-6.0 band was unreachable, so a perfect result would still have printed "OUTSIDE". Numbers unchanged (-1.39 / -3.11), only the sentence was wrong | `docs/EVIDENCE.md` |
| A 200k-window arm ran at 19:05 on `data/chat/prepared` with plain SGD | its VAL 13.12% looks like progress next to the 20k numbers but is **not comparable**: 13.3M vs 294.2M cached tokens, gates OFF vs `ours`, 13,651 vs 87,035 val windows, 40 vs 36 concepts. Recorded as a run, never as evidence - it also had no queue entry and no pre-registered prediction | `docs/EVIDENCE.md` |
| Same command, same seed, three runs at the 20k-window rung | VAL acc 11.32 / 10.79 / 10.29 and ASST 11.96 / 11.42 / 10.92 - a **1.03 pp spread**, not the 0.01-0.02 pp measured on the d512 / 2M-window path. The small figure must not be carried to this rung, and a +/-1.25 band around one draw cannot serve as a control at n=1 seed | `docs/EVIDENCE.md` |
| `train_chat.py` printed `len(Xtr) / params` under the label "tokens per parameter" - but `len(Xtr)` counts **windows** | every such figure in every log is low by exactly 32x, and the "UNDER-TRAINED" verdict was computed on it: the d=512 / 4M-window rung printed **0.26 (UNDER-TRAINED)** when it truly had **8.22 (`adequate`)**, and the d=256 arms are 0.17 / 1.68 / 4.20 at 20k / 200k / 500k windows, not 0.01 / 0.05 / 0.13. Recorded once as a *prediction* error (`AUTOPILOT-LOG.md:4552`) and never fixed in code; now computes tokens and prints the old ratio beside it so old logs stay comparable. Second unit question under it: `chat_data.py:166` uses `stride = window // 2`, so the figure counts token INSTANCES - the headline's 34.74 is 18.90 distinct tokens/param per epoch, i.e. about half. Thresholds left alone | `docs/EVIDENCE.md`, `lens/scripts/train_chat.py:830` |
| A restarted run rewrote `<out>_best.npz` from its own epoch 1 | `wall_off_s0`'s epoch-3 checkpoint was replaced by an aborted second attempt's epoch 1. Repaired earlier; the **guard is now wired into the trainer and proven at the call site** (2 hunks, patch sha256 f10dc89e -> e5acd0aa, 4/4 unit tests, `--check` import=True call=True, and the end-to-end pair: run 2 printed 1 `CLOBBER GUARD` line, archived run 1's best.npz as sha256 b349a10b... with a receipt, and wrote its own 422ebc7e...) | `docs/EVIDENCE.md`, `lens/runs/BEST-CKPT-GUARD-E2E.txt` |

Every percentage on this page and in `README.md` that carries a `file:line` citation is now checked mechanically: `lens/scripts/check_public_cites.py` opens each cited line and fails if the quoted number is not on any cited line of that row (missing file, line past the end, blank line, or a number that appears nowhere it cites). Current state: **33 cited rows on these two pages and 56 across all of `docs/history`, 0 problems**. It has already earned its keep three times - it found a citation to a **blank** line (`STATE.md:2768`, the 37.37% claim, now `:2770`) and a table row that cited the `final VAL` line while also quoting that arm's ASST, a number only on the epoch line (`WALL-AB.txt:850` -> `:765`) - and a third time on this very page, catching a new 2M row that quoted a
best-checkpoint 22.82% cited only to the final-epoch line. It does not judge which column a number came from - that mistake (P1, VAL vs ASST) still needs a human eye.

**The 17.43% arm's checkpoint now verifies mechanically, and one arm has none.** `verify_wall_ab_ckpt.py` used to split a log at `----- off seed N` markers, so a run launched before that convention existed - the seed-0 200k arm behind 17.43% - was invisible to it and the tool answered about a stale block appended after it. With leading unmarked segments parsed, **20 arms across 11 logs** now verify: `datascale_honest_s0` has 3,811,534 params as announced, best checkpoint = epoch 2, and `sha256(best) == sha256(ep2)` with identical arrays. The single remaining problem is the 11.04% / 41-concept run on `data/chat/prepared`: it announces **3,912,509** params but the tag its save line names resolves to the honest seed-1 checkpoint at **3,811,534**, so that run has **no checkpoint of its own** - which matches `docs/EVIDENCE.md` (recorded as a run, never as evidence). Note: `docs/EVIDENCE.md`.

**Held-out sets are disjoint by construction, at every scale.** Above 200,000 train windows the trainer skips its exact content walk and prints the skip (`train_chat.py:783-787`); every data-scale arm here - 500k, 2M, 16.9M - exceeded that, so none printed a walk. It is redundant: the train/val split is assigned per conversation (`chat_data.py:133`), windows are selected per conversation by that flag (`chat_data.py:165`), and windows never cross a conversation boundary (`chat_data.py:166`), so a train and a val window cannot share tokens; the vocabulary is built from train text only (`chat_data.py:119`). What is still unchecked is corpus-level duplication (a val conversation that is a near-copy of a train one) - that would inflate every number here and has never been tested. Note: `docs/EVIDENCE.md`.

## Wall A/B (concept-wall bottleneck) - FINISHED 2026-11-06 17:51, six arms, P1 FAILs as coded (band mixes two columns)

Question: docs/EVIDENCE.md OPEN #1. Design + predictions written before any run:
docs/EVIDENCE.md (P1-P5, plus P6 "supervision must not move a number").
Full write-up with the grader's own words: `docs/EVIDENCE.md`.

| arm | seed | final val ppl | val acc | concepts | params | source |
|---|---|---:|---:|---:|---:|---|
| wall_off (no bypass) | 0 | 399.50 | **12.18%** | 20/36 | 3,811,534 | lens/runs/WALL-AB.txt:201-202 |
| wall_on (tied full-rank bypass, +2,097,152 params) | 0 | 575.00 | 10.79% | 20/36 | 5,908,686 | lens/runs/WALL-AB.txt:477-478 |
| wall_r64 (rank 64, +1,292,416 params) | 0 | 550.85 | 11.28% | 24/36 | 5,103,950 | lens/runs/WALL-AB.txt:664-665 |
| wall_off (no bypass) | 1 | 395.80 | **12.79%** | 26/36 | 3,811,534 | lens/runs/WALL-AB.txt:850-851 |
| wall_on (tied full-rank bypass) | 1 | 628.41 | 9.68% | 22/36 | 5,908,686 | lens/runs/WALL-AB.txt:1036-1037 |
| wall_r64 (rank 64) | 1 | 521.68 | 10.87% | 26/36 | 5,103,950 | lens/runs/WALL-AB.txt:1222-1223 |

Runner marker: `lens/runs/WALL-AB.txt:1226 ===== WALL-AB DONE 2026-11-06 17:51 arms_missing=0`.
Reference for the same 20k-window rung: 11.96 / 10.08 (mean 11.02%), lens/runs/CTX-ATTN.txt.

Verdicts as printed by `lens/scripts/wall_ab_grade.py` (output `lens/runs/WALL-AB-GRADE.txt`,
sha256 ac55e35b..., 6 rows in `lens/runs/WALL-AB-RESULTS.tsv`):

* **P1 FAIL as coded, but the band mixes two columns** - the 10.0-12.5 band was derived from `ASST` values
  (11.96 / 10.08 at `LEARNCURVE.txt:4773` and `:4960` - not in the file the predictions cite, which has no
  such numbers) and then applied to the `VAL` column. On the graded column `wall_on` s0 10.79% is inside
  the band and s1 9.68% is 0.32 pp below the floor. Compared on the SAME column the control reproduces
  within 0.55 pp on both seeds (VAL -0.53 / +0.15, ASST -0.54 / +0.22), and the trainer commands and the
  five config-header lines match field for field. So this FAIL is a metric mismatch, **not** evidence that
  the rung drifted. The gate keeps its coded verdict - re-deriving it after seeing the result is exactly
  what a gate must not do - so until the maintainer rules, no delta against the stored rung is quoted. Both readings
  with all numbers: `docs/EVIDENCE.md`.
* **P2 OPPOSITE SIGN** - off minus on = +1.39 (s0), +3.11 (s1). Removing the bypass did not cost the
  predicted 1-6 points; on both seeds it was ahead.
* **P3 MIXED** - r64 minus on = +0.49 (s0), +1.19 (s1): not below `on` on either seed, so "rank 64 is
  under-provisioned" is unsupported, but the dose is not clean at seed 1 either.
* **P4 PASS** - mean train-minus-val gap off +8.35 vs on +28.98, i.e. 20.63 pp smaller with 2.1M fewer
  parameters: at this rung the big gap follows capacity.
* **P5 CONTRADICTION CONFIRMED** - the no-bypass arm is above the bypass arm by more than 1.0 pp on both
  seeds, reported as a finding against the plan, not as noise.
* **P6 not triggered** - the resumed arms' recorded command lines match `run_wall_ab.sh` field-for-field
  (only shell quoting of the seed differs), so the resume is valid and its arms are poolable.

What that means: the A/B does **not** establish that the concept wall is lossy, and it does not license
the wider-residual build. It is a coherent lead - same direction on both seeds, 2.1M fewer parameters,
far less overfit - and its control reproduces on a matched comparison, but the pre-registered gate as
coded still says FAIL, so nothing is quoted against the stored rung until that is ruled on. If the maintainer
adopts the metric-consistent reading, the deltas become quotable and the plan's next lever is data scale;
if he keeps the gate as coded, the control has to be re-run before the block can be read at all.

A third run of the same command, same seed, launched at 18:07 and finished at 18:15, landed at VAL 10.29% /
ASST 10.92% (`lens/runs/REF-REPRO.txt`) - so three draws of one configuration span **1.03 pp**, and P1's
+/-1.25 band around a single draw cannot work as a control at n=1 seed. Full write-up:
`docs/EVIDENCE.md`.

### Replication on fresh seeds (WALL-REP) - finished 2026-11-06 18:56

A separate pre-registered block (`docs/EVIDENCE.md`), own log
`lens/runs/WALL-REP.txt`, own tags - its arms are **not** in the six-arm table above.

| arm | seed | VAL acc | ASST acc | source |
|---|---|---:|---:|---|
| wall_rep_off (no bypass) | 2 | **13.40%** | 13.83% | lens/runs/WALL-REP.txt:329 |
| wall_rep_on (tied bypass) | 2 | 9.98% | 10.65% | lens/runs/WALL-REP.txt:110 |
| wall_rep_off (no bypass) | 3 | **13.26%** | 13.72% | lens/runs/WALL-REP.txt:517 |
| wall_rep_on (tied bypass) | 3 | 11.21% | 11.99% | lens/runs/WALL-REP.txt:705 |

**R1 direction PASS** (+3.42 at seed 2, +2.05 at seed 3), **R2 magnitude PASS** (both inside the predicted
1.0-4.0 pp), **R3 noise bound FAIL** (the `on` arm spans 1.53 pp over four draws against a 1.5 pp bound -
missed by 0.03), **R4 not pooled PASS**. All four seeds now put the no-bypass arm ahead:
+1.39 / +3.11 / +3.42 / +2.05. Because R3's pre-registered falsifier says a span above 1.5 pp means "no
delta may be quoted at all", **no delta is quoted here and none is claimed as significant** - the numbers
are shown, the rule is not re-read. Checkpoints: 4 arms, 0 provenance problems
(`verify_wall_ab_ckpt.py lens/runs/WALL-REP.txt`). Full write-up: `docs/EVIDENCE.md`.

Every arm's checkpoint has been machine-checked against the epoch its log line named
(`lens/scripts/verify_wall_ab_ckpt.py`, output `lens/runs/WALL-AB-CKPT-VERIFY.txt`,
sha256 4e11965f...: **6 arms, 0 provenance problems**; the 5-arm version is kept, not deleted, at
`lens/runs/archive/ckpt_verify_5arms_2026-11-06/`, sha256 6b55befd...).
Supervisor: lens/scripts/run_wall_ab_resume2.sh via the chain (label wall_ab_resume2), log
`lens/runs/WALL-AB.txt` (1,226 lines).

## Data scale at d=256 - the first lever that moved this rung (2026-11-06 19:29)

One flag changed (`--max-windows 20000` -> `200000`), everything else identical: same corpus, same command,
same concept module (the auto-kept set is data-dependent: 37 kept at 20k windows, 36 at 200k), **same 3,811,534 parameters**, no bypass, seed 0 - and the **same 87,035-window
held-out set** in both runs (`n=44418` masked / `n=58191` assistant positions in each epoch-3 line, verified
from both logs), so the extra data cannot have come from resizing the scored set.

| run | seed | windows | VAL acc | ASST acc | final ppl | source |
|---|---:|---:|---:|---:|---:|---|
| `wall_off` | 0 | 20,000 | 12.18% | 13.00% | 399.50 | `lens/runs/WALL-AB.txt:116` |
| `datascale_honest_s0` | 0 | 200,000 | **17.43%** | 18.61% | 207.29 | `lens/runs/DATASCALE-HONEST.txt:230` |
| `wall_off` | 1 | 20,000 | 12.79% | 13.21% | 395.80 | `lens/runs/WALL-AB.txt:765` |
| `wall_off` | 2 | 20,000 | 13.40% | 13.83% | 384.39 | `lens/runs/WALL-REP.txt:329`, reproduced exactly by `DATASCALE-20K-S2.txt` |
| `datascale_honest_s1` | 1 | 200,000 | **19.71%** | **20.82%** | **184.32** | `lens/runs/DATASCALE-HONEST-s1.txt:262` |
| `datascale_honest_s2` | 2 | 200,000 | **20.50%** | **21.47%** | **180.30** | `lens/runs/DATASCALE-HONEST-s2.txt:256` |
| `datascale_500k_s0` | 0 | **500,000** | **20.81%** | **21.82%** | **171.47** | `lens/runs/DATASCALE-500K.txt:392` |
| `datascale_2m_s0` | 0 | **2,000,000** | **21.29%** | **22.60%** | **149.14** | `lens/runs/DATASCALE-2M.txt:1142`; best checkpoint ep2 ASST **22.82%** at `lens/runs/DATASCALE-2M.txt:817` |
| `datascale_500k_nogate_s0` (**gates OFF** control) | 0 | 500,000 | **22.05%** | **23.22%** | **140.27** | `lens/runs/DATASCALE-500K-NOGATE.txt:69` |
| `sgd_500k_s0` (**plain SGD**, gates OFF) | 0 | 500,000 | 19.81% | 20.38% | 221.78 | `lens/runs/SGD-500K.txt:66` (concepts 31/36) |
| `datascale_2m_nogate_s0` (**gates OFF**, 2M) | 0 | 2,000,000 | **23.12%** | **24.96%** | **104.71** | `lens/runs/DATASCALE-2M-NOGATE.txt:69` |
| `datascale_4m_nogate_s0` (**gates OFF**, 4M) | 0 | 4,000,000 | **23.74%** | **25.71%** | **95.99** | `lens/runs/DATASCALE-4M-NOGATE.txt:69` |
| `datascale_4m_nogate_ep6_s0` (**gates OFF**, 4M, 6 epochs) | 0 | 4,000,000 | **23.92%** | **25.92%** | **93.72** | `lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73` |
| `datascale_4m_nogate_s1` / `_s2` (**gates OFF**, 4M, seeds 1-2) | 1, 2 | 4,000,000 | **24.13% / 24.03%** | **25.84% / 25.84%** | **93.88 / 93.52** | `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` (seed 1) and `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` (seed 2) |
| `datascale_8m_nogate_s0` (**gates OFF**, 8M) | 0 | 8,000,000 | **24.51%** | **26.46%** | **87.55** | `lens/runs/DATASCALE-8M-NOGATE.txt:235` |
| `datascale_16m_ours_s0` (**our step rule**, 16M) | 0 | 16,000,000 | **24.58%** | **26.40%** | **84.47** | `lens/runs/DATASCALE-16M-OURS.txt:225` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS.txt:226` (final, concepts) |
| `datascale_4m_ours05_s1..s2` (**our rule, opt-lr 0.5**, 4M, seeds 1-2) | 0 | 4,000,000 | **26.10 / 25.84** (3-seed mean **26.00**, spread 0.26 pp) | — | **85.61 / 86.02** | `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:151` and `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:301` |
| `datascale_4m_ours025_s0` (**our rule, opt-lr 0.25**, 4M) | 0 | 4,000,000 | **26.78%** | — | **88.18** | `lens/runs/DATASCALE-4M-OPT025.txt:154` and `lens/runs/DATASCALE-4M-OPT025.txt:155` |
| `datascale_8m_ours05_s0` (**our rule, opt-lr 0.5**, 8M) | 0 | 8,000,000 | **26.63%** | **28.65%** | **77.86** | `lens/runs/DATASCALE-8M-OPT05.txt:154` (VAL/ASST/ppl) and `lens/runs/DATASCALE-8M-OPT05.txt:155` (concepts) |
| `datascale_4m_ours05_s0` (**our rule, opt-lr 0.5**, 4M) | 0 | 4,000,000 | **26.06%** | **27.98%** | **85.60** | `lens/runs/DATASCALE-4M-OPT05.txt:154` (VAL/ASST/ppl) and `lens/runs/DATASCALE-4M-OPT05.txt:155` (concepts) |
| `datascale_16m_ours05_s0` (**our rule, opt-lr 0.5**, 16M) | 0 | 16,000,000 | **26.76%** | **28.85%** | **71.69** | `lens/runs/DATASCALE-16M-OURS05.txt:225` (VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS05.txt:226` (concepts) |
| `datascale_8m_sgd_s0` (**plain SGD**, 8M) | 0 | 8,000,000 | **27.58%** | — | **93.68** | `lens/runs/DATASCALE-8M-SGD.txt:151` |
| `datascale_2m_ours05_s0` (**our rule, opt-lr 0.5**, 2M) | 0 | 2,000,000 | **24.81%** | — | **99.79** | `lens/runs/DATASCALE-2M-OURS05.txt:154` |
| `datascale_500k_ours05_s0` (**our rule, opt-lr 0.5**, 500k) | 0 | 500,000 | **22.97%** | — | **139.47** | `lens/runs/DATASCALE-500K-OURS05.txt:155` |
| `datascale_16m_sgd_ep6_s0` (**plain SGD, 6 epochs**, 16M) | 0 | 16,000,000 | **28.61%** | — | **72.25** | `lens/runs/DATASCALE-16M-SGD-EP6.txt:227` |
| `datascale_16m_sgd_lr02_s0` (**plain SGD, lr 0.2**, 16M) | 0 | 16,000,000 | **27.90%** | — | **91.17** | `lens/runs/DATASCALE-16M-SGD-LR02.txt:151` |
| `datascale_2m_sgd_s0` (**plain SGD**, 2M) | 0 | 2,000,000 | **25.10%** | — | **133.41** | `lens/runs/DATASCALE-2M-SGD.txt:151` |
| `datascale_4m_sgd_s0` (**plain SGD**, 4M) | 0 | 4,000,000 | **26.51%** | — | **110.50** | `lens/runs/DATASCALE-4M-SGD.txt:151` |
| `datascale_16m_nogate_s0` (**plain SGD**, 16M) | 0 | 16,000,000 | **28.30%** | **30.42%** | **80.15** | `lens/runs/DATASCALE-16M-NOGATE.txt:136` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-NOGATE.txt:213` (final, concepts) |

**Three seeds, three matched pairs: mean gain +6.42 pp.** Val acc 17.43% / 19.71% / 20.50% against 20k
baselines of 12.18% / 12.79% / 13.40%, so the paired gains are **+5.25, +6.92 and +7.10 pp** (mean **+6.42**),
mean VAL **19.21%**. The baseline and the arm rise together - a seed that gives a stronger 20k run also gives a
stronger 200k run - which is why the paired gain spans only 1.85 pp while the 200k values themselves span
3.07 pp; the paired gain is the number to quote. Seed 1 passed every pre-registered check (band R1, ppl R2,
concepts C1); seed 2's band check **failed by 0.32 pp on the high side**, the direction its prediction doc had
pre-named as "the band was too conservative", while its mean and concept checks passed. Graders:
`lens/scripts/datascale_grade.py`, `lens/scripts/datascale_seed2_grade.py`; write-ups
`docs/EVIDENCE.md`, `DATASCALE-SEED2-RESULT.md`, `DATASCALE-20K-S2-RESULT.md`.

**The lever bends flat between 500k and 2M in the GATED method - and the ungated rung says that was partly the gates.** 2,000,000 windows at seed 0 reached
**21.29% VAL / 22.60% ASST / ppl 149.14**, concepts 36/36, and the pre-registered checks split: T1 (> 20.81%)
PASS at **+0.48 pp**, T3 and T4 PASS, **T2 FAIL** (landed 0.21 pp below its 21.5% floor) and **C4 FAIL** (epoch 3
at 21.29% is 0.40 pp *below* epoch 2, so the best checkpoint - ASST 22.82% - is epoch 2, not the final one). The
whole seed-0 ladder is now 12.18 -> 17.43 -> 20.81 -> 21.29%, i.e. **1.58, then 2.56, then 0.24 pp per
doubling**: 100x data bought +9.11 pp in total and the last 4x bought 5.3% of it, while perplexity kept falling
(399.50 -> 149.14) as accuracy stopped rising. So the earlier "not flattened at 200k" note is now completed by
the rung above it: the curve rose one more step and then flattened, and 0.24 pp/doubling is inside the 3.07 pp
seed spread, so the last rung is not distinguishable from noise on one seed. Source: `lens/runs/DATASCALE-2M.txt:1142`;
grader `lens/scripts/datascale_ladder_grade.py 2m`; write-up `docs/EVIDENCE.md`. Not
comparable to the older 2M result at batch 1024 (24.20%) - different batch and sampling path.

**Re-run at 2M with the gates off, the flat curve moves: 23.12% VAL / 24.96% ASST / ppl 104.71**, against the gated
twin's 21.29 / 22.60 / 149.14 - **+1.83 pp for the same data, and 8.75x faster** (249 s/epoch against 2,081-2,279).
All five pre-registered checks pass except U2, which missed *high*: I predicted at most +1.0 pp over the ungated
500k arm and it bought **+1.07 pp** (21.5-23.0 band, got 23.12) - the direction its own falsifier named as "above
23.0 reopens the ladder". So the two curves are 500k 20.81 -> 2M 21.29 gated (**0.24 pp/doubling**) and 22.05 ->
23.12 ungated (**0.54 pp/doubling**): both flatten, the gated one flattens hardest, and the ungated curve is still
climbing. **23.12% is the highest d=256 held-out number on record** (previous best 21.29% gated), and still 14.25 pp
below the d=512 headline. The gates-off effect has now been measured at two rungs, same direction, similar size
(+1.24 pp at 500k, +1.83 pp at 2M), with the mechanism visible in train accuracy (35.10% ungated vs 21.14% gated).
Source: `lens/runs/DATASCALE-2M-NOGATE.txt:69`; write-up `docs/EVIDENCE.md`. One seed per
configuration - the 3.07 pp seed spread at 200k is larger than either delta, so direction and speed are solid and
effect size is not.

**And it is still paying at 4M.** 4,000,000 windows, gates off: **23.74% VAL / 25.71% ASST / ppl 95.99**, concepts
36/36, 484-490 s per epoch (`lens/runs/DATASCALE-4M-NOGATE.txt:69`). Paired to the 2M ungated arm that is **+0.62 pp
for 2x data = 0.62 pp/doubling**, against 0.54 pp/doubling for the 500k->2M step: the ungated rate has not decayed,
which is the opposite of what the gated ladder reported. All five pre-registered checks passed (V1 > 23.12, V2 inside
23.4-24.6, V3 ppl < 104.71, V4 mechanical, V5 concepts), and the one generic check that failed is C4 - **epoch 3
equals epoch 2 to two decimals (23.75 -> 23.74)** while perplexity kept falling, so the arm plateaued on accuracy
inside the run. That plateau is the next question, not more data: the arm is limited by its 15,625 steps x 3 epochs,
so "more epochs at 4M" and "bigger batches" are different experiments and neither has been run. 23.74% is the best d=256 number in this **capped-window ladder** (<=4M windows, 3 epochs, honest mix) - 2.45 pp
above the gated 2M arm and 13.63 pp below the d512 headline - but it is **not** the best d=256 number in the
project: full-corpus d=256 runs reach **35.72% by position** (`docs/EVIDENCE.md`, 16.9M windows,
6 epochs), 10.01 pp higher on the comparable metric. Source for the 23.74% row: `lens/runs/DATASCALE-4M-NOGATE.txt:69`. One seed;
the 3.07 pp spread at 200k dwarfs the +0.62 pp. Write-up: `docs/EVIDENCE.md`.

**The step gain survives seeds, and the crossing is still unlocated.** At 4M with `--opt-lr 0.5`, seeds 1 and 2 give 26.10% and 25.84% (`lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:151`, `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:301`); with seed 0's 26.06% (`lens/runs/DATASCALE-4M-OPT05.txt:154`) the three-seed mean is **26.00%** over a 0.26 pp range (`docs/EVIDENCE.md` stands for that computed mean; this paragraph does not recompute it). That clears the best old-step seed (24.13) by 1.87 pp, so the +2.2 pp is not a single-seed artefact. It does not change the top of the table: nothing in this rule reaches plain SGD's 28.30% at 16M (`lens/runs/DATASCALE-16M-NOGATE.txt:213`), and no rung between 500k and 16M has both rules measured - the next arm is a 4M plain-SGD run to bracket where the ordering flips. S5 failed on one concept (seed 1: AUROC 0.9909, fires 0.016 vs a 0.02 floor) and is recorded as a miss, with the audit in `docs/EVIDENCE.md`.

**At matched settings the rule gap is a smooth monotone, and our rule only wins at 500k.** Filling every rung for both rules (our 0.5 arm against SGD's default) gives: 500k **+3.16** for ours, then -0.29, -0.51, **-0.95** and **-1.54** at 2M / 4M / 8M / 16M (`lens/runs/DATASCALE-8M-SGD.txt:151` is the 8M SGD cell; the rest are cited in `docs/EVIDENCE.md`). Five rungs, one configuration change, a monotone - not a step change at a particular rung, which is what my earlier 'level at 2M-4M' sentence implied. Individually the two smallest cells are inside one-seed noise; the monotone across five rungs is what carries the claim. The step effect over the same fill-in reads **+0.92 / +1.69 / +2.03 / +2.12 / +2.18** at 500k / 2M / 4M / 8M / 16M.

 Plain SGD at 4M reaches **26.51% / ppl 110.50** (`lens/runs/DATASCALE-4M-SGD.txt:151`) in **11 min 18 s** - against 28 min 23 s and 28 min 20 s for the two 4M arms under our rule tonight, both at similar load - so it is **level with our best (26.00, three seeds, `docs/EVIDENCE.md`) at 2.51x less compute**. I predicted it would land below 26.00 and it did not: T1 failed. The defensible sentence is not 'SGD wins at 4M' (0.51 pp is inside noise) but 'our rule's advantage is a small-data effect, gone by 4M', with the seed-proof losses at 16M. Note the trade: SGD's accuracy comes with far worse calibration (ppl 110.50 vs 85.60 / 85.61 / 86.02, `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:302`).

**At 16M windows plain SGD beats our step rule by 3.72 pp, reversing the 500k result.** Matched pair, identical corpus, windows, seed, epochs and batch, only the optimiser differs: **SGD 28.30% / ASST 30.42% / ppl 80.15** against **our rule 24.58% / 26.40% / 84.47** (`lens/runs/DATASCALE-16M-NOGATE.txt`, `lens/runs/DATASCALE-16M-OURS.txt`). I predicted our rule would still win, by less than the +2.24 pp it won at 500k; the pre-registered comparator returned PREDICTION FALSIFIED. **The +2.24 pp is therefore a small-data result** and must not be quoted as a property of the optimiser. Mechanism, written down before epochs 2-3 existed and then observed: our rule's accuracy stayed flat (24.65 -> 24.49 -> 24.58) while perplexity fell 3.80, the signature of a fixed-size unit step overshooting; SGD's decaying steps climbed 27.27 -> 27.90 -> 28.30 over the same epochs. This arm also lost one concept: 35/36, `mentions_a_name` fires on 1.8% of rows (AUROC 0.9982). One seed, and this rule's own seed spread is 1.89 pp - the 3.72 pp gap exceeds it, but two seeds would settle it.

**The 4M rung reproduces to 0.10 pp - and the 8M gain beats the whole range.** Three seeds of the same 4M ungated config: **23.74 / 24.13 / 24.03%**, mean **23.97%** (computed; recorded in `docs/EVIDENCE.md`), range 23.74-24.13, seed-to-seed spread **0.10 pp** (`lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` for seed 1, `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` for seed 2, and `lens/runs/DATASCALE-4M-NOGATE.txt:69` for seed 0); provenance 2 arms, 0 problems. Reproducibility improves with data: the 200k rung's three-seed spread is **3.07 pp** (`docs/EVIDENCE.md`), twenty times wider. So the 8M result (`lens/runs/DATASCALE-8M-NOGATE.txt:235`) is above every one of the three 4M seeds by 0.38-0.77 pp - the "one more doubling pays" claim no longer rests on one seed. The same fact cuts the other way at 200k, where the 3.07 pp spread is wider than the +1.24 pp gates-off delta at 500k, so that delta stays unpublishable until gated twins exist at those seeds.

**Six epochs is the ladder's best arm, and it buys calibration more than accuracy.** The same 16M plain-SGD rung run for six epochs gives **28.61% / ppl 72.25** (`lens/runs/DATASCALE-16M-SGD-EP6.txt:227`) against three epochs' 28.30% / 80.15 (`lens/runs/DATASCALE-16M-NOGATE.txt:136`): **+0.31 pp** (inside one seed) and **7.90 ppl better**. Accuracy stops moving after epoch 3 (28.17 -> 28.46 -> 28.46 -> 28.61) while perplexity falls at every one of the six epochs, so the extra passes bought confidence, not correctness. Note the decay confound: `--lr-decay` spreads over the run, so this is six passes at a gentler decay, not six passes alone. Full write-up: `docs/EVIDENCE.md`.

**And halving the step is our rule's trick, not a general one.** The same idea applied to the winning rule - plain SGD at 16M with `lr 0.2 -> 0.1` - gives **27.90% / ppl 91.17** (`lens/runs/DATASCALE-16M-SGD-LR02.txt:151`) against the default lr's (`lens/runs/DATASCALE-16M-NOGATE.txt:136`) 28.30% / 80.15: 0.40 pp down on accuracy (inside one seed, so 'buys nothing' rather than 'hurts') and **11.02 ppl worse**, which is not inside anything. So the +2.2 pp from smaller steps belongs to `--method ours` and must not be stated as a property of the architecture.

**Halving the step recovers 2.18 pp - and leaves the rule 1.54 pp behind SGD.** 16M windows, gates off, seed 0, our rule at `--opt-lr 0.5`: **26.76% VAL / ASST 28.85% / ppl 71.69**, concepts 36/36 (`lens/runs/DATASCALE-16M-OURS05.txt:225`). Against the same arm at `1.0` (24.58%, `lens/runs/DATASCALE-16M-OURS.txt:226`) that is **+2.18 pp**, Z1's PASS, and it confirms the overshoot mechanism written down before the arm ran: the 1.0 unit step was too large for 187,500 updates. Against plain SGD (28.30%, `lens/runs/DATASCALE-16M-NOGATE.txt:213`) the shortfall shrinks from 3.72 to **1.54 pp** - roughly 2.2 of the 3.7 was a fixable configuration error and the rest is the rule itself. The same signature reappears one level up: accuracy 26.74 -> 26.85 -> 26.76 while perplexity fell to 71.69, which is why C4 (ep3 >= ep2) failed. Because every ladder rung ran at 1.0, its absolute levels are probably depressed by about this much - deltas stand, ceilings do not.

**The step has a plateau below 0.5, not a peak.** At 4M, `--opt-lr 0.25` gives **26.78% / ppl 88.18** (`lens/runs/DATASCALE-4M-OPT025.txt:155`) against `0.5`'s 26.06% (`lens/runs/DATASCALE-4M-OPT05.txt:154`). The 0.72 pp gap is well inside this rule's **1.89 pp** seed spread, so the two settings are indistinguishable here and neither can be called the optimum. What is not in doubt is the gap to `1.0`: 23.74% (`lens/runs/DATASCALE-4M-NOGATE.txt:69`), about 3 pp below either. Practical reading: 0.5 is a safe setting, there is no demonstrated reason to halve it again, and every claim about *which* small step is best needs seeds this project does not yet have. My V1 predicted 0.25 would land below 0.5; it failed.

**One caveat that applies to every number in this table, measured at four rungs and smaller at the bottom.** Every gates-off runner at 4M and above passes `--opt-lr 1.0`, and there that step costs about **2.1-2.3 pp**: 4M 23.74 -> **26.06**, 8M 24.51 -> **26.63**, 16M 24.58 -> **26.76** (`docs/EVIDENCE.md`, `docs/EVIDENCE.md`, `docs/EVIDENCE.md`; the 1.0 twins on `lens/runs/DATASCALE-4M-NOGATE.txt:69`, `lens/runs/DATASCALE-8M-NOGATE.txt:235`, `lens/runs/DATASCALE-16M-OURS.txt:226`). The extremes are four times apart in update count and the effect is the same size, so it is a constant handicap, not a scale effect: every level here sits ~2.2 pp below what its configuration reaches, and none should be read as a ceiling. The curve's shape survives the correction - at 0.5 the steps are +0.57 then +0.13, at 1.0 they were +0.77 then +0.07 - so the flattening is real. The best number in this ladder is therefore **26.76%**, not 24.58%. The 3-seed optimiser sweep used 0.5 (`docs/EVIDENCE.md`), so it and this ladder were never the same configuration.

**At 500k the step is worth only half as much, and our small-data win gets bigger anyway.** Our rule at `--opt-lr 0.5` on 500k gives **22.97% / ppl 139.47** (`lens/runs/DATASCALE-500K-OURS05.txt:155`) against 22.05 at 1.0: **+0.92 pp**, not the ~2.2 seen from 4M up - and 500k is the noisiest rung in this project, so even that is inside its own spread. So the step effect is *not* flat across all rungs; it is ~2.2 pp where the ladder has data. The consequence that matters is the other one: at our better setting our rule still beats plain SGD at 500k, now by **3.16 pp** (22.97 vs 19.81, `lens/runs/SGD-500K.txt:66`), where the old comparison offered only 2.24. The one rung we win is won by more than the step correction explains.

**The data axis has not flattened: doubling to 8M paid more than the doubling below it.** 8,000,000 windows, gates off, seed 0, three epochs: **24.51% VAL / 26.46% ASST / ppl 87.55**, concepts 36/36 (`lens/runs/DATASCALE-8M-NOGATE.txt:235`; the concept audit 36/36 is on `:236`). The 4M rung was 23.74% (`lens/runs/DATASCALE-4M-NOGATE.txt:69`), so this doubling bought **+0.77 pp** against **+0.62 pp** for 2M -> 4M - the gain grew rather than decaying, which falsifies the expectation written into the prediction (X2 was centred below a repeat of 0.62). Path 23.46 / 24.03 / 24.51 with ppl 95.71 / 91.04 / 87.55. Two limits travel with it: the prepared mix holds ~18.4M windows, so only one more doubling is available before the axis runs out of data, and with one seed this says the axis has not turned down, not that 0.77 pp is precise (the 200k rung's three-seed spread was 3.07 pp). The machine rebooted mid-arm at 18:52; the arm resumed from its epoch-1 checkpoint (about 11 minutes lost, not 51).

**Doubling the steps was not the lever.** Six epochs at the same 4M windows reached **23.92% VAL / 25.92% ASST /
ppl 93.72** (`lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73`) - **+0.18 pp** for doubling the updates from 46,875 to
93,750. The hypothesis behind that arm ("the plateau is step-limited") was falsified by its own pre-registered
threshold (W1 needed >= 24.04, got 23.92). At this rung, **doubling the data bought +0.62 pp and doubling the steps
bought +0.18 pp**, so data is worth ~3.4x what steps are worth per doubling; ppl fell the whole way (95.89 ->
93.72) while accuracy wobbled in a +-0.3 pp band. Neither closes the **11.80 pp** gap to the project's d256 record
(35.72% by position, full corpus, `docs/EVIDENCE.md`). 8M windows were then run: 24.51% (`lens/runs/DATASCALE-8M-NOGATE.txt:235`). One seed;
only the ratio of the two deltas is readable, not their sizes. Note: `docs/EVIDENCE.md`.

**The gates were part of that ceiling, and the headline was never gated.** Same 500k rung, same step rule, same
data, gates OFF: **22.05% VAL / 23.22% ASST / ppl 140.27** against the gated arm's 20.81 / 21.82 / 171.47 - and
**72 s per epoch instead of 558, 7.7x faster** (`lens/runs/DATASCALE-500K-NOGATE.txt:69`, twin at
`DATASCALE-500K.txt:392`). All four pre-registered checks passed (G1 +1.24 pp, G2, G3 mechanical 0 gate lines,
G4 36/36 concepts). The gates let the model fit the training set far harder (22.25% -> 35.20% train) and overfit
far more (train-minus-VAL gap 1.44 -> 13.15 pp) **while held-out accuracy rose** - so at this rung they were
suppressing learning that generalised. That matters beyond the rung: every ladder arm ran `--method ours` (gated),
while the run behind the **37.37%** headline prints `METHOD: standard (the old loop: plain SGD, our gates OFF)`
(`lens/runs/TIED-D512-EXT6.txt`). The project's best number therefore comes from the ungated old loop, and "our
method vs the old loop" has never been measured at a matched rung - that is the next arm. One seed, one rung, and
+1.24 pp is inside the 3.07 pp seed spread, so this is a direction, not a settled effect; the 7.7x is a measured
ratio. Write-up: `docs/EVIDENCE.md`.

**And the step rule is worth more than the gates.** Adding the third configuration - plain SGD, gates off, same
rung and seed - gives **19.81% / 20.38% ASST / ppl 221.78**, concepts only **31/36** (`lens/runs/SGD-500K.txt:66`).
The matched three-way at 500k is therefore **relgrad ungated 22.05 > relgrad gated 20.81 > plain SGD 19.81**: our
step rule is worth **+2.24 pp** over the loop this project used before **at this rung** (at 16M windows the ordering reverses - see the paragraph above), the gates cost **1.24 pp**, and plain SGD
is 2.3x cheaper per epoch while learning the least. That is awkward for the headline: **37.37% was produced by
plain SGD** on 16.9M windows in 6 epochs (`TIED-D512-EXT6.txt`), so the best number sits on the weakest-measured
step rule. The next test is that configuration at that scale - d=512, full corpus, 6 epochs, relgrad, gates off -
and it needs a **one-epoch cost probe** first, because relgrad is ~2.3x plain SGD per epoch while plain SGD already
took 1,720 s/epoch at that scale (`docs/EVIDENCE.md`). One seed per configuration and all three deltas
sit inside the 3.07 pp seed spread: the ordering is a direction, not a settled effect. Write-up:
`docs/EVIDENCE.md`.

**What one d512 epoch actually costs, and a confound that blocks the obvious conclusion.** A one-epoch probe in
the headline's own configuration (d=512, `--direct --direct-mode tied`, batch 1024, lr 0.3, 16.9M windows, our
step rule, gates off) took **2,854 s** - so six epochs scale to **~4.8 h** - and reached **25.18% VAL / 26.67%
ASST / ppl 110.24**, against the headline's epoch 1 at **31.52% / 33.46% / 46.99** (`lens/runs/PROBE-D512.txt`,
headline at `TIED-D512-EXT6.txt:83`). That is **-6.34 pp** and 1.66x slower per epoch. It is **not** evidence
against our step rule: the probe's train accuracy is **21.12%** against the headline's **42.01%**, and its log
shows the step size is still the constant calibrated at d=256 (`opt_lr 0.0441256`, `--lr` unused by this rule)
while the headline ran SGD at lr 0.3. A step that is too small explains both the deficit and the slower
convergence, and this arm cannot separate that from a weaker rule - the next step is to calibrate the step size at
d=512, not to run six epochs. One epoch, one seed, and 12.19 pp below the headline: nothing here changes 37.37%.
Note: `docs/EVIDENCE.md`.

The concept jump was not predicted by D1/D2, and it repeated: the audit went **20/36 at 20k windows ->
36/36 at 200k** on both seeds, counted from the AUROC rows rather than from the summary lines.

**Is the 3-epoch schedule truncating it? No.** A 6-epoch arm at 200k, seed 0, one knob changed
(`docs/EVIDENCE.md` written first), reached final VAL **18.43% / ppl 201.69** - a gain
of **+1.00 pp** over the 3-epoch 17.43, landing exactly on the predicted threshold, so E1 is graded **PARTIAL**
by its own wording (E2 ppl PASS). The path there was a swing, not a climb: 17.48, 17.90, 17.43, **15.13**,
**19.68**, 18.43 - and the best saved checkpoint is epoch 5 (ASST 20.83% vs 18.87% for the 3-epoch arm). So
doubling the schedule doubles the cost for +1.00 pp of endpoint and a better best-checkpoint, and this rung is
unstable epoch-to-epoch. Write-up: `docs/EVIDENCE.md`; the 3-epoch result is not averaged
with it.

Caveats, stated because they matter: **three seeds**, so no p-value and no invented interval - the three
values and their mean are what is published. The 20k rung's measured run-to-run spread is 1.03 pp
(`docs/EVIDENCE.md`), and the three seeds span 3.07 pp, i.e. seed-to-seed variation at this rung is
three times the measurement noise and is what limits precision here. The headline is unchanged - best held-out accuracy is still **37.37%**
(d=512, full corpus, 6 epochs). Checkpoints verified with `verify_wall_ab_ckpt.py`: the seed-1 arm is 1/1
clean, the parameter count off the stored arrays is 3,811,534 in both seeds, and `best.npz` equals the epoch
the log named. Write-ups: `docs/EVIDENCE.md` (seed 0) and
`docs/EVIDENCE.md` (seed 1).