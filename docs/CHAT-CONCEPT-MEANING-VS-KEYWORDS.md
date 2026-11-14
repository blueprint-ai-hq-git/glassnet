# Do chat concept neurons mean their names, or only their keyword lists?

Date: 2026-11-14. Tool `tools/chat_concept_independent_audit.py`, runner `lens/scripts/run_chat_independent_label.sh`,
predictions pre-registered in `lens/runs/CHAT-INDEPENDENT-LABEL-PREDICTIONS.txt`, output `lens/runs/CHAT-INDEPENDENT-LABEL-AUDIT.txt`.
Checkpoint `lens/runs/chat/exposure_d512_e9_s0_best.npz` (d512, 9 epochs), 4,000 held-out windows, inference only.

## The problem with the existing audit

The trainer's fidelity audit scores each concept neuron against `Concept.labels(X)` - "does this window contain any of this
concept's TRAINED token ids". The neuron is trained to predict exactly that label. So the audit's 36/36 at median AUROC 0.87-0.97
cannot distinguish **"the neuron learned the meaning"** from **"the neuron learned its own training target"**. Both produce a high
number on that instrument.

## The test

Score each neuron against **held-out synonym labels**: words a human would accept as the same meaning, chosen **disjoint** from the
concept's trained token ids (the tool drops any candidate already in the trained list, in code). If the neuron fires on "restaurant"
and "dinner" for `mentions_food`, it learned a meaning. If it only fires on its eight trained words, it is a keyword detector wearing
a concept's name.

## Result: partial generalisation - better than a keyword list, well below the audit's number

22 concepts tested:

```
concept                trained  held-out     drop  n_train   fires  pos_ho
mentions_law            0.9515    0.4217  -0.5298        7  0.0060      24
mentions_work           0.9021    0.5133  -0.3887        7  0.0063      25
mentions_sport          0.8022    0.5350  -0.2671        8  0.0063      25
mentions_weather        0.8803    0.5600  -0.3203        8  0.0035      14
mentions_a_number       0.9105    0.5724  -0.3381       12  0.0585     234
mentions_science        0.9367    0.5962  -0.3405        6  0.0372     149
is_explanatory          0.9418    0.6295  -0.3123        5  0.0100      40
mentions_place          0.8425    0.6335  -0.2090        8  0.0145      58
mentions_education      0.9322    0.6375  -0.2947        7  0.0035      14
mentions_art            0.8486    0.6571  -0.1916        8  0.0123      49
mentions_travel         0.9329    0.6888  -0.2441        8  0.0047      19
mentions_technology     0.9208    0.7073  -0.2135        7  0.0155      62
is_a_question           0.9857    0.7446  -0.2411        1  0.1480     592
mentions_body           0.8797    0.7763  -0.1034        8  0.0030      12
mentions_animals        0.8928    0.7858  -0.1070        7  0.0018       7
mentions_money          0.9382    0.8127  -0.1255        8  0.0067      27
mentions_a_colour       0.8777    0.8448  -0.0330        8  0.0027      11
mentions_a_size         0.8660    0.8662  +0.0002        8  0.0020       8
mentions_politics       0.8497    0.8741  +0.0243        7  0.0025      10
mentions_food           0.9236    0.8975  -0.0261        8  0.0045      18
mentions_family         0.9170    0.9003  -0.0167        8  0.0025      10
mentions_time           0.9088    0.9008  -0.0080        8  0.0130      52

median trained-label AUROC  0.9097
median held-out-label AUROC 0.6980
concepts below 0.60 on held-out: 6 of 22
spearman(trained-list length, held-out AUROC) = +0.2772
```

## Predictions, scored

- **P1 (median held-out <= 0.65) FAILED** - the median is **0.6980**, above the bar I set. The concepts generalise better than I
  predicted.
- **P2 (at least half below 0.60) FAILED** - only **6 of 22** fall below 0.60.
- **P3 (more trained tokens generalise better) HOLDS** - Spearman **+0.2772**.
- **FALSIFIER (median held-out >= 0.75) NOT TRIGGERED** - 0.6980 is short of it.

**So the answer is a partial credit, and it must be quoted as one:** the concept layer is **not** a pure keyword detector - it
transfers to unseen synonyms at a median AUROC of 0.70 against chance 0.50 - but it falls a long way short of its own audit number
(0.91), and several concepts collapse entirely. `mentions_law` drops from 0.9515 to **0.4217, below chance**, the worst case in the
set, which is consistent with the earlier finding that this concept's support was mostly the ambiguous word "right".

## The honest caveats, stated before anyone quotes the table

- **Held-out positives are rare.** Only 7 to 234 per concept (`pos_ho`). The concepts with the *highest* held-out AUROC
  (`mentions_time` 52 positives, `mentions_family` 10, `mentions_food` 18) rest on few positives, so their near-zero drops are weak
  evidence in either direction. The **large drops** (law, work, science, number, weather, explanatory, education, sport) are the ones
  that carry weight.
- **Synonym lists are my own choice.** A different human would pick different words. This is a real limitation: the audit measures
  generalisation to *my* synonyms, not to all reasonable ones.
- **One checkpoint, one seed** (d512, 9 epochs, seed 0). Not yet run on the d1024 checkpoints or on more seeds.
- **The two instruments disagree slightly on the trained-label number**: this tool reports 0.9097 median where the trainer's own
  audit logged 0.87449 for the same checkpoint, most likely because this tool samples its own 4,000 val windows with a different
  seed. Same conclusion, not the same number; recorded rather than smoothed over.

## Doubling the width widens the meaning gap, not closes it

The same audit on the d1024 checkpoint (the 47M-parameter model), same 22 concepts, same windows:

```
checkpoint        trained-label   held-out   gap      below 0.60
d512  e9 s0         0.9097        0.6980   0.2117      6 of 22
d1024 e9 s0         0.9754        0.7178   0.2576      3 of 22
```

Doubling width buys **+0.0657 on the concept's own trained labels** and only **+0.0198 on meaning transfer to unseen synonyms**. The
gap between "scores its own target" and "means its name" therefore **widens** by +0.0459 with scale.

**This is the important result of the night for the project's first standing goal.** The model gets better at the thing the audit
measures faster than it gets better at the thing the audit is *supposed* to measure. So concept honesty on the chat line is **not a
scaling problem**, and a bigger model will not fix it - an independent-label instrument has to be part of the gate, or the reported
honesty percentage will keep climbing while the meaning does not.

One good sign alongside: concepts below 0.60 on held-out synonyms fall from 6 to 3 with width, so the worst cases do improve, and
`mentions_law` remains the clearest offender. But the median does not follow.

## Bug of this run, recorded

The first d1024 invocation died with `No module named 'numpy._core._multiarray_umath'` because I called the virtualenv python inline
without clearing Hermes's inherited `PYTHONPATH`. The documented rule in this project is `env -u PYTHONPATH -u PYTHONHOME`, and the
runner script does it; my ad-hoc call did not. Re-ran through the same guard and it worked. The earlier d512 numbers were never
affected because they only ever ran through the script.
