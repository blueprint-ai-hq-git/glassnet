# The chat model is 83% readable, not 100% - and the audit names the cause with a number (2026-11-18)

"100% readability" is a standing requirement. It is not met, and this is the measurement that says so rather than an impression. `tools/concept_label_audit.py`, 800 matched
validation windows, the same audit gates the trainers print:

```
readable d512   30/36 concepts pass   mean AUROC 0.6886   mean FIRE RATE 47.4%
copy-path d512  36/36 concepts pass   mean AUROC 0.8695   mean FIRE RATE 28.3%

failing in the readable model only: is_a_list, mentions_a_name, mentions_a_quantity,
                                    mentions_body, mentions_family, mentions_sport
failing in BOTH models: none        failing in the copy-path model only: none
```

## Two facts that matter more than the pass count

**1. Every failure is the readable model's alone.** No concept fails in both, and none fails only in the copy-path model. So the six failures are **not** the concept formulas being
wrong for this corpus - a matched model with a direct path passes all 36 with the same formulas. The failure is architectural.

**2. The readable units fire far too often.** The concept definitions have supports from **0.58%** (`mentions_weather`) to **13.9%** (`mentions_a_quantity`), and the readable model's
neurons average a **47.4%** firing rate - **5 to 20 times** more often than the property they are named after. A unit that fires on nearly half of all windows cannot be a sharp
detector of a 0.6% property, which is exactly why its AUROC sits at 0.6-0.7 while the name is in its own supervision. The copy-path model's units sit at 28.3% and score 0.87 mean
AUROC: still above their supports, but far closer.

This is the same defect class found in the image model this morning (`centre_heavier`, AUROC 0.3196 because its formula fired on 96.9% of images), now measured on the chat side with a
real spread of supports to compare against.

## The dial that has never been used

`train_chat.py` has `--lam-act`, a penalty on mean concept activation, and it has been **0.0 in every run this project has done** - it is in the flag list and in the checkpoint
metadata of both models above. Nobody has ever turned it on.

`lens/scripts/run_concept_act.sh` runs the readable d512 recipe twice with `--lam-act 0.3` and `1.0`, everything else copied verbatim, and re-audits each arm:

- **P1** mean fire rate falls from 47.4% to **below 25%** at lam_act 1.0
- **P2** mean AUROC rises from 0.6886 to **above 0.80**
- **P3** concepts passing rises from 30/36 to **at least 33/36**
- **FALSIFIER** mean AUROC stays **below 0.72** at lam_act 1.0: then firing rate is not the cause, the failure is in what these formulas can be computed from over a flattened
  32-token window, and the next lever is the concept **definitions** rather than any penalty.

## Why this is the right next lever, in one line

The failure is architectural rather than definitional (measured: same formulas pass in the copy-path model), and the architecture's symptom is measurable (units firing 5-20x too
often), and there is a dial for that symptom that has never been tried.
