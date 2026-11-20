

---

# TWO CLOSURES: THE MEMORY BANK IS WORTH +0.49, AND AT MATCHED PARAMETERS THE READABLE MODEL IS 1.53 POINTS WORSE (2026-11-20 14:26)

Both are pre-registered verdicts, not judgements made after seeing the numbers.

## 1. Memory bank: the decisive test, at the task that requires memory

`off2pair` moved the supervised target to **offset 2**, so the preceding token can no longer supply the answer - the one condition in
which a memory window has something to sell. Two arms differing only in `K` (8 = readable bank, 0 = no bank), same seed, batch order,
6000 steps, same draw of 104 windows.

```
                          held-out acc   vs constant 14.47%
bank    K=8               16.57%         +2.10  BEATS
no bank K=0               16.08%         +1.60  BEATS
paired delta              +0.49
```

- **P1 (bar: at least +1.0) NOT MET:** +0.49.
- **FALSIFIER (bank at most +0.5 above no-bank) FIRED:** 0.49 <= 0.5, by its own condition.
- **P2 (both arms beat their matched constant) PASSED:** +2.10 and +1.60, so both arms genuinely trained and the comparison is valid.

By the rule written before the run: *"the readable window bank is worth nothing even when the task requires memory, and the memory
line closes permanently rather than only at offset 1."* **The memory line is closed permanently.** The bank provably retains content in
isolation (delayed-recall probe 81.25% at delay 16 against 3.22% control); it simply does not convert that into next-token accuracy at
any offset tested. I am not re-opening it with another seed or another weight - that is the "redefine after results" move this project bans.

## 2. Custom versus regular, at matched parameters: the readable model LOSES

The 3-seed matched comparison (`run_matched_gap_3seed.sh`) had already run; its summary line was **stale and wrong** - it printed
"NO RESULTS - every arm failed" because a pre-2026-11-18 version of the glob matched nothing. The three arms all passed. Recomputing
from the per-seed JSONs, and **deduplicating** (the `recovered/` copies are byte-identical, sha256 confirmed, so 3 seeds not 6):

```
seed   readable    plain     gap      params: readable 16,665 | plain 16,705  (0.24% apart = matched)
 0     84.36%     86.42%   -2.06      honesty 13/13 = 100.0%
 1     84.27%     85.81%   -1.54      honesty 12/13 =  92.3%
 2     83.69%     84.67%   -0.98      honesty 13/13 = 100.0%
mean: -1.53 points, sd 0.44
```

**P1 HOLDS** (mean gap between -1.0 and -2.6), **P2 HOLDS** (above -3.0, i.e. the old 4.96 figure was indeed a size artefact), **P3
HOLDS** (honesty mean 97.4%, well above 85%). Falsifier (mean <= -3.0) not triggered.

So on the image task: **at matched parameter counts over 3 seeds, the readable architecture is 1.53 points WORSE than a plain network,
while being 97.4% honest.** Honest is not the same thing as better. The correct sentence is "we can read it, and it costs about 1.5
points", not "it is 10x better".

## What this means for the ask

- **"10x better than a normal model"**: measured, it is -15% relative on the only matched task available. Not claimed, not spun.
- **Chat**: no plain matched control exists (the model refuses an empty concept layer by design - verified), so no custom-vs-regular
  claim can be made there at all. That build is queued.
- **What the project does have that is real:** a from-scratch engine, a verified-autograd, a readable concept/rule model whose concepts
  are 97.4% label-honest, an `explain()` whose named rules are causally load-bearing (346.7% vs 10.8% for random rules), certificates
  for every gradient, and a documented list of eleven falsified hypotheses with their numbers.

The stale summary line is superseded in place in `lens/runs/MATCHED-GAP-3SEED.txt`; nothing was deleted.
