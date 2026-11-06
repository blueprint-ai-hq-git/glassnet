# GLOSSARY - every term we use, in plain words (written 2026-10-18)

Kept short on purpose: one line each. If a term is not here and you need it, ask and it gets added.

## The model itself
- **GlassNet / lens** - the project name, and the name of the code package inside it.
- **From scratch** - we wrote the learning engine ourselves. No ready-made AI parts, no pretrained model.
- **Token** - one word or punctuation mark, the smallest piece the model reads or writes.
- **Window** - the 32 tokens the model looks at to guess the next one. (We also tried 64.)
- **Vocabulary** - the fixed list of 12,000 words the model knows.
- **d256 / d512** - how wide the model is (its size). Bigger number = bigger model.
- **Concept wall** - the middle layer where every unit has a human name, like "is_an_apology".
- **Concept** - one named unit of that layer. Each one is supposed to mean exactly its name.
- **Rule layer** - the layer above it: each rule is allowed to look at only 2-3 concepts, so a human can read it.
- **Sparse** - "uses very few connections", so a person can actually read the rules.
- **explain()** - the model printing its own arithmetic: which concepts fired and which rules produced the answer.
- **Attention** - letting the model use word order instead of just which words are present.
- **lam-concept** - the dial that decides how hard we force the middle units to match their names.

## The data
- **Corpus** - a big collection of text we train on.
- **Merged / splitfix / curated / prepared** - our four text collections. Merged is the widest and best.
- **Repaired corpus (name_honest)** - a rebuilt copy where no test passage can appear in training.
- **Train / val (test) split** - text used for learning, and text held back to test on.
- **Held out** - "kept out of training", so a score on it means something.
- **Leak / contamination / memorisation** - when test text was actually inside the training text, so the score is fake.
- **Overlap** - how much test text also appears in training. We measure it; it must be 0.

## Measuring
- **Raw number** - the old-style score, counted even on text the model had already seen (flattering).
- **Honest number** - the score after removing every test passage that appears in training (the real one).
- **Perplexity (ppl)** - how surprised the model is. Lower is better. It can look great while accuracy stays bad.
- **Masked accuracy** - we only score the AI's own words, ignoring the human's words.
- **Chance / baseline** - what you would get by guessing. A score at chance means zero skill.
- **Seed** - the random starting point. Different seeds = different runs; we need 3 to trust a small win.
- **Error bars** - the spread across seeds, so we can say "this is real" instead of "this looked better once".
- **Gate** - a pass/fail test set before work. Pass or fail, never "passed in spirit".
- **Control** - a check that must give a known answer, proving the meter itself works.

## Our meters (the tools that measure)
- **Talk-quality meter** - scores text as "real language" or "word salad" using real text and shuffled words as the two ends.
- **Salad bar** - the midpoint between real text and shuffled words. Below it = word salad.
- **Leak meter** - tests whether a named concept secretly detects something else.
- **Permuted-activation control** - a fake-signal check: it must read 0.50 (pure chance) or the leak meter is broken.
- **Stop control** - tests whether the model ever hands the turn back or ends a sentence on its own.
- **Benchmark ruler** - outside tests: MMLU (general knowledge), GSM8K (school maths), code tests.
- **pass@1** - does the code actually run and pass the test, not just look like code.

## Running experiments
- **Arm** - one experiment with one setting changed.
- **Ship** - a finished, saved model file that we can use and measure.
- **Epoch (round)** - one full pass over the training text.
- **Batch / learning rate** - how many examples at a time, and how big each learning step is.
- **Checkpoint** - the model saved mid-training, so a crash or a stop costs almost nothing.
- **Resume / skip** - continue a stopped arm from its last checkpoint; never re-run a finished one.
- **VRAM / watts (W)** - how much of the graphics card is used, and how much power the machine is drawing.

## Checks and safety
- **selfcheck** - one command that tests every area of the product against its own gates.
- **product check** - one command for a normal person: runs the checks, redraws the pictures, prints a verdict.
- **Label honesty** - our rule: every claim we make about our own output must be checked and true.
- **Enforced drop** - if a named unit fails its own name, it is removed, never kept with a disclaimer.
- **Watchdog / digest** - the hourly message that reports only when something finished, stalled, or went quiet.
- **BUGS.md / LOG.md / BENCH.md** - the bug list, the running log, and the results ledger.

## The outside comparison
- **Prior art** - published work that already did something similar.
- **CB-LLM (ICLR 2025)** - the closest published match to our design. Ours is a recombination, not a new invention.
- **The one open piece** - auditing named units AND forcing a drop when they fail. Two literature searches found nobody publishing that.
