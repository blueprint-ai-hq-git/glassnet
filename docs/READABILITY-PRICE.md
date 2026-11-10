# The price of making the readable concepts decide: 19.17 pp (2026-11-10)

The readable path explains only 7.3% of this model's choices (`READABLE-PATH-EXPLAINS-7PCT.md`), so the design
question is: what does it cost to give the 36 named concepts the decision? Measured on 600 matched validation
windows, same harness for every arm (`tools/concept_only_eval.py`):

```
full (concept path + copy path)   22.17%
copy path only                    20.17%
concept path only                  3.00%
concept path agrees with full       8.2%
PRICE OF MAKING THE CONCEPTS DECIDE: 19.17 pp
```

The concept path alone lands at **3.00%**, which is *below* the 4.23% frequency baseline for this corpus. Read that
carefully: a person reading the 36 named numbers and taking the model's answer from them would score worse than
always answering `,`.

## What this settles
The current architecture's ability is, numerically, the copy path. Making the readable layer the decider costs
almost all of it. So:

1. **`explain()` must report the split**, because the honest statement today is "these concepts fired, and the
   answer came from the echo path".
2. **Readability at the answer level is a training problem, not a wording problem.** The concept layer tracks its
   labels (35/36 AUROC) and cannot decide, so the fix is to train it to carry the decision - which will cost
   accuracy - rather than to describe the current model more cleverly.
3. **The 22.17% here is not the 25.52% arm result.** The trainer counts assistant-only target positions with a
   different window convention; `full` in this file is the reference the other two arms are compared against.

## A bug this file caught, and the guard that now stops it
The first run printed `full 3.00% / cost 0.50 pp` from the **wrong corpus**: `chat_data.build()` defaults to
`data/chat/prepared` (13,318,447 tokens, 86,360 conversations) while the models and the matched split live in
`prepared_mix_ctrl_honest` (294,225,090 tokens, 740,665 conversations), and a 740,665-entry split indexed silently
into an 86,360-conversation corpus. The tool now sets the corpus explicitly and asserts both the split length and
the vocabulary size, so a mismatched corpus fails loudly instead of printing a plausible number. Same class of bug
as the `--split-file` guard, in a different file - which means the guard belongs in every tool that touches a
split, not only in the trainer.
