

---

# AT THE QUALIFYING BUDGET THE READABLE WINDOW BUYS NOTHING: +0.06 AND +0.32 AGAINST A +0.5 BAR (2026-11-19 16:38)

This is the test that was supposed to be the fair one. Both arms at **6000 steps** - the budget established this afternoon as the lowest where a model clears the
trivial constant - on both instruments, with the falsifier written before the run: *"both arms beat the constant AND the window is not >= 0.5 above its control on
BOTH instruments -> the readable window buys nothing when the model is capable, and the memory line is dead on measurement."*

```
                              trainer line          independent (matched protocol)
control (K=0)    params 1,684,224    22.82% (loss 5.3159)     20.30%   -> BEATS constant
window  (K=8)    params 1,848,320    22.88% (loss 5.3002)     20.62%   -> BEATS constant
paired delta                          +0.06                    +0.32

trivial constant on the same draw:    14.70% all positions | 13.14% <unk>-excluded
window at the end of training:  ` | class | for | a | library | management | ` | book   (8 distinct real words)
```

- **P1 MET:** both arms beat the matched constant - by **7.16 and 7.48 points**. These are the first properly qualified arms in this project's history, and
  that part is a real gain over everything before them.
- **P2 FAILED:** +0.06 against a +0.5 bar.
- **P3 MET:** the independent re-score gives the same sign (+0.32), so this is not an instrument artefact - both instruments say "a little positive".
- **P4 MET:** the window still holds eight distinct real tokens.
- **THE FALSIFIER FIRED.** Both arms clear the constant, and the window is not >= 0.5 above its control on either instrument.

## What this settles

**The readable window bank, as it was built, buys nothing measurable on the language objective** - +0.06 on the trainer's line and +0.32 on the independent one,
both under a bar of 0.5, at the only budget where the model is capable enough for the question to be meaningful. This is consistent with every earlier memory
result today and it is the cleanest one: the same answer from two instruments at a qualified budget.

It does **not** say the window is useless. It says the window is **neutral for next-token prediction in a model that already has a 2-token effective context** -
which is exactly what should be expected of a recent-token buffer when the objective rarely needs more than the immediately preceding token.

## What is still open, and it is one thing

**The age-key arm (`glassnet-agekey6k`, in flight)** is the same bank with the addressing fix, paired against this very control. The delayed-recall probe says the
key is worth **+65.01 points at delay 8** (three seeds, per-seed +62.1 to +69.5); this run says the plain bank is worth **+0.06** on real text. If the key moves
the language objective too, addressing was the whole story. If it does not, then real text at this budget simply does not demand positional recall, and the
honest next move is FLAW 4 (the 5.34% `<unk>` positions) rather than more memory architecture.

That is the falsifier written for the age-key arm, unchanged: **no gain on both instruments -> the key fixes synthetic recall and nothing else.**
