

---

# A GUARD AGAINST DEAD FLAGS - BUILDING IT TOOK TWO BUGS IN THE GUARD ITSELF (2026-11-20 10:32)

Three separate incidents today had one shape: a launcher passes a setting and the code never reads it.

```
RESUME=     silently ignored by the trainer  -> a "continuation" restarted from step 0
DRAW_SEED=  silently ignored by midband_split -> a "re-draw" measured the same windows; identical digits were the only clue
my own patch matched `rng` while the variable is `rr`, applied nothing, and reported success
```

So `tools/check_env_wiring.py`: it takes a launcher, finds every UPPERCASE variable in the **env prefix of a python invocation**, and checks the target
file (and one or two hops of project-local imports) actually mentions the name. Anything set but never read is reported as DEAD, and the exit code is
non-zero so a launcher can gate on it.

## It took two bugs of its own, both caught by controls rather than by reading it

1. **First version flagged shell bookkeeping** - `RC=$?`, `CK=...`, `FAIL=0` - as dead flags: **12 false positives** across four launchers. A flag only
   matters if it is handed to the interpreter, so it now walks back over the backslash-continuation lines that precede a `.py` invocation and reads only
   those.
2. **Second version reported 0 flags on everything, including a deliberately planted dead flag.** The bug was assembly order: I appended the preceding
   continuation lines to the END of the joined string, so `split(".py")[0]` cut off exactly the env prefix being searched for. A guard that silently
   never fires is worse than no guard, because it manufactures confidence.

## Validated both ways before trusting it

```
planted dead flag  TOTALLY_DEAD_FLAG=1  -> DEAD FLAGS TOTAL: 1, exit 1   (caught)
live flag          RANK_BOOST=1.0       -> DEAD FLAGS TOTAL: 0, exit 0   (no false positive)
four real launchers (rank50, rank50resume, untied6k, agekey)             -> 0 dead flags
```

The negative control matters as much as the positive one: a checker that flags everything is as useless as one that flags nothing.

## Honest limit

It is a **name-based heuristic**. It proves a name is never mentioned; it cannot prove a value is used correctly. That is exactly enough to catch all
three incidents above, because in each case the name appeared nowhere in the code that was supposed to read it.
