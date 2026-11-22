

---

# IT SCALES: ONE, TWO AND THREE ITEMS ALL AT 100% WHEN THE WORKING IS WRITTEN (2026-11-22 15:04)

The written-comparison fix was found on two items. The honest next question was where it stops. Answer: at least one rung further.

## Three digits, eight written tokens, two seeds

```
  [pos1][digit1] [pos2][digit2] [pos3][digit3] [pos of the LARGEST][the largest]
```

```
seed 0: position 100.0% | value 100.0% | both 100.0%   moved 100.0% | swapped value 100.0% | swapped position 100.0%
seed 1: position 100.0% | value 100.0% | both 100.0%   moved 100.0% | swapped value 100.0% | swapped position 100.0%
P1, P2, P3 all MET. Falsifier not fired. Both seeds "it really follows the input".
```

**Six of the eight tokens are working, and the answer is the last pair.** The model tracks all three digits, compares them, and reports the winner's position and value - and survives every perturbation.

## The full ladder now, same task family, same 95% bar, same 67k parameters

| items to choose between | answer tokens | clean | flip case | verdict |
|---|---|---|---|---|
| one | 2 | 100.0% | 100.0% | reliable |
| two | 2 (no working) | 100.0% | **88.0% / 94.0%** | **failed** |
| two | 6 (written) | 100.0% | 100.0% | reliable |
| three | 8 (written) | 100.0% | 100.0% | reliable |

**The controlling variable is not the number of items - it is whether the intermediate state is written down.** Two items with no
written working fail; three items with written working pass. That is the sharpest form of this project's thesis measured so far, and
it was obtained by holding parameters fixed and changing only the tokens.
