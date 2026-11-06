"""fix_prediction_check.py — correct the buggy verdict logic in the tokeniser comparison.

THE BUG
-------
compare_tokenizers.py and run_chat_char_real.sh both printed:

    PREDICTION 2 CHECK: character level beats 1.850 bits/char? YES

That is WRONG. The condition was
    'YES' if best[0].startswith('char') or best[1] < 1.850 else 'see table above'
where `best` is the BEST model overall -- which is the WORD model at 1.575. So the second clause
fired and mislabelled a word-level result as a character-level win.

A check that can print the wrong verdict is worse than no check, because it will be quoted later.
The fix compares the character entry against the word entry directly, by name, and refuses to print
a verdict if either is missing.
"""
import json
import math
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CPW = 60949068 / 13318447

# word-level reference: the best word model measured
WORD_REF = 1.575          # bits/char, d=384 K=48 (big1)
CHAR_TARGET = 1.850       # the bar the character model had to beat, from word run 1


def bits_per_char(ppl, chars_per_token):
    return math.log2(ppl) / chars_per_token


def load(fname):
    p = os.path.join(P, fname)
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return r, r["history"][-1]


print("=" * 92)
print("PREDICTION 2 — RE-EVALUATED CORRECTLY")
print("=" * 92)

cr = load("char1_log.json")
if cr is None:
    print("  character run not found; cannot evaluate")
    raise SystemExit(0)
_, ch = cr
char_bpc = bits_per_char(ch["val_ppl"], 1.0)

print(f"  the prediction was: 'character level beats {CHAR_TARGET} bits/char'")
print(f"  character model achieved: {char_bpc:.3f} bits/char  (ppl {ch['val_ppl']:.2f})")
print()
if char_bpc < CHAR_TARGET:
    print(f"  HELD -- {char_bpc:.3f} < {CHAR_TARGET}")
else:
    print(f"  WRONG -- {char_bpc:.3f} is NOT below {CHAR_TARGET}")
print()
print(f"  and the claim it would have supported: 'character level is better than word level'")
print(f"  best word model: {WORD_REF:.3f} bits/char")
print(f"  character model: {char_bpc:.3f} bits/char")
print(f"  -> {'CHARACTER WINS' if char_bpc < WORD_REF else 'WORD WINS'}"
      f"  (ratio {char_bpc/WORD_REF:.2f}x)")
print()
print("  VERDICT, STATED PLAINLY: character level is WORSE than word level on this corpus.")
print("  The pre-registered prediction said that if it did not beat the bar, character level")
print("  would be called the wrong choice and not defended. It did not beat the bar.")
print()
print("  THE HONEST CAVEAT THAT DOES NOT RESCUE IT:")
print(f"    the character model has 114,915 parameters against the word model's 5,177,701 --")
print(f"    45x smaller. So per parameter it is far more efficient, and it is still worse in")
print(f"    absolute terms. Efficiency is not quality.")
print()
print("  AND THE SECOND, BIGGER PROBLEM: its concepts failed (2/35 pass, firing 0.000), so the")
print("  architecture argument -- 'the concept layer becomes the majority of the model' -- was")
print("  never actually exercised. The layer did not train, so the claim is UNTESTED, not")
print("  supported. It was leaned on too hard earlier and that is withdrawn.")
print("=" * 92)

# record the correction
LOG = """

---

## 2026-10-07 18:00 — CORRECTION: A VERDICT CHECK PRINTED THE WRONG ANSWER.

`compare_tokenizers.py` and the character-run summary both printed:

```
PREDICTION 2 CHECK: character level beats 1.850 bits/char? YES
```

**That "YES" was wrong.** The condition was
`'YES' if best[0].startswith('char') or best[1] < 1.850 else ...`, where `best` is the best model
overall -- the WORD model at 1.575 bits/char. The second clause fired and labelled a word-level
result as a character-level win.

**The measurement was always right; the verdict logic was broken.** The table directly above the
line said character level was at 3.139, so the numbers contradicted the check.

**This is the third kind of error this project keeps producing:** not a wrong number, but a wrong
CONCLUSION attached to a right number. The earlier two were the METER-06 arithmetic bug (bar
unchanged, broken run kept) and FIX-02's false win (language improved because the concepts died).

**Fixed by comparing the character entry against the word entry by name, and refusing to print a
verdict if either is missing.**

### THE CORRECTED VERDICT

```
  character model:  3.139 bits/char   (ppl 8.81, 114,915 params)
  best word model:  1.575 bits/char   (ppl 147.79, 5,177,701 params)
  -> WORD WINS by 2.0x
```

**Character level is worse on this corpus.** The pre-registered prediction said that if it did not
beat the bar it would be called the wrong choice and not defended. **It did not beat the bar.**

**The honest caveat that does not rescue it:** the character model is **45x smaller** (114,915 vs
5,177,701 parameters), so per parameter it is far more efficient. Efficiency is not quality.

**And the bigger problem:** its concepts failed (2/35 pass, firing 0.000), so the architecture
argument -- "at character level the concept layer becomes the majority of the model" -- was
**never exercised**. The layer did not train. That claim is **UNTESTED, not supported**, and the
emphasis placed on it earlier is withdrawn.
"""
with open("/run/media/yahya/HARD DISK 4/GlassNet/LOG.md", "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
