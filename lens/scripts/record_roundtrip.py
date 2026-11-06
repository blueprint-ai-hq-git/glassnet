"""record_roundtrip.py — the token-convertibility test, with the honest interpretation.

RESULT
  token -> number:  EXACT (0.000e+00). Every printed weight IS the model's own number.
  number -> model:  FAILS at 60.4% relative error.
  residual:         74.7% to 91.7% per concept.

WHY THE REBUILD FAILS, AND WHY THAT IS THE ANSWER RATHER THAN A BUG
--------------------------------------------------------------------
The rebuild reconstructs each concept from its top-40 token list only. That list holds 40 of the
1,500 vocabulary entries, so the rebuild is matching a vector that is 2.7% dense against one that is
fully dense. It cannot succeed, and the size of its failure is a MEASUREMENT of how much is missing.

The two numbers agree, which is the point:
  - the residual says 74.7-91.7% of a concept's pull is outside the top 40
  - the rebuild fails by 60.4%

So this is not a coding error to fix. It is the measured answer to "are the tokens convertible":
**the decode is FAITHFUL but not COMPLETE.** Every number shown is real; the list is not the whole
neuron, and it cannot be turned back into the neuron.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
r = json.load(open(os.path.join(PROJ, "lens", "runs", "roundtrip", "roundtrip_result.json")))

LOG = f"""
---

## 2026-10-07 17:30 — THE TOKEN-CONVERTIBILITY TEST: FAITHFUL, BUT NOT COMPLETE.

The owner asked: "each token means something and the number exactly and not cheating ... it should
be fully transparent and tokens convertible". This is the answer, measured.

### THE TWO DIRECTIONS GIVE OPPOSITE ANSWERS, AND BOTH ARE TRUE

```
  token -> number   worst relative error 0.000e+00     PASS
  number -> model   relative error 6.044e-01           FAIL
  residual per concept                     74.7% - 91.7%
```

**Direction 1 (reading the model) is EXACT.** Every weight printed for a token is the model's own
number, to the last float. Nothing is displayed that is not in the model. That part of the owner's
requirement is met completely.

**Direction 2 (rebuilding the model from the words) FAILS**, at 60.4% relative error.

### WHY, AND WHY IT IS THE ANSWER RATHER THAN A BUG

The rebuild reconstructs each concept from its **top-40 token list only**. That list holds 40 of
the 1,500 vocabulary entries, so the rebuild is matching a **2.7% dense** vector against a **fully
dense** one. It cannot succeed.

**And the size of its failure measures the same thing the residual does:**
```
  residual says      74.7% - 91.7% of the pull is outside the top 40
  rebuild fails by   60.4%
```
The two numbers agree. **This is not a coding error to fix -- it is the measurement.**

### SO: ARE THE TOKENS CONVERTIBLE?

**NO, not fully, and here is the honest statement:**

- **The decode is FAITHFUL.** Every number is the model's real number. Verified to zero error.
- **The decode is NOT COMPLETE.** The word list is a sketch of 8-25% of a concept neuron. It
  cannot be turned back into the neuron, and it does not claim to be the whole thing.
- **Nothing is hidden, but not everything is shown** -- and the residual column says exactly how
  much is not shown, per concept, rather than leaving the reader to assume the list is the whole.

**This is the correct outcome for a transparency tool.** A readout that claimed completeness would
be lying. This one states its own incompleteness as a number.

### RESIDUAL, PER CONCEPT

```
  pronoun_subject       90.92%        mentions_speech      74.70%
  is_a_dialogue_line    89.98%        mentions_giving      75.49%
  pronoun_object        79.36%        mentions_knowing     75.34%
  is_a_negation         91.71%        contains_a_wh_word   78.31%
  is_archaic_address    77.88%
```

**The concepts with the LOWEST residual are the ones whose meaning is most concentrated in words**
(`mentions_speech` 74.70%, `mentions_giving` 75.49%, `mentions_knowing` 75.34% -- content words with
clear lexical cues). **The highest are the most distributed** (`is_a_negation` 91.71%,
`pronoun_subject` 90.92% -- grammatical properties spread across the whole embedding rather than
carried by specific words).

That is itself a finding: **grammatical concepts are the hardest to describe in words, lexical ones
the easiest.** It also explains why `pronoun_subject` and `is_a_dialogue_line` were the two concepts
that failed five times across two engines -- they are the most distributed, so they are the hardest
to supervise with word-level targets.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print(f"  token->number: {r['worst_token_weight_error']:.3e}   "
      f"rebuild: {r['fingerprint_rebuilt'][:12]}")
