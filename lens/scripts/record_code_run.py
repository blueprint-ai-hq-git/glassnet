"""record_code_run.py — the code model, and the two test bugs that hid its real behaviour.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "code")

run = json.load(open(os.path.join(P, "code1_replies.json")))
re = json.load(open(os.path.join(P, "code1_reexec.json")))
h = run["history"][-1]

LOG = f"""
---

## 2026-10-07 21:10 — THE CODE MODEL, AND TWO TEST BUGS THAT MADE IT LOOK WORSE THAN IT IS.

### WHAT TRAINED

```
                       params      ppl     acc   concepts   syntax   runs
code smoke            656,950  1112.21   7.35%      2/38      0/8     0/8
code real           1,502,774    44.26  36.44%     28/38    0/25    0/25
```

**ppl 44.26 against the chat model's 65.96, accuracy 36.44% against 26.44%.** Code is lower entropy
than conversation, so the same engine does measurably better on it. That part of the prediction held.

**28/38 concepts pass.** The failures are the same shape as the chat model's: AUROC 1.0000 with a
firing rate under the 2% floor (`uses_len` 0.007, `uses_range` 0.003, `mentions_order` 0.005).

### THE "0/25" WAS MY TEST, TWICE OVER

The run reported **syntax 0/25, runs 0/25** and I nearly wrote that down as the model's limit. Two
bugs in the harness, both mine:

**Bug 32 — the token join.** `code_data.TOKEN_RE` drops whitespace and `train_code.gen()` joined with
`""`, so `def foo ( x ) :` came back as `deffoo(x):`. A generator that cannot emit a space cannot
emit a function definition.

**Bug 33 — the prompt demanded something impossible.** The prompt was cut at `def `, so the model had
to INVENT a function name. Names are the highest-entropy part of code -- any identifier is legal --
so it emitted `<unk>` and the snippet was dead before the body began. **I built a test that could not
pass, then reported its result as if it were the model's ceiling.**

### WHAT IT ACTUALLY DOES (fair prompt: full signature given, model writes the body)

```
BEFORE join=''   syntax 4/25   runs 4/25
AFTER  join=' '  syntax 4/25   runs 4/25
```

**4/25.** And the reason those 4 pass is itself a caution: the join fix turned `<unk>` into
`< unk >`, which **is** valid Python -- a comparison chain -- so those snippets parse and run while
computing nothing.

**So 4/25 measures "parses and does not crash". It does NOT measure correctness. Not one generated
function has been checked against the expected answer.** Reporting 4/25 as a success would be the
same mistake as the 0/25, pointed the other way.

### THE HONEST READING

1. The engine learns code structure better than conversation (44.26 vs 65.96 ppl) -- real, measured.
2. It cannot invent identifiers, so it cannot write a function from a bare `def ` prompt.
3. Given a full signature it writes a body that parses and runs 16% of the time.
4. **Whether any of them computes the right answer is UNTESTED.** The next honest step is to run the
   generated functions against the reference outputs on the same inputs and count exact matches.
5. The output is a code/prose MIX -- it alternates between writing code and explaining it, because
   the training text is problem-then-solution and it learned both halves as one stream.

### PATTERN, NOW FOUR TIMES

Every one of the four verdict bugs in this project has been a claim attached to a correct
measurement -- never a wrong measurement. This time the same failure appeared in the TEST DESIGN
rather than the summary: a harness that could not succeed, reported as a model limit. The rule that
catches it: **before reporting a zero, prove the test can produce a non-zero.**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

print("=" * 100)
print("CODE MODEL — the numbers")
print("=" * 100)
ar = run.get("args", {})
print(f"  params {run['params']:,}   d={ar.get('d','?')}  W={ar.get('window','?')}  "
      f"K={run['concepts']['n_kept']}")
print(f"  val ppl {h['val_ppl']:.2f}   acc {h['val_acc']*100:.2f}%")
print(f"  bottleneck {run.get('ratio', 0):.0f}:1")
print()
print(f"  concepts {run['audit']['n_pass']}/{run['audit']['n_concepts']}   "
      f"coverage {run['audit']['coverage']*100:.2f}%")
print()
print("  execution test (fair prompt, full signature given):")
print(f"    parses and runs : {re['stats']['ran']}/{re['stats']['total']}")
print(f"    checked correct : 0/{re['stats']['total']}   <- NOT TESTED. do not report this as a win.")
print("=" * 100)
