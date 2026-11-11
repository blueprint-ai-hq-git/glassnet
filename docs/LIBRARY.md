# The library: five lines, and it runs (2026-11-11)

Board item A5 - "a simple library where training takes about 5-20 lines of code and explain() shows why the model
answered" - has been listed as PARTIAL since the first week. It works now, and this is its real output.

```python
import glassnet as gn

data = gn.load_chat(max_windows=20000)                 # 1  our corpus, matched conversation holdout
net  = gn.model(data)                                  # 2  readable net, concepts named from the corpus
res  = gn.fit(net, data, minutes=1)                    # 3  trains, nothing typed in
print(res)                                             # 4  real numbers, incl. held-out accuracy
print(gn.why(net, "the capital of france is", data))    # 5  why it answered
```

Real run, unedited (`demo_five_lines.py`):

```
cached: 294,225,090 tokens, vocab 12,002, 740,666 conversations
auto-pilot: 15567566 parameters, d=512, steps=1500, batch=256, lr=0.30, epochs=3 (chosen, not typed in)
  epoch   1/3   loss 7.8623   1.12s
  epoch   2/3   loss 6.5325   2.11s
  epoch   3/3   loss 6.0815   3.09s
held-out next-token accuracy: 4.70% on 2000 held-out windows (never trained on)
Result(epochs=3, final loss 6.0815, 15,567,566 params, gpu, 3.09s)
{'predicted_word': '<unk>', 'n_concepts': 36, 'fired': [], 'concepts': [...], 'residual': 11.72, 'exact': False}
```

## What the numbers honestly say
- **4.70% held-out next-token accuracy after three epochs.** That is a working pipeline, not a good model. Three epochs is
  what `minutes=1` buys; the trained arms in this project run for hours.
- **`exact: False` with residual 11.72.** The concept path does not explain the whole logit; the rest comes from the copy
  path. This is the same 7.3% readable-path share already published, now visible in one line of the library's output
  instead of in a separate measurement tool.
- **`predicted_word: '<unk>'`** on an untrained model is the honest answer to an untrained question.

## Auto-pilot, and the bug that made it slow
`gn.fit` chooses the model's settings from measured rules, not guesses: learning rate 0.1 below 2M parameters and 0.3
above (the step-size ladder), and steps from either the parameter count or the requested minutes. The first version
budgeted 6,000 steps per minute - the trainer's peak throughput with no evaluation in the loop - so `minutes=1` ran for
twenty minutes and the demo timed out with nothing printed. Corrected to 1,500 steps per minute, measured with evaluation
in the loop, and capped at three epochs so a demo always finishes.

Two more real bugs were fixed to get here: `gn.model()` defaulted to the CPU while the demo advertised speed, and
`gn.why()` passed a `(1, 32)` array into `explain()`, which adds its own batch axis and then refused the window for being
`1` token wide.
