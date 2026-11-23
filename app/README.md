# GlassNet Runner

**One app. The model is a separate file.** The app contains no model and no network code path — on Android it is built without the
INTERNET permission, so it cannot open a connection even if asked.

| platform | how to run |
|---|---|
| Android | install `GlassNet-Runner.apk`, tap **Open model**, pick a `.glassnet` file |
| Linux | `bash app/run-linux.sh`, then **Open model** |
| Windows | run `app/run-windows.ps1`, then **Open model** |

On a computer you can also drag a `.glassnet` file onto the window.

## Why it is separated

A runner that carries a model can only run that one model. This one reads the model's own metadata and adapts: dimensions, layers,
vocabulary, context length, how many tokens the written working has. So the same app runs any GlassNet model of this shape — a maths
model today, a talking model next.

## Model file format: `.glassnet`

JSON, so it is readable and needs no dependencies:

```
{ "__meta__": { task, vocab, context, out_start, write_len, d_model, layers, heads, emb_rank, params, heldout_accuracy },
  "tensors": { "A": {shape,b64}, "B": {...}, ... } }
```

The same weights are also exported as `.safetensors` and `.gguf` for standard tooling; `.glassnet` is just the form this app reads.

## What it shows

* **The working** — the tokens the model writes, column by column: units and carry, tens and carry, then the leading digit.
* **Answer selection** — the real scores, probabilities, and the gap between winner and runner-up. The winner is simply the largest.
* **Neurons** — the exact contribution of each rank dimension to the winning score; they sum to it because the output is linear.
* **Attention** — which earlier positions each head reads. Causal, so no future information exists for it to use.
* **How it learned** — the measured acquisition curve.

## Honest note

Math-1 scores 100.0% when the unused input slots hold a fixed filler token and 75.3% when they vary. The gap is real and is the next fix.
