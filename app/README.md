# GlassNet Math-1 app

Runs the 67,484-parameter model entirely in the browser. No server, no install, no account, no network once the page is loaded.

| platform | how to run |
|---|---|
| Linux | `bash app/run-linux.sh` |
| Windows | right-click `app/run-windows.ps1` -> Run with PowerShell |
| Android / iPhone | serve the folder from the machine (`python3 -m http.server 8781`), then open `http://<machine-ip>:8781/` on the phone |

## What it shows

* **The working** - the tokens it writes, column by column: units and carry, tens and carry, hundreds.
* **Answer selection** - the real logits, the probabilities and the gap between the winner and the runner-up. The winner is the
  largest number; there is no hidden step.
* **Neurons** - the 64-number state, and the exact contribution each of the 4 rank dimensions makes to the winning score. The four
  contributions add up to the winner's logit to the last decimal, because the output is a linear map.
* **Attention** - which earlier tokens each head reads. Causal by construction, so no future token is available.
* **How it learned** - the measured acquisition curve: near zero for 1400 steps, then the whole procedure on inside a 400-step window.

## Known weakness, stated on screen

100.0% with a fixed filler token in the four unused input slots, **75.3% with varying filler**. The app uses varying filler, which is
the harder and more honest condition. Closing that gap is a training fix, not a display fix.

## Files

* `model/math1.safetensors` - standard container
* `model/math1.gguf` - GGUF container
* `model/math1_weights.json` - what the page loads (base64 float32, so the app has no dependencies)
