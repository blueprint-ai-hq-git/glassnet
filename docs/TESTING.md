# Test GlassNet yourself

The point of this project is that you do not have to trust our numbers. Here is how to produce your own.

## 1. Verify the mathematics (5 minutes, CPU only)

```bash
python -m venv .venv && . .venv/bin/activate && pip install numpy
python tests/test_gradcheck.py
```
Every operation must print `PASS`. If any prints `FAIL`, stop and open an issue with the numbers it printed.

## 2. Sanity: can the engine learn XOR?

```bash
python gates/phase1_xor.py
```
Expected: 4/4 correct, loss under 1e-3. This catches a broken backward pass immediately.

## 3. Reproduce the baseline

```bash
python gates/phase1_fmnist.py --seeds 0 1 2
```
Writes a checkpoint per epoch and a JSONL log. Report the three final accuracies you get, with your machine
spec - we will publish your number next to ours, including if it is worse.

## 4. Readability test (the interesting one)

```bash
python -m lens.explain_check --checkpoint runs/chat/<best>.npz
```
It asks the model a set of prompts and prints which named concepts fired. **Your job is to say whether the
explanation is honest.** If the model says "because of `mentions_food`" and the text has no food in it, that
is a false fire - file it, with the sentence.

## 5. Report back

Open an issue titled `RESULT: <what you ran>` with:
1. Your hardware and Python version
2. The exact command
3. The final numbers (raw log, not a summary)
4. Whether anything surprised you

## Known traps

- **numpy vs CuPy, measured 2026-11-06.** The GPU training venv needs `numpy >= 2.0, < 2.6`, because
  `cupy-cuda12x 14.2.0` declares `Requires-Dist: numpy<2.6,>=2.0`. With `numpy==1.26.4` installed,
  `import cupy` fails with `ImportError: numpy.core.multiarray failed to import` and every `--device gpu`
  run dies at model construction, after the corpus is built and concepts are measured (so it looks like a
  training bug and is not one). Tested working pair: `numpy==2.4.6` + `cupy-cuda12x==14.2.0` - GPU matmul
  runs, `tests/test_gradcheck.py` GATE 1 PASS on all nine entries. CPU-only installs can stay on 1.26.4.
- The dataset cache is large; do not commit it.
- Batch 4096 is a speed setting, not an accuracy setting - it halves held-out accuracy at fixed epochs.
