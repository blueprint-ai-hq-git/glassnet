
## Pre-registration: training on its own rollouts, and the trap that would have faked a win (2026-11-17)

Exposure bias is measured: teacher-forced **22.19%**, free-running **4.06%**, repetition **0.93**. The lever is training the model on its own rollouts. **This is the
pre-registration, written before any code runs, and it exists mainly because the obvious implementation is a trap.**

## The trap, found by reading the target line before writing the training loop

`lens/scan_direct_batched.py:28`:

```python
tg = chunks[:, 1:][:, pos].reshape(-1)      # the TARGETS come from the same array as the INPUTS
```

`train_step(p, chunks, ...)` takes **one** array and uses `chunks[:, :-1]` as inputs and `chunks[:, 1:]` as targets. So the "cheap" scheduled-sampling implementation - build
chunks from the model's own predictions and pass them in - would make the model **predict its own predictions**. Every target would be a token the model itself emitted.

The consequences are specific and would look like success:

- training accuracy would jump toward **100%** almost immediately,
- the loss would fall,
- teacher-forced held-out accuracy would barely move,
- and the arm would be reported as a breakthrough when it had learned to copy itself.

**That is the same defect class as every instrument failure this week: a number that looks like progress and measures nothing.** One line of reading prevented it.

## The required change

`train_step` gains a second array so inputs and targets cannot be conflated:

```python
def train_step(p, chunks, tgt_pos, UNK, lr, ..., tgt_chunks=None):
    ...
    src = chunks if tgt_chunks is None else tgt_chunks       # inputs from chunks, TARGETS from the corpus array
    tg  = src[:, 1:][:, pos].reshape(-1)
```

With `tgt_chunks=None` the behaviour is bit-identical to today, which is checkable: same loss and gradient norm on the same inputs before and after the change.

## Mandatory verification before the arm launches

1. **Targets are corpus tokens.** assert `tgt_chunks` is the untouched corpus slice, and print `max |constructed_input - corpus|` over scored positions to confirm the inputs
   differ from the targets by the expected amount rather than by zero.
2. **Rollouts actually entered the input.** print the fraction of constructed input tokens that differ from the corpus token at that position. It must be non-zero and should
   rise with the scheduled-sampling rate; **0.0 means the construction silently did nothing**, which is the silent-no-op defect this project bans.
3. **Gradient check** of the modified path against finite differences, on the real shapes, before any training.
4. **Fake-fix detector, run as a hard gate at step 25.** If teacher-forced accuracy reaches **above 90%** within 25 steps, targets have been contaminated: **STOP and report**, do not
   continue the run. A real scheduled-sampling arm should leave teacher-forced accuracy near its starting 16.9%.
5. **Same-protocol baseline** re-measured in the same run: teacher-forced and free-running accuracy on the same 320-token probe `tools/decode_path.py` uses.

## The gate and the falsifier, fixed now

- **Gate:** free-running accuracy **> 10%** on the 40-step probe, with teacher-forced accuracy **>= 16%** and repetition **< 0.5**.
- **Falsifier:** if free-running accuracy stays **< 6%** after treatment, exposure bias was not the binding constraint. The loop must then be sought in generation-context
  construction versus the training window, not in the model's exposure to its own errors.
- **Void condition:** if the fake-fix detector fires, the arm is **VOID by construction** and no conclusion may be drawn from it. (This project has already had one such arm.)

## What would count as a win

Not a higher teacher-forced number. **The only thing that counts is free-running accuracy rising while teacher-forced accuracy holds and repetition falls** - because
free-running accuracy is the generation condition, and that is the condition every previous scan run failed in without measuring it.
