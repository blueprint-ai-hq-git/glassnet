# 84% is a fair-weather number: a 2-pixel shift costs 27.8 points (2026-11-18)

Every accuracy figure this project has published for the readable image model was measured on the **clean** test set. A number that only holds on pristine inputs is not a
shippable number, so `tools/vision_robustness.py` applies the shifts a real user produces - 3,000 held-out images, same model, same labels.

```
input                           accuracy      cost
clean                             84.60%    +0.00
brightness +0.10                  82.97%    -1.63
brightness +0.20                  78.13%    -6.47
brightness -0.20                  78.77%    -5.83
contrast x1.30                    83.63%    -0.97
contrast x0.70                    81.83%    -2.77
gaussian noise sd 0.05            84.23%    -0.37
gaussian noise sd 0.10            83.30%    -1.30
blur 3-tap binomial               82.73%    -1.87
shift 1 px right                  76.83%    -7.77
shift 1 px down                   78.33%    -6.27
shift 2 px right                  56.77%   -27.83     <- collapse
```

## The weakness is specific, which makes it nameable and testable

**Translation is the killer, not noise.** Gaussian noise at sd 0.10 costs 1.3 points and blur costs 1.9, but a **two-pixel shift costs 27.8** - accuracy falls from 84.60% to
**56.77%**, barely better than guessing on a 10-class problem with 10% of the training signal. Brightness is intermediate (-6.5 at +0.20), which is unsurprising: the concept
set is built on brightness and contrast formulas, so a global brightness shift moves many concept activations at once.

The cause of the translation sensitivity is structural and exactly as the architecture predicts: the model reads a **fixed patch grid** (4 patches of 14x14), so shifting the
image by a pixel moves content across patch boundaries and changes what each patch sees. Concepts like `top_half_heavier` and `left_right_symmetric` are **alignment-sensitive by
construction** - a shifted image genuinely has a different top half.

## What this means for anyone using the model

The honest disclosure is: **the readable image model expects a centred, fairly-exposed input.** Within those conditions it is at parity with a plain model of the same size and
fully readable. Outside them it degrades in a way that is predictable from its concept set rather than mysterious.

## The test of whether it is fixable, pre-registered

`lens/scripts/run_shift_aug.sh` trains the same parity config with **random translation augmentation up to 2 pixels** during training, then re-runs the whole robustness table on
the clean and shifted test sets:

- **P1** clean accuracy drops at most 1.5 points (>= 83.1%)
- **P2** accuracy under a 2-pixel shift rises from 56.77% to **at least 75%**
- **P3** the clean-to-2px gap narrows from 27.83 points to **under 10**
- **FALSIFIER** 2px-shift accuracy stays below **65%**: then grid alignment is not an augmentable nuisance but a structural property, and the disclosure above becomes the final
  word on it rather than an intermediate note.
