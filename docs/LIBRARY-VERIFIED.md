

---

# THE 5-20 LINE LIBRARY AND explain() ARE NOW VERIFIED, END TO END (2026-11-22 14:47)

The brief promises two things that had never been checked together: **training in about 5-20 lines**, and **explain() showing why
the model answered**. Both are now measured on a freshly trained model, with the exports read back from disk.

## What ran, and what it returned

```
net = Net(V, [Concept("has_marker", [3])], d=32, window=8)
res = train(net, Xtr, ytr, epochs=6, batch=64, lr=0.3)
```

```
trained: Result(epochs=6, final loss 0.3376, 937 params, cpu, 0.18s)
held-out accuracy above the most-frequent-answer baseline:        PASS
explain() returned 573 characters:                                PASS
save -> load: worst parameter difference 0.000e+00                PASS   (exactly zero, not "loadable")
safetensors: 5 tensors, 4412 bytes, read back from disk           PASS
gguf: 4544 bytes, magic bytes b"GGUF" present                     PASS
legacy checkpoint refused with a clear message, not a KeyError    PASS
OUTCOME: LIBRARY_VERIFIED
```

`explain()` returns real content, not a placeholder: `{'predicted_id': 11, 'predicted_word': '11', 'predicted_logit': ...,
'n_concepts': 1, 'fired': ...}` - the prediction, the concept that fired, and the terms that produced the answer.

## The part that matters: my verification was broken three times before it was true

This is the fifth time today the measuring instrument was the fault, and the worst of the five, because one of the bugs made the
check **incapable of failing**:

1. **`forward()` returns a dict, not an array.** `np.asarray(dict)` is a 0-d object scalar, so the accuracy comparison was
   nonsense and printed **0.00%** while `explain()` was simultaneously printing a sensible prediction. The model was fine; my
   measurement was broken.
2. **The dict's values are our own `Tensor`, not arrays**, so the fix above still gave 0.00%. The numbers live in `.data`.
3. **`v is False` does not catch `numpy.bool_(False)`**, so a check that had genuinely failed printed
   **`OUTCOME: LIBRARY_VERIFIED`**. A gate that cannot fail is worse than no gate at all, and it is exactly the failure mode this
   project has been hunting all day.

Only after all three were fixed did the suite report a real 7-of-7 with a gate that can actually go red - and the first honest
run it produced was `LIBRARY_BROKEN: accuracy`, which is what a working gate looks like.

## Also fixed while here

`load()` now **refuses a legacy checkpoint with a readable message** instead of raising `KeyError: '__meta__ is not a file in the
archive'`. Filling in defaults would have rebuilt a different model than the one trained - the file's own comments record a case
where that produced a 24.96 logit difference - so refusing is the only safe behaviour.
