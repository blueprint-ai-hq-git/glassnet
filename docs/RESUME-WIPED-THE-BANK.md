

---

# BUG: every resume silently wiped the trained memory bank (found and fixed 2026-11-19)

This is the most consequential defect found in the project so far, because it invalidates the *interpretation* of several arms rather than one number.

## The bug

The trainer restores weights from a checkpoint, and then - further down the file - initialises the memory bank:

```python
for _k in p:                 # restore ordinary weights ...
    if _k in _z.files: p[_k] = _z[_k]
...
if MEM_SLOTS > 0:
    _mrng = np.random.default_rng(1234)
    p["mem_Wq"] = (_mrng.standard_normal((H, H)) / np.sqrt(H))   # ... then overwrite the bank
    p["mem_Wk"] = ...
    p["mem_Wo"] = np.zeros((H, H))                                # trained Wo back to ZERO
    p["mem_decay"] = _mrng.uniform(0.85, 0.95, MEM_SLOTS)
```

`mem_Wq`, `mem_Wk`, `mem_Wo`, `mem_decay` are **trained parameters** (the resumed checkpoint's `mem_Wo` has |sum| = 111.06, i.e. it was trained), and the
restore loop never touched them because they are created *after* it. The trainer logged "weights loaded" while reinstating a **virgin bank every time**.

## How it was found

A resume from `ep8000` read `loss 4.7916 / acc 42.43%` at step 8025 where the original process had read `4.7316 / 44.12%`, with the learning rate
proven equal (0.0116) and the weights loaded. An offline replay from the same checkpoint reproduced the **original** exactly, which proved the
checkpoint was faithful and the *process* was wrong.

Fixing it also required noticing a second defect: guarding the init block on the set of restored tensors did nothing, because that set was built by
iterating `p`, which does not yet contain `mem_*`. The "fixed" run produced numbers **identical to the buggy one** - a silent no-op fix. The restore
loop now iterates the **checkpoint's** keys.

## Proof of fix

```
step 8025          loss      ema      acc      gnorm    lr
original          4.7316   4.0703   44.12%   2.95   0.0116
offline replay    4.7316   3.9774   44.12%   2.95   0.0116
resumed (buggy)   4.7916   4.0391   42.43%   3.46   0.0116
resumed (FIXED)   4.7316   3.9774   44.12%   2.95   0.0116
```

A resume is now a true continuation on loss, accuracy and gradient norm. (`ema` restarted because it was not stored; it is now saved and restored too.)

## Which results are affected

Any arm launched with `RESUME=...` **and** a bank was training a fresh bank while claiming to continue one. Eight runner scripts:

`run_memchan.sh`, `run_memchan_sym.sh`, `run_memchan_sym_resume.sh`, `run_tgt_delay16.sh`, `run_td4_dose.sh`, `run_td4_extend.sh`, `run_td16_extend.sh`, `run_td4_resume8000.sh`

- **Clean:** `td16` and `td4` resumed from a **bankless** base, so a fresh bank was correct there.
- **Confounded:** `td16x` (from `td16 ep7000`) and `td4x` (from `td4 ep7000`) **wiped the bank those arms had just trained for 7000 steps**. So the
  "extension" arms were not extensions of the same model - they were a new bank trained from scratch for 1500 steps on carried-over language weights.
  The same applies to the memory-channel resume arms.

**Therefore the previous conclusion "more delay-16 training erodes the middle band" is confounded and is withdrawn pending the corrected re-run.** The
erosion may be what a bank looks like when it is reset, not what continued training does.

The corrected extension (`td16xf_r090`, resuming `td16 ep7000` with the bank preserved) is running now, with its predictions written before launch: the
middle band should stay within ~2 points of the ep7000 high-N value (0.43%) rather than collapsing. If it collapses anyway, the confound explanation is
wrong and yesterday's conclusion stands.
