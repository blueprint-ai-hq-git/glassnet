# SCALING AND SMARTNESS - the honest path (written 2026-10-18)

Correcting an overstatement I made: I said a 1B model "cannot be trained on this machine". That was
wrong. It is an engineering problem with three real constraints, and here they are measured.

## The machine (measured today)
- GPU: GTX 1660 SUPER, **6,144 MiB VRAM** (~5.0 usable), no tensor cores, fp32 ~5 TFLOPS
- RAM: 14 GB total, ~4 GB free right now
- Disk: Hard Disk 4, 466 GB total, **101 GB free**
- CPU: 12 threads

## Constraint 1 - memory (solvable, and we know how)
Plain training in VRAM needs, for N parameters in fp32:
- parameters 4N bytes, gradients 4N bytes, Adam moments 8N bytes -> **16 GB for 1B**, does not fit.
- Drop Adam for plain SGD/momentum: **8N = 8 GB**, still does not fit 6 GB, fits in RAM tightly.
- **Layer streaming (the fix):** keep parameters and gradients on disk (memmap), bring ONE layer at a
  time to the GPU, update it, write it back. Resident footprint becomes one layer, not the model:
  a 4096x4096 layer = 67 MB fp32. Fits in 6 GB with room for activations.
- Activation checkpointing on top of that keeps batch memory flat.
This is the same family of techniques used before on this machine for far bigger models (disk-backed
layer-by-layer training of 70B, MoE upcycling at 2x3072 inside 6 GB). GlassNet does not have such a
trainer yet; writing it is a concrete, testable job.

## Constraint 2 - data (this is the real wall)
- Our chat corpus: **38.8 M tokens** (the widest one).
- Standard guidance is ~20 tokens per parameter for a well-trained model, so 1B wants **~20 B tokens**.
  We are short by a factor of **~500**.
- Disk is not the blocker: 20 B tokens is roughly 40-60 GB of text, and we have 101 GB free.
- The blocker is licensed supply and download time. Allowed licences are MIT, Apache-2.0, CC0,
  CC BY 4.0 - that rules out several of the popular web corpora (ODC-By, unclear licences). The clean
  big option is the Apache-2.0 family (RedPajama-style), plus CC BY 4.0 sets; it must be streamed and
  sliced, not stored whole.

## Constraint 3 - time and power (the honest estimate)
Training compute is roughly 6 x parameters x tokens. For 1B x 20 B tokens that is **1.2e20 FLOPs**.
- At a realistic 5 TFLOPS: **~280 days** of continuous running.
- Optimistically at 20 TFLOPS with fp16 and good utilisation: **~70 days**.
- Power: 120 W for 70 days is ~200 kWh; at ~Rs 8/kWh that is roughly **Rs 16,000-20,000** of
  electricity, on top of the months of a busy machine.
So: possible, not impossible - but a months-long campaign, not a weekend.

## What that means for "beat every model in the world"
Honest statement, not a surrender: with 2-9 months of a 1660 SUPER and 20 B tokens, a 1B model we train
would NOT beat frontier models, because they used thousands of GPUs and trillions of tokens. Saying
otherwise would be inventing a result, which this project forbids.
What IS achievable, and what nobody else has:
1. **Beat models of comparable size and compute** on our own tasks, measured on the same held-out text.
2. **Be the only model of its class that is fully auditable**: every middle unit named, every name
   tested against an independent label, and failed units dropped rather than excused.
3. **Small beats big on a restricted, auditable task** - this is where a 10M model can genuinely beat a
   1B model, and it is a real, publishable, honest claim.

## The rung plan (each rung gated before the next)
| rung | parameters | tokens needed (~20x) | fits how | rough time here | gate to pass |
|---|---|---|---|---|---|
| R0 | 7 M (today) | 140 M | VRAM, as now | hours | honest score above the current 9.84% |
| R1 | 30 M | 600 M | VRAM with SGD | 1-2 days | beats R0 on the same held-out text |
| R2 | 100 M | 2 B | VRAM + checkpointing | 1-2 weeks | beats R1; text meter clears the salad bar |
| R3 | 300 M | 6 B | layer streaming | 3-5 weeks | beats R2; audit still 100% on every named unit |
| R4 | 1 B | 20 B | layer streaming + streamed corpus | 2-9 months | beats R3, and a same-size plain model |

Rules for every rung: honest val only; 3 seeds before any "win"; the audit must stay at 100%; if a rung
cannot be measured honestly it does not count as passed.

## The first three jobs (in order)
1. **Streaming trainer** (`lens/stream_train.py`): SGD, memmap parameters and gradients, one layer
   resident, activation checkpointing. Prove it on a 7M model first: same data, same honest number as
   the in-VRAM trainer within noise. Only then scale.
2. **Stop control and turn structure** (so it can chat at all): train on explicit turn boundaries, and
   require the boundary at generation time. Gate: the stop test must move off 0/10.
3. **Licensed data expansion**: stream Apache-2.0 / CC BY 4.0 text in slices to disk, with the licence
   of every source written into DATA_LICENSES.md before a single byte is used.
