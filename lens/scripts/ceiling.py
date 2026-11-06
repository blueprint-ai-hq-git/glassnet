"""ceiling.py — WHAT THIS MACHINE CAN ACTUALLY TRAIN. Real arithmetic, no vibes.

The owner's target is a model that talks, codes, "and much better than any other model".
That last part has a measurable answer, so this computes it instead of asserting it.

METHOD
------
Training compute is well approximated by:   FLOPs ~= 6 * params * tokens
(2 for the forward pass, 4 for the backward pass.)

The GPU's measured throughput on this machine, from this project's own diagnostic, is 1900.3
GFLOPS with CuPy (lens/tests/diag_cupy.py). Sustained training throughput is lower than a
microbenchmark -- kernels are smaller, there are synchronisation points, and the optimizer adds
work -- so this uses 50% of that as a realistic figure and states the assumption.

Chinchilla-optimal (Hoffmann et al. 2022) is ~20 tokens per parameter: the point where more data
stops being the binding constraint for a given model size. It is the standard yardstick for "is
this model adequately trained", so it is used here.
"""
import math

PEAK_GFLOPS = 1900.3          # measured by this project on this GPU
EFFICIENCY = 0.50             # sustained training throughput as a fraction of the microbenchmark
FLOPS_PER_S = PEAK_GFLOPS * EFFICIENCY * 1e9
CHINCHILLA_TOKENS_PER_PARAM = 20

print("=" * 96)
print("WHAT THIS MACHINE CAN TRAIN — measured, not estimated")
print("=" * 96)
print(f"  GPU throughput measured by this project : {PEAK_GFLOPS:.1f} GFLOPS (CuPy, matmul 512x784x256)")
print(f"  assumed sustained training efficiency  : {EFFICIENCY*100:.0f}%  ->  "
      f"{FLOPS_PER_S/1e9:.1f} GFLOPS")
print(f"  training FLOPs model                   : 6 * params * tokens")
print(f"  Chinchilla-optimal data                : {CHINCHILLA_TOKENS_PER_PARAM} tokens per parameter")
print()

print("  HOW LONG TO TRAIN A MODEL TO CHINCHILLA-OPTIMAL ON THIS GPU")
print(f"  {'params':>12}{'chinchilla tokens':>20}{'FLOPs':>12}{'time':>16}   note")
for label, n in [("1 M", 1e6), ("10 M", 1e7), ("50 M", 5e7), ("100 M", 1e8),
                 ("500 M", 5e8), ("1 B", 1e9), ("7 B", 7e9), ("70 B", 7e10),
                 ("175 B (GPT-3)", 1.75e11), ("1 T", 1e12)]:
    tok = n * CHINCHILLA_TOKENS_PER_PARAM
    flops = 6 * n * tok
    secs = flops / FLOPS_PER_S
    if secs < 3600:
        t = f"{secs/60:.1f} min"
    elif secs < 86400:
        t = f"{secs/3600:.1f} hours"
    elif secs < 86400 * 365:
        t = f"{secs/86400:.1f} days"
    else:
        t = f"{secs/86400/365:.1f} years"
    note = ""
    if n <= 1e7:
        note = "FEASIBLE on this machine"
    elif n <= 5e7:
        note = "feasible with patience"
    elif n <= 1e8:
        note = "weeks of continuous compute -- not with the PC off nightly"
    elif n <= 1e9:
        note = "out of reach"
    else:
        note = "physically out of reach here"
    print(f"  {label:>12}{tok:>20.3g}{flops:>12.3g}{t:>16}   {note}")

print()
print("  THE COMPARISON THAT MATTERS")
print("-" * 96)
# GPT-3 was trained on ~300B tokens with 175B params
gpt3_flops = 6 * 1.75e11 * 3.0e11
print(f"  GPT-3 (175B params, 300B tokens)   ~ {gpt3_flops:.3g} FLOPs")
print(f"  this machine, running FLAT OUT     ~ {FLOPS_PER_S:.3g} FLOPs per second")
years = gpt3_flops / FLOPS_PER_S / 86400 / 365
print(f"  time for this machine to do it     ~ {years:,.0f} years of continuous training")
print(f"  hardware gap                       ~ {gpt3_flops / (FLOPS_PER_S*86400*365*5):,.0f}x "
      f"(at 5 years of continuous training)")

print()
print("  WHAT IS ACHIEVABLE HERE, HONESTLY")
print("-" * 96)
for label, n, hours in [("a real (tiny) chatbot", 1e7, 0.5),
                        ("a better (small) chatbot", 5e7, 2.5)]:
    tok = hours * 3600 * FLOPS_PER_S / (6 * n)
    print(f"  {label:<26} ~{n/1e6:.0f}M params, {tok/1e6:.0f}M tokens in {hours:.1f} GPU-hours")
print()
print("  A 10-50M parameter model trained on 100M-1B tokens is a REAL language model. It will")
print("  form grammatical sentences and answer simple questions. It will NOT be competitive with")
print("  frontier models, and no amount of cleverness on this GPU changes that -- the gap is")
print("  roughly five orders of magnitude in compute, which is a hardware fact, not an effort one.")
print("=" * 96)
