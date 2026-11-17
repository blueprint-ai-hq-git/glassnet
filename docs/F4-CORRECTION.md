
## CORRECTION: F4 was blind. The trained readout works - 16.78% against noise's 0.0000% (2026-11-17)

**The previous entry published a wrong conclusion and it is withdrawn here.**

It said: *"the trained answer layer is indistinguishable from noise at separating inputs"*, based on F4: trained KL **1.1702** against a noise readout's **1.1660**, ratio
**1.0036x**.

## The check that broke it

`tools/f4_instrument_check.py` asks the direct question F4 never asked - does the readout **predict** the right next token, over the full 12,002-word vocabulary?

```
next-token accuracy over 5,120 held-out tokens:
  TRAINED readout : 16.7773%
  NOISE   readout :  0.0000%
  chance          :  0.0083%

trained/noise ratio: 859x
```

**The trained readout is doing real work, by a factor of 859.** It is not noise, and F4's number did not measure what its name claimed.

## Why F4 was blind

F4 computes **KL across six one-word prompts**. That measures how *differently* a projection responds to six inputs. **A random projection responds differently to different inputs
by construction** - that is what random projections do. So the metric saturates at a high value for both arms, and comparing 1.1702 to 1.1660 tells you the noise projection
has *similar response variety*, not that the trained one lacks predictive competence.

The two quantities answer different questions:
- **KL across six prompts**: how much the response *varies* with input. Noise passes this trivially.
- **Next-token accuracy over the vocabulary**: whether the response is *right*. Noise scores 0.

F4 measured the first and drew a conclusion about the second. It is the classic defect this project keeps finding - **a proxy metric standing in for the thing itself** - and this
time the proxy was mine, written into a pre-registration as "the sharpest" gate.

## What survives from run 9's scorecard, and what does not

**Survives (measured, unaffected):**
- F3 held-out accuracy **16.93%** against a 5.34% bar - PASS, and now corroborated by the independent 16.78% above on a different token slice.
- F2 state cosine **0.1736** - PASS.
- G1 readout gain bounds **0.978-1.119**, bounded.
- The readout's competence: **16.78% vs 0.0000% for noise**, which is the strongest positive result run 9 produced.

**Withdrawn:**
- *"F4 FAIL: the trained readout is not input-sensitive / is indistinguishable from noise."* **Wrong.** The readout is highly input-sensitive and predictive; F4's metric
  could not see it.
- The follow-on inference that *"the failure sits between the state and the answer layer"* is **unsupported** and withdrawn with it.

**Open, and now flagged rather than assumed:**
- G2 reported mean predictive entropy **8.835 nats = 94.1% of uniform**. A readout that predicts the next token correctly 16.9% of the time should have substantially lower entropy
  on real text than 94% of uniform. That combination has not been reconciled, and it suggests G2 - like F4 - was computed on the six-prompt probe rather than on corpus-like data.
  **This needs measuring before G2 is quoted as a property of the model.**

## The rule this reinforces

Two gates, same run, and the one called "the sharpest" was the wrong instrument while the plain accuracy number was right all along. **A pre-registered gate is only as good as its
proxy.** The correction is recorded here rather than quietly edited, and the withdrawn sentence is quoted above so a reader can see exactly what changed.
