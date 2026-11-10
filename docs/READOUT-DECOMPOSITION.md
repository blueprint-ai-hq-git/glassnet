# Where the input dies: the logit decomposition (scan8_ep1000)

logits = state term (h @ Wa + ba)  +  direct term (relu(emb[last] @ Wd) @ emb.T)

The obvious hypothesis was that the direct term swamps the state, so the logits are basically "the last
token's own embedding" and the recurrent state is ignored. Measured with
`tools/readout_decomposition.py` on `scan8_ep1000.npz`:

```
prompt     state sd   direct sd   direct/state
hello        0.1147      0.0775        0.7
quantum      0.1225      0.0668        0.5
sad          0.1135      0.0818        0.7
france       0.1241      0.0729        0.6
computer     0.1110      0.0847        0.8
sea          0.1112      0.0650        0.6

KL across six prompts, full logits                      : 0.0151
KL when ONLY the state may vary (direct frozen)         : 0.0108
KL when ONLY the direct term may vary (state frozen)    : 0.0050
```

## The hypothesis was wrong
The state term is **larger** than the direct term (standard deviation 0.111-0.124 against 0.065-0.085), and it
carries most of the variation across prompts: freezing the direct term leaves **0.0108** of the 0.0151 KL,
while freezing the state leaves **0.0050**. So the input is not dying in the direct path, and the direct path
is not drowning the state.

## What the numbers do say
The variation that does exist is small in absolute terms: about 0.1 in logit space across six very different
prompts, against a 12,002-way softmax whose normalisation washes out differences of that size. The state
cosine also rose from 0.2261 at step 750 to **0.3139** at step 1000, still under the 0.5 bar but drifting.

So the failure is one of SCALE, not attribution: the readout receives states whose differences produce logit
differences too small to move the distribution. That is precisely what run 9's readout normalisation changes -
it divides the state by its own RMS and applies a per-unit learned gain, which is the only place in the model
that can amplify signal before the softmax.

## Consequence
Run 9's F4 (trained readout must beat a noise readout by 2x) is now the decisive test, with a mechanism
behind it rather than a guess. If F4 still fails with a learned gain available, then the gain is not being
learned for a reason worth measuring next: the gradient reaching it may be vanishing compared with the
gradient reaching the direct term.
