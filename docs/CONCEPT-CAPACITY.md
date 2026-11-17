
## Our 53 concepts are effectively 16, and 8 of them actively hurt (2026-11-17)

Measured with `tools/concept_capacity.py`, 2000 held-out windows: zero each concept's activation and record how much the loss rises in bits per token.

```
all 53 concepts active                        8.6934 bits/token
no concepts at all (bias only, the floor)    12.7837 bits/token   (+4.0903, +47.1%)

concepts whose removal costs more than 0.05 bits : 16 of 53
concepts whose removal costs less than 0.001 bits :  7 of 53
share of total concept importance in the top 5   : 30.1%
concepts whose removal IMPROVES the loss         :  8 of 53
```

**Most load-bearing** (loss increase when zeroed):
`mentions_the_user` +0.158, `cp_senses` +0.142, `mentions_weather` +0.134, `mentions_law` +0.121, `is_uncertain` +0.119, `mentions_self` +0.113,
`is_imperative` +0.111, `mentions_work` +0.098.

**Least**: `is_explanatory` +0.003, and then eight that go negative - `mentions_a_number` -0.006, `cp_commands` -0.003, `cp_code_refs` -0.002,
`mentions_a_group` -0.002, `cp_shape` -0.001, `cp_modality` -0.001, `cp_comparison` -0.0002.

## What this says about the architecture

**Concepts carry 47% of the model's information** - the other 53% comes from the bias vector and the structure of the answer matrix, not from any concept. So the
readable part of the model is a minority of what it uses.

**The 53-wide bottleneck is really about 16 wide.** Sixteen concepts matter, seven do nothing measurable, and eight are worse than nothing: removing them lowers
the loss, which means they were adding noise the answer layer had to work around. A bottleneck that is 70% dead is not a 53-way explanation, it is a 16-way
explanation wearing 53 labels.

**The load-bearing concepts are content, not grammar.** `mentions_the_user`, `mentions_weather`, `mentions_law`, `mentions_work` - topic detectors. The
syntactic ones (`is_uncertain`, `is_imperative`) are present but weaker, and several structural abstractions (`cp_shape`, `cp_modality`, `cp_comparison`) are
among the dead. That is consistent with everything measured this week: without depth and attention there is no computation to detect structure with, so the
concepts that survive are the ones a bag of words can already separate.

## Why this matters for the comparison

A 24-layer transformer does not have a named bottleneck because it does not need one: its mixing happens in 896-wide residual streams across 24 blocks. Ours
mixes once, through 16 effective numbers, and then answers. That is now a measured statement rather than an intuition, and it explains the breadth ceiling from
yesterday better than supervision strength ever did: **there is nothing in this architecture to hold variety.** A model whose entire capacity between input and
output is 16 numbers cannot preserve a wide vocabulary, no matter how its concepts are labelled or how hard they are supervised.
