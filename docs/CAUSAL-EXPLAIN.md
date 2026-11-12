# Causal test: the concepts really drive the answer, but the rules that carry them read as nonsense (2026-11-12)

Every concept number reported before today was a **correlation**: a neuron predicts its own label better than another
label. That does not show the answer layer *uses* the concept, which is the entire basis of explain(). So
`tools/intervention.py` forces one concept neuron to 0 or 1, recomputes the rules and the answer logits with the saved
weights, and measures whether the model's answer changes.

Model tested: the 12-concept, 96-rule, 2,692,780-parameter warmup model, held-out accuracy 15.28%, concept honesty 84.0%.

## The concepts are causally used
```
concept                          honesty   flip if ON   flip if OFF   mean KL(ON)
last token is '?'                  99.5%       84.8%         1.2%        1.1821
last token is a digit              96.1%       71.8%         9.2%        1.1596
window contains <|im_start|>       63.0%       63.9%        32.8%        1.0375
window contains '{'                73.6%       43.6%        13.9%        0.7638
window contains '`'                92.4%       41.1%         6.4%        0.6626
window contains '$'                79.5%       40.0%        21.8%        0.7451
last token is punctuation          93.7%       38.9%        11.4%        0.6588
last token is '.'                  99.3%       37.1%         2.4%        0.5552
window contains '\'                90.9%       27.1%         5.1%        0.6760
window contains '<unk>'            60.5%       26.4%        52.7%        0.3209
window contains <unk>              67.6%       23.2%        45.9%        0.1836
window contains '}'                93.1%       20.3%         3.1%        0.1579

mean honesty 84.1%    mean flip-if-ON 43.2%
correlation honesty vs causal influence: pearson +0.253   (n=12)
```

Forcing a single concept changes the model's answer on **43.2% of held-out windows on average**, and **84.8%** for the
question-mark concept. The readable path is not decoration.

## But honesty and usefulness are only weakly related
`window contains '}'` is **93.1% honest** and forcing it flips only **20.3%** of answers. `last token is '?'` is 99.5%
honest and flips **84.8%**. Pearson between the two is **+0.253** over 12 concepts: a concept can be perfectly honest and
nearly ignored. So "honest" and "load-bearing" are two different claims and both have to be measured.

## The bad news, and it is the reason to keep looking
The rules that carry the answer weight are semantically arbitrary. Three real held-out windows, printed with the numbers
the forward pass actually produced:

```
window: ... 'what' 'your' 'question' 'is' '.'
  answer 'the' (true '<|im_end|>')   scores: the 4.13, do 3.80, to 3.57
  rule #21 fires 0.528 -> pushes 'the' +1.44  (reads: last token is '.' AND last token is punctuation)
  rule #9  fires 0.550 -> pushes 'the' +1.30  (reads: window contains <|im_start|> AND last token is a digit)
  rule #4  fires 0.465 -> pushes 'the' +0.59  (reads: window contains <unk> AND window contains '}')

window: ... 'in' 'the' 'fridge' '?' '<|im_start|>' 'assistant' 'i' 'don'
  answer ''' (true '’')              scores: ' 5.57, ’ 4.74, <|im_start 3.28
  rule #1  fires 1.157 -> pushes ''' +1.70  (reads: last token is '?' AND window contains <unk>)

window: '<|im_start|>' 'user' 'yes' '.' '<|im_start|>' 'assistant' 'here' 'is'
  answer 'a' (true 'what')           scores: a 4.72, the 4.54, much 3.35
  rule #9  fires 0.701 -> pushes 'a' +1.10  (reads: window contains <|im_start|> AND last token is a digit)
  rule #4  fires 0.465 -> pushes 'a' +0.83  (reads: window contains <unk> AND window contains '}')
```

Every sentence above is mechanically true - those are the weights, those are the activations, that is the contribution to
the answer. And every one of them is a conjunction a human cannot use: "the window contains an unknown token AND a closing
brace" does not explain why the next word is "a". Rule #4 alone, a rule with no coherent meaning, is pushing three
different answers by +0.6 to +1.6.

**So the architecture is readable in form and not yet meaningful in content.** Accuracy cannot see this at all: a model
full of nonsense conjunctions scores exactly the same as one with sensible ones. That makes rule semantics the next real
gate, and it is measurable: a rule should name two concepts that plausibly co-occur for a reason, and the gate can require
that the concepts in a high-weight rule are not statistically independent in the data.

## Side fix: every run now keeps its own weights
`train_rules.py` overwrote `lens/runs/rules_v1.npz` on every run, so the earlier configurations' weights were lost (their
logs survive). It now writes `rules_<TAG>.npz` when `TAG` is set, and prints the path it wrote.
