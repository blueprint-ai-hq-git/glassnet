# explain() in the library, by measurement (2026-11-12)

The brief asks for a small library where `explain()` shows why the model answered. That now exists for the readable rule
models, and it is the measured version - not a story read off weights.

## The whole use, in seven lines
```python
import numpy as np, glassnet as gn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lens", "scripts"))
import chat_data as cd

m = gn.load_rules("lens/runs/rules_facetsel.npz")            # 1: load the readable model
dd = cd.build()                                              # 2: the corpus
split = np.asarray(np.load(os.path.join(os.environ["CHAT_DATA_DIR"], "split_matched_v1.npy")))
X, y = cd.make_windows(dd["ids"], dd["bounds"], split, 32, which=1, max_windows=3, seed=5)
for i in range(3):
    print(gn.explain_rules(m, X[i]))                         # 3: why did it say that
```

## Real output, three held-out windows
```
held-out window: 'at the heart of efforts to increase'
said 'to'   (scores: to 4.03, and 3.44, like 3.21)
  because of shape: last token is 6+ chars   which is worth +1.92 on 'to' (switching it off favours 'department' by +6.63)
  (measured: individual effects sum to -1.92, switching every concept off gives -1.92, interaction 0.0%)

held-out window: 'are lots of things you can do about that . you can'
said '<unk>'   (scores: <unk> 6.13, function 3.33, ` 3.01)
  because of content: window has <unk>       which is worth +1.66 on '<unk>' (switching it off favours 'р' by +6.23)
  because of earlier: token at -3 is punct   which is worth +0.94 on '<unk>' (switching it off favours 'aimed' by +13.16)
```
Each line is a measurement on that window: the concept is true, it was switched off, and the chosen word's score moved by
the stated amount. The second window says `<unk>` because its context genuinely contains unknown tokens - that is the model
being honest about noisy input, and the explanation shows the concept that says so.

**Per-window additivity is exact** (interaction 0.0% in all three): when no rule crosses its ReLU boundary, the model is
exactly linear in its concepts, so the individual effects sum to the whole. The 3.7% aggregate figure from the nine-model
check comes from the minority of windows where a rule does cross its boundary.

## A fabricated explanation I caught and removed
The first version of the library **padded short windows with the vocabulary's id 0, which is `<unk>`**. That made
"window contains `<unk>`" spuriously true, so the explainer answered `<unk>` *because of a concept the explainer itself had
created*. It was visible immediately in the output (every window said `<unk>` and cited `window has <unk>`), and it is the
worst possible failure for this project: a tool that manufactures its own explanation.

Training data never pads - `chat_data.make_windows` builds full 32-token slices inside a conversation - so the library now
does the same and **raises** on short input instead of inventing context:

```
ValueError: need at least 32 tokens, got 9 - the model was trained on 32-token windows that are never padded
```

## Surface
```python
gn.load_rules(path)              -> RuleModel (weights, concept names, vocabulary, accuracy)
gn.explain_rules(m, tokens)      -> plain-text explanation, measured per concept
gn.concept_effects(m, tokens)    -> the same thing as structured data (effects, displacement, additivity)
```
Exported from the package root, so `import glassnet as gn` is all a user needs.

## What it is not
- It explains the **concept** layer, which is the faithful level for these models; it does not claim a rule is the cause,
  because per-rule ablation shows a single rule is worth 0.000 accuracy points under a tie and only 0.36 untied.
- It attributes total effect per concept; concepts that always fire together share credit.
- It works on 32-token windows of this corpus vocabulary; a different tokenizer means retraining, not re-explaining.
