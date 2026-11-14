# The chat concepts are not redundant: the defect that cost the image model does not exist on the chat side

Date: 2026-11-14. Tool `tools/chat_concept_overlap.py`, output `lens/runs/CHAT-CONCEPT-OVERLAP.txt`. Definitions read from
`lens/scripts/chat_concepts.py` (the module `train_chat.py` imports) and `lens/scripts/chat_concepts_wide.py`.

## Why this check existed

On the image side the central defect of this session was **redundant concepts**: the shipped 14 labels had 8 pairs correlating
above 0.80, so no pair of concept neurons could both mean their own name. De-duplicating them took the model from **2 of 14
mislabeled to 0 of 14**. Before spending GPU time rebuilding the chat concept layer the same way, the obvious question was whether
the chat side has the same defect.

Chat concepts are **token lists** (`Concept(name, ids)` where `ids` are exact token ids), so redundancy there manifests as
**overlapping vocabulary**: if two concepts share most of their words, a window containing one nearly always contains the other.

## Result: no overlap at all

```
chat_concepts.py (the 12 concepts train_chat.py uses)
  no two concepts share a single token - the label set is disjoint by construction
  pairs with jaccard > 0.50: 0     > 0.20: 0
  tokens per concept: min 1, median 5, max 11

chat_concepts_wide.py (43 concepts)
  highest overlap in the whole set: mentions_education ~ mentions_science at jaccard 0.08 (the word "study")
  next: is_an_agreement ~ mentions_law 0.08 ("right"), is_a_farewell ~ mentions_time 0.08 ("later")
  pairs with jaccard > 0.50: 0     > 0.20: 0
  8 of 43 concepts share any word with any other
```

**The image defect does not exist here.** The 12-concept set is perfectly disjoint, and even the 43-concept wide set never exceeds
0.08 - one shared word in twelve.

## What this does and does not establish

It establishes that the chat concept layer is **not** suffering from label redundancy, so the de-duplication work that moved the
image model from 2 mislabeled to 0 is **not needed** on the chat side. That is one hypothesis about the chat model's poor concept
honesty eliminated cheaply, before any GPU time was spent.

It does **not** establish that the chat labels are uncorrelated in practice. Two concepts can share no words yet co-occur
constantly - `is_a_greeting` and `is_a_farewell` through conversation position, `mentions_food` and `is_a_request` through topic.
The next check is therefore the honest one: compute the 12 concept labels over the prepared chat data and measure their pairwise
correlation, the same measurement that found the image defect. **That needs the 1.1 GB token-id file loaded, so it waits for a
moment when the machine is free rather than being run while a game is up.**

## Prediction for that next check, written now

The 12 concepts are semantically distinct categories (question, greeting, farewell, apology, thanks, you, I, feeling, food, number,
request, health), so I expect **no pair above 0.50 correlation** and at most two pairs above 0.30 - chiefly the conversational
position pair (greeting/farewell) and possibly food/request. If a pair does exceed 0.50, the chat concept set inherits the image
defect through co-occurrence rather than vocabulary, and de-duplication becomes necessary after all.
