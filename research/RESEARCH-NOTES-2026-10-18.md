# RESEARCH NOTES - read 2026-10-18 (done by hand after the delegation API key expired)

Every line below comes from a search result I actually read; anything I could not verify is marked
UNVERIFIED. Numbers are quoted as the sources state them, not as ours.

## TRACK 1 - Multimodal concept bottlenecks: the idea is published, the enforcement is not
| paper | arXiv | what it does | enforces a drop? |
|---|---|---|---|
| Multimodal CBM (MM-CBM) | 2606.19882 | dual concept bottleneck layers, image + text mapped into ONE shared concept space; inference only in concept space; reports up to +51.26% average accuracy over prior CBMs and within 5% of black-box | no - reports faithfulness, no drop |
| Faithful multimodal CBM (f-CBM) | 2603.13163 | adds a DIFFERENTIABLE LEAKAGE LOSS to train leakage away, plus a KAN prediction head; handles image and text | no - mitigates leakage in training, no drop |
| Partially Shared CBM (PS-CBM) | 2511.22170 | merges redundant concepts by activation patterns; +1.0-7.4% accuracy, +2.0-9.5% CEA on 11 datasets | no - merges, does not drop for failing a name |
| Concept-centric multi-modality | 2412.13847 | modality-agnostic concept space with per-modality projection models; faster convergence | no |
| CONCIL | 2411.17471 | continual learning for CBMs by turning concept/decision layers into linear regression (no gradient updates) | no |
VERDICT: a shared concept wall across modalities is ALREADY PUBLISHED (with pretrained CLIP backbones).
Leakage is an active, named problem with a published differentiable loss. What nobody in this set does:
audit each named concept against an INDEPENDENT label and ENFORCE A DROP of the failing concept.
Our one open piece survives this search. Note also: f-CBM's differentiable leakage loss is a technique we
should test against our own leak meter rather than ignore.

## TRACK 2 - Tiny chat models: our "never stops" result has a known cause and a known fix
- Real-world evidence, two independent projects: a model keeps generating because the TURN-END token was
  never trained. Axolotl issue: for Gemma the terminator `<end_of_turn>` is silently masked (-100) so
  "the only thing never learned - stopping - is never in the loss", and loss curves look healthy.
  Unsloth PR: stopping only on the config eos token makes a 0.8B model run the full 400-token budget and
  re-emit itself; stopping on the turn-end marker too gives a clean 38-token answer.
  => Our measured 0/10 is consistent with this: nothing in our setup trains or enforces a turn end.
- Repetition/looping causes and fixes (Raschka FAQ, verified summary): exposure bias (training sees real
  prefixes, generation sees its own), low-temperature/greedy decoding, overfitting on narrow data,
  duplicated or templated training records, and a missing/wrong end-of-turn token. Fixes: train the
  terminator, enforce a max-new-token bound, moderate temperature and top-k, deduplicate the data.
- Brandon-Tiny 10M (10.7M parameters, single RTX 3090, 600M tokens, 3 phases: pretrain, distill from
  their own 30M teacher, instruction-tune): reports fine-tune val loss 2.40, instruction following 80%,
  and ZERO repetition loops in 20 free generations. The anti-repetition recipe is explicit: label
  smoothing 0.1 + unlikelihood training (Welleck et al. 2020) alpha 0.5 + entropy regularisation
  beta 0.01. They also report their 10M beating their own 30M, with the honest caveat that validation
  sets differed between variants, so it is a pipeline comparison, not a controlled head-to-head.
  => Directly usable: our model is the same size class, and their teacher is their OWN model, which
     respects our no-pretrained-weights rule (self-distillation inside our family).
- Token Loss Dynamic Reweighting (TLDR, arXiv:2003.11963) reduces repetitive generation by reweighting
  the loss of hard tokens.

## TRACK 3 - Beyond next-token prediction: the biggest lever is exactly at our size
- Multi-token prediction (arXiv:2404.19737, Gloeckle et al.): predict the next n tokens with n heads on a
  shared trunk. Reported: 13B models solve 12% more HumanEval and 17% more MBPP than comparable
  next-token models; no train-time or memory overhead; up to 3x faster inference with speculative
  decoding. KEY DETAIL FOR US: "2-token prediction loss leads to a vastly improved formation of
  induction capability for models of size 30M non-embedding parameters and below, with their advantage
  disappearing for sizes of 100M non-embedding parameters and above."
  => We are a 7M-30M model. This is the size band where the benefit is reported to be largest, and it is
     cheap to implement (two output heads). Highest-value single experiment on this list.
- UNVERIFIED in this session (search backend failed twice with a 403 from the search provider):
  forward-forward, predictive coding, JEPA-style latent prediction, diffusion language models. Do not
  quote numbers for these until read properly; they are still open questions, not findings.

## TRACK 4 - Small beating large: the levers are data, not size
- Data quality vs quantity for small models (arXiv:2411.15821, TinyStories study): quality mattered more
  than quantity; 25% duplication slightly IMPROVED accuracy (+0.87%) with +0.52% perplexity, while 100%
  duplication was catastrophic (-40% accuracy). Also quotes Falcon trained on 600B cleaned tokens
  matching GPT-3 zero-shot, beating Pile-based models by up to 3.5%.
- Deduplication (arXiv:2107.06499): dedup makes models emit memorised text ~10x less often and needs
  fewer steps for the same loss.
- D4 (arXiv:2308.12284): careful data SELECTION on top of dedup gave 20% training efficiency and up to
  +2% average accuracy on 16 tasks at 6.7B; and repeating SELECTED data beat adding new random data,
  while repeating RANDOM data performed WORSE than baseline.
  => That last sentence is our own 192-epoch result in someone else's experiment: more passes over the
     same data hurt. Our finding is not a fluke; it is the expected behaviour of random repetition.
- SlimPajama-DC (arXiv:2309.10818): global dedup plus more data DIVERSITY improved training outcomes at
  1.3B, extended to 7B.
- Phi-2 claim (2.7B matching models up to 25x larger via textbook-quality data): UNVERIFIED here - seen
  only in a secondary blog, not read from the paper. Treat as a lead, not a fact.

## WHAT THIS CHANGES IN OUR PLAN (each item is cheap and gated)
1. Train the turn-end marker and enforce it at generation. Predicted outcome, written BEFORE running:
   the stop test moves off 0/10 (expect it to hand the turn back in most generations).
2. Add 2-token prediction heads. Predicted: honest next-word accuracy up at our size, because the
   published advantage is concentrated at <=30M parameters.
3. Add the anti-repetition triple (label smoothing + unlikelihood + entropy regularisation).
   Predicted: fewer repeated n-grams; the text meter moves toward the salad bar, not past it yet.
4. Deduplicate and diversify the corpus properly before any more training: our 192-epoch collapse is the
   documented signature of random repetition.
5. Keep the audit-with-enforced-drop as the differentiator, and test f-CBM's differentiable leakage loss
   against our leak meter instead of ignoring it.
