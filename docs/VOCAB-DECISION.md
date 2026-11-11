# The vocabulary question, settled by measurement: keep words (2026-11-11)

Three tokenisations, one model shape (d=64 embeddings, one hidden layer, output TIED to the embeddings), identical
steps and batch, judged in bits per character on the same held-out text. Lower bits/char is a better model; fewer
tokens/char is faster to train.

```
tokeniser   entries   params     bits/char   tokens/char   unknown   verdict
words        12,002    800,960     2.1049      0.2561       2.82%    BEST on both counts
hybrid       12,258    817,344     2.1648      0.2670       0.00%    +2.8% bits, +4.3% cost
pieces       16,000  1,056,832     2.2729      0.2751       0.00%    +8.0% bits, +7.4% cost
```

## Both of my hypotheses
- **H1 FAILED.** I predicted the hybrid would beat the word list because 2.82% of positions stop being `<unk>` noise.
  It did not: 2.1648 against 2.1049 bits/char. Removing that noise does not pay for spelling missing words out in
  bytes, which costs +4.3% tokens per character.
- **H2 PASSED.** Pieces lose to words, 2.2729 against 2.1049, matching the 10.9% gap measured earlier at a smaller
  budget (this run says 8.0%).

## What the whole dictionary thread now says
1. The cap is real: `<unk>` is the model's first choice on 13.73% of positions, and on text from a different source
   the word list destroys **23.01%** of tokens (`VOCAB-IS-A-CAP.md`).
2. Both ways of removing the cap cost accuracy-per-compute **in-domain**: the hybrid 2.8%, full pieces 8.0%
   (`TOKENS-VS-PIECES.md`, this file).
3. Therefore: **keep the word list for the work we are doing**, which is in-domain chat. The coverage alternatives are
   insurance against text unlike our training data, and the premium is 2.8% (hybrid) or 8.0% (pieces).

That insurance becomes worth buying the moment we care about being good on text unlike our training data - code,
other languages, names, emoji. It is a decision for when we have a benchmark that measures those, not before.

## Cost of this thread, stated plainly
Five measurements, three of my predictions wrong (2.6 bytes/token, 2.4 bytes/token, hybrid within 0.5%), two bugs in
my own tools (a bits-per-bit metric, a case-folding mistake that manufactured 14% unknowns), and a design I proposed
yesterday and then measured out of existence today. The result is a decision backed by numbers rather than an opinion,
and the cap is documented for whenever out-of-domain text starts to matter.
