# Words versus pieces, judged in bits per character: the word list wins in-domain (2026-11-11)

Same text, same model shape (embeddings d=64, one hidden layer, output TIED to the embedding so neither side gets
free parameters), same 4,000 steps, same batch. Held-out slice is the last 10% of the text.

```
tokeniser   entries   bytes/token   unknown   params     held-out CE      bits per char
words        12,002      3.90        2.11%    800,960    5.6778 nats       2.1003
pieces       16,000      3.64        0.00%  1,056,832    5.9497 nats       2.3581
```

**The word list wins by 10.9% in bits per character**, and it did so while the pieces model had 32% MORE parameters
(1.06M against 0.80M), so the handicap was against words.

## The metric bug I have to confess first
My tool computed bits/token divided by bytes/token and then divided by 8, which reports **bits per bit** and
understates every number by a factor of eight. The printed 0.2622 and 0.2952 are therefore wrong as absolute numbers.
The comparison between the two arms is unaffected because the same wrong factor was applied to both, but I recomputed
the correct figures above from the stored losses and that is what is published. The tool is fixed so this cannot
recur.

## What this means, and it argues against my own proposal
I have spent the day arguing that the word vocabulary is a cap and that byte-level pieces are the fix. The cap is real
(`VOCAB-IS-A-CAP.md`: `<unk>` is the model's first choice 13.73% of the time). But **switching the whole model to pieces
would cost about 12% efficiency on in-domain text**, and this test is the first measurement that prices that trade
instead of assuming it.

The reason is visible in the same table: the pieces tokeniser is only mildly more granular on in-domain English
(3.64 against 3.90 bytes/token) while forcing the model to choose among 16,000 candidates instead of 12,002. Coverage
it has; efficiency it does not.

## What the measurements now say to do instead
A **hybrid**: keep the 12,002 word tokens as the main alphabet, and add the 256 byte tokens ONLY as a fallback for words
that are not in the list. Known words keep their compression, unknown words stop being destroyed, and the vocabulary
grows by 0.13M parameters instead of being replaced. That is the version of the unlimited dictionary that does not pay
the 12% tax.

Prediction for the hybrid, written before building it: identical bits per character to the word list on in-domain text
(within 0.5%), and on out-of-domain text it beats the word list by at least 10% because the word list destroys 23% of
out-of-domain tokens.

## Caveats, stated rather than buried
- 4 MB of text, 4,000 steps, an 0.8-1.1M parameter model. This is a tokeniser comparison at small scale, not a verdict
  on the d=512 model.
- The comparison holds compute and shape equal, not bytes seen: the pieces side read 16.7 MB against the words side's
  18.0 MB.
