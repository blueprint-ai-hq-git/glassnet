# 2026-11-25 - the missing spaces are real, and my explanation for them is WITHDRAWN (falsifier fired)

## The defect is real

  * The raw source text has 801 spaces per 4,000 bytes (198.2 per 1,000 characters in val_text.txt).
  * Every prepared vocabulary checked - 12,000, 12,002, 24,000 and 3,500 entries, six files - contains ZERO tokens with a
    space in them.
  * Decoding the real corpus ids back to text gives `hi,iwanttolearntoplay` instead of `hi, i want to learn to play`.
  * The real corpus is also 2.701% `<unk>` over 500,000 sampled tokens (13,504 tokens), making `<unk>` the FIFTH most common
    token in the corpus.

So: every English model in this project has read unbroken text, and can never emit a properly spaced sentence. As an OUTPUT
problem that stands.

## The explanation is withdrawn

I claimed this also made the language-modelling task harder and therefore explained the poor English results. Measured on
identical text, identical model, identical steps, in bits per character:

    arm      vocab   params     nats/token   bits per char   unk rate   chars/token
    BROKEN    9748   3,079,572      6.1628        1.7714        0.36%       5.02
    FIXED     9749   3,079,861      3.1654        1.8196        0.18%       2.51

Difference **-0.0483 bits per character in favour of the BROKEN arm** - inside the 0.05 bar, so the pre-registered falsifier
fired. The FIXED arm halves its loss per token (6.16 -> 3.17) but has twice as many tokens per character (5.02 -> 2.51), and
per character it is a wash.

**Conclusion: dropping spaces costs nothing in learning efficiency.** Spaces are nearly deterministic given the words, so the
information is not lost, only unrepresented. My explanation is withdrawn and the corpus defect remains a cleanliness and
OUTPUT-FORMAT problem only: generated text cannot contain spaces.

## A void result inside this experiment, caught by its own impossible number

My first version shared one vocabulary between the arms to isolate the variable. That made the space-prefixed arm map every
`Ġword` to `<unk>`, so its sequence degenerated into a regular alternation between `<unk>` and the space marker - perfectly
predictable - and it reported **0.0001 nats/token and 0.0000 bits per character with a 50.00% unknown rate**. That is not a
language model, it is a two-symbol metronome. Fixed to build one vocabulary per arm at matched size, and a hard guard added:
any arm above 5% unknown rate prints INVALID_INSTRUMENT and prints NO VERDICT.

A number that good should always be treated as a bug before it is treated as a result.
