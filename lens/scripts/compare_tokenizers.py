"""compare_tokenizers.py — word-level vs character-level in the SAME units.

Perplexity is NOT comparable across tokenisations: the word model predicts 1 of 12,000 words while
the character model predicts 1 of 432 characters. A word model with perplexity 353 looks worse than
a character model with perplexity 43, but they are not measuring the same thing.

THE COMMON CURRENCY IS BITS PER CHARACTER OF TEXT. That is what actually compares:
    bits per token   = log2(perplexity)
    chars per token  = (characters in corpus) / (tokens in corpus)
    bits per char    = bits per token / chars per token
"""
import math

# measured
WORD_PPL = 353.28          # chat run 1, 3 epochs, 900k windows, d=128, K=9, 1.65M params
WORD_VOCAB = 12000
CHAR_PPL = 43.49           # char smoke, 1 epoch, 8000 windows, d=32, K=35, ~150k params
CHAR_VOCAB = 432

CORPUS_CHARS = 60_949_068
CORPUS_WORDS = 13_318_447
chars_per_word = CORPUS_CHARS / CORPUS_WORDS

print("=" * 92)
print("WORD-LEVEL vs CHARACTER-LEVEL — the same text, measured in the same units")
print("=" * 92)
print(f"  corpus: {CORPUS_CHARS:,} characters / {CORPUS_WORDS:,} words = "
      f"{chars_per_word:.3f} characters per word")
print()

w_bits_tok = math.log2(WORD_PPL)
w_bits_char = w_bits_tok / chars_per_word
c_bits_tok = math.log2(CHAR_PPL)
c_bits_char = c_bits_tok / 1.0            # a character token IS one character

print(f"  {'':<22}{'word-level':>16}{'char-level':>16}")
print(f"  {'vocabulary':<22}{WORD_VOCAB:>16,}{CHAR_VOCAB:>16,}")
print(f"  {'perplexity':<22}{WORD_PPL:>16.2f}{CHAR_PPL:>16.2f}")
print(f"  {'bits per token':<22}{w_bits_tok:>16.3f}{c_bits_tok:>16.3f}")
print(f"  {'chars per token':<22}{chars_per_word:>16.3f}{1.0:>16.3f}")
print(f"  {'BITS PER CHARACTER':<22}{w_bits_char:>16.3f}{c_bits_char:>16.3f}   <- the real comparison")
print()
winner = "word-level" if w_bits_char < c_bits_char else "character-level"
print(f"  LOWER IS BETTER.  ->  {winner} wins on bits per character.")
print()
print("  BUT THE TWO RUNS ARE NOT MATCHED, AND THAT MATTERS:")
print("  " + "-" * 88)
print(f"  {'':<24}{'word run 1':>16}{'char smoke':>16}")
print(f"  {'epochs':<24}{3:>16}{1:>16}")
print(f"  {'training windows':<24}{900000:>16,}{8000:>16,}")
print(f"  {'tokens/characters seen':<24}{900000*8:>16,}{8000*32:>16,}")
print(f"  {'parameters':<24}{1653200:>16,}{'~150000':>16}")
print()
print(f"  the character run saw {900000*8/(8000*32):.0f}x LESS data and has ~11x FEWER parameters,")
print(f"  and it is still a smoke test, not a trained model.")
print()
print("  SO THE HONEST CONCLUSION FROM THIS COMPARISON IS NOT 'one is better'. It is:")
print("    - on the data actually seen, the WORD model is ahead on bits per character")
print("    - the character model is running on 1/238th of the corpus and 1 epoch")
print("    - the real test is a FULL character run, compared in bits per character")
print()
print("  AND THE ARCHITECTURE ARGUMENT IS SEPARATE FROM QUALITY, and it still stands:")
print("  " + "-" * 88)
print(f"  {'':<26}{'word d=128 K=9':>18}{'char d=64 W=32 K=35':>22}")
print(f"  {'embedding share':<26}{'92.9%':>18}{'~19%':>22}")
print(f"  {'concept layer share':<26}{'0.6%':>18}{'~67%':>22}")
print(f"  {'bottleneck ratio':<26}{'114:1':>18}{'29:1':>22}")
print(f"  {'<unk> rate':<26}{'1.92%':>18}{'0.00%':>22}")
print()
print("  At character level the named concept layer stops being 0.6% of the network and becomes")
print("  the part that carries the model -- which is the whole point of the architecture. Whether")
print("  that also produces better TEXT is what the full run measures.")
print("=" * 92)
