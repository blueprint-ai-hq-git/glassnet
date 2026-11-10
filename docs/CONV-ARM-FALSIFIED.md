# Prediction falsified: the conversational corpus loops worse (2026-11-10)

`SOURCE-REPETITION.md` recorded a prediction before the arm ran: *if the repetition is mostly a corpus property,
training on the conversational portion only should show lower in-window copying and less looping than the
merged-corpus arms at similar accuracy; if it does not, my account of the cause is wrong again.*

It did not.

```
                            merged corpus   conversational only
assistant accuracy             25.52%            17.35%
in-window copying              74.3%             69.1%
previous-token echo             0.0%             39.6%
copy mass on visible tokens    30.8%             46.2%
symbol share in generation      8.7%             58.3%
longest symbol run               3                23
distinct 4-grams               54.0%             60.7%
```

The conversational arm copies **more** (46.2% mass against 30.8%), echoes the previous token **39.6%** of the time
where the merged model never does (0.0%), and its generated text is dominated by quote marks:

```
what is your name ?      -> human was a " \" " \" " \" " \" " \" " \" " \"
the capital of france is -> a " \" " \" " \" " \" " \" " \" " \" " \" " \"
```

## What this means
1. **My account was wrong.** I said the looping is mostly the math/code portion inviting repetition. The
   conversational data, which repeats *less* by my own measurement (38.32% in-window against 51.75%), produces a
   model that loops *more*. Corpus repetition rate is not what drives it.
2. **The pattern in the text is.** The human/assistant dialogue corpora are full of turn markers and quotation
   marks - `human`, `assistant`, quotes, backticks - and those are cheap, highly predictable, useless tokens. The
   model finds them, and the copy path makes them self-reinforcing. That is why the conversational arm emits 58.3%
   symbols with a 23-token run, and why its accuracy is worse: most of its capacity went into boilerplate.
3. **The merged model is better behaved than the "clean" one.** 8.7% symbols and a 3-token run. So "use the
   conversational data" was not an improvement in any measured respect.

## What the next look should be, and the prediction for it
The variable to measure is the share of degenerate boilerplate per source - turn markers, lone quotes, backticks,
repeated punctuation - not the repetition rate. Prediction, written here before measuring: the conversational
corpora will have far more boilerplate tokens per conversation than the math/code corpora, and a model trained on
the same conversational data with the boilerplate stripped will emit fewer symbols and shorter runs than this arm
at equal or better accuracy.

If that also fails, the remaining explanation is the architecture's feedback path itself, and the next place to
look is inside the copy path's inputs during generation rather than at any property of the corpus.
