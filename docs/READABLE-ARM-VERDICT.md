# Readable-only arm: readability costs 5.22 pp, and the loop survives without the copy path (2026-11-10)

One change against the 25.52% baseline: `--direct` dropped, so the 36 named concepts are the only route to an answer.
d512, window 32, 3 epochs, batch 1024, matched split, seed 0. 7,178,958 parameters (the copy path's 8.39M matrix is
gone), 8.91 tokens per parameter, 98-112 s per epoch.

```
epoch     baseline (copy path)     readable only        gap
1              19.69%                15.41%          -4.28 pp
2              24.26%                18.70%          -5.56 pp
3              25.52%                20.30%          -5.22 pp
final VAL ppl   99.92                247.75
concepts        35/36                28/36
```

## Result 1: readability is affordable, at about 5 pp
Yesterday's framing was that giving the concepts the decision costs 18.50-19.17 pp. That was the price of retrofitting
concepts onto a copy-trained model. A model that never had a copy path pays **5.22 pp** - less than a third as much.
So the trade is real but far cheaper than the retrofit number implied, and the readable path does carry language: 20.30%
assistant-by-position with nothing but 36 numbers between the input and 12,002 possible answers.

One honest blemish: only **28 of 36** concepts pass the label-honesty gate here, against 35/36 for the copy-path model.
The concepts that decide the answer are less label-honest than the ones that only watched.

## Result 2: the loop is not the copy path, and my earlier conclusion was wrong
This morning's symbol forensics showed the concept path holding symbol probability at 0.007-0.008 while the copy path
climbed to 0.965, and I concluded the loop came from the copy path. The concept-only model has no copy path, and:

```
                        copy-path model    readable only
symbol share                 8.7%             27.8%
longest symbol run            3                23
distinct 4-grams            54.0%             19.8%
repeated-adjacent tokens     0.0%             26.4%
copy rate st 17-24          87-90%           100.0%
```

```
hello , how are you ?  -> the number of the number of the number of the number of the number of the
the capital of france is -> the ` ` ` ` ` ` ` ` ` ` ` ` ` ` ` `
```

**Removing the copy path made the loop worse, not better.** So the copy path was the carrier of the loop in that one
model, not its cause. The loop forms in a small next-token model under greedy decoding whichever route the answer
takes, and the honest statement is now: it is a property of the decoding loop and the sharpness of a small model's
distribution, not of a particular path through the network.

## What I am correcting publicly
The claim "the loop comes from the copy path" appeared in STATUS and in `SYMBOL-FORENSICS.md` today. It is true of the
model that was measured and false as a general explanation. What survives is the measurement, and the practical
conclusion is unchanged: the only thing that ever removed the looping was repetition control at decoding time.

## Where this leaves the readability question
The two variables are now separated and both measured on the fair split:
- readable-only, copy path gone: **20.30%** and truthful `explain()` by construction
- copy-path model: **25.52%** and an `explain()` that describes 4-8% of its decisions

That is a real choice with a real price, and it is the first time today the price has been small enough to be worth
paying.
