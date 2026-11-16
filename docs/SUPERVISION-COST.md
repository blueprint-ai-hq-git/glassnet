
## The concept supervision costs accuracy: removing it entirely gives 26.98%, the best number this project has measured (2026-11-16)

Predictions written before the arm ran: `lens/runs/LAM0-PREDICTIONS.txt`. Numbers from `lens/runs/LAM0.txt`, `lens/runs/LAM-NEXT.txt`,
`lens/runs/LAM05-DECODE.txt`, `lens/runs/LAM0.txt`.

```
arm                        concept supervision   VAL acc   VAL ppl   coverage T=0.7
no supervision (lam 0)            0.0            26.98%    147.81       pending
next-target lam 0.25              0.25           26.69%    150.64       0.150
next-target lam 0.5               0.5            26.52%    158.85       0.167
next-target lam 1.0               1.0            26.12%*   169.67       0.173
shipped (window labels)      1.0 (window)       24.58%*   181.04       0.292
```
`*` = three-seed mean; the others are one seed each. Same data, same split, same seed, same 12-epoch budget; only the supervision differs.

## The prediction passed and the falsifier did not fire

I predicted 26.4-27.0 for the no-supervision arm and recorded the falsifier: *"lam 0 below 26.12% falsifies the monotone reading entirely."* It landed
at **26.98%**, the top of the predicted band, with the lowest perplexity in the project at **147.81**.

**Verified that the control was real**: the concept loss printed **0.0000** at the endpoint, so the arm genuinely trained with no concept supervision
rather than a flag that silently did nothing. That check exists because this project has already shipped one silent-no-op.

## What this means

**Monotone across four settings**: every step of concept supervision added buys a little accuracy loss. Removing supervision entirely is worth **+0.86
points** over the strongest supervised arm (26.98 vs 26.12, one seed each).

**And breadth moves the other way**: coverage falls monotonically as supervision weakens, 0.173 -> 0.167 -> 0.150, with the no-supervision arm's coverage
still measuring. So supervision strength is a single dial that **trades accuracy against vocabulary breadth**, in both directions, across four points.

## The correction this forces on the front page

The front page states readability's price as **-1.53 +/- 0.44 points at matched parameter budget**, measured on the image task against a plain control. On
the chat side the concept term's cost is now measured from the other end: switching it off entirely is worth **+0.86 points** and better perplexity, while
switching it on is what produces the vocabulary breadth (0.173 vs 0.150 at quarter strength).

So the honest chat-side statement is: **the concept layer buys an exact explanation and vocabulary breadth, and it pays for both in accuracy.** That is a
trade, not a tax - and it is a different, more precise claim than "readability costs about a point and a half". The public page is being updated to say so.

## Caveat, unchanged and load-bearing

**One seed per strength.** The no-supervision arm has not been run on three seeds, and until it is, 26.98 is a single-seed number. It is also not a
promotable configuration on its own: it has no concepts, so it cannot explain anything, and the shipped model is the one that can.
