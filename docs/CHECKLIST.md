# GLASSNET - FULL CHECKLIST IN PLAIN WORDS
### Saturday 21 November 2026, evening. Written for a reader with no maths background. No formulas.

---

## 1. What this thing is

A small artificial brain built from nothing. No borrowed brain libraries, no ready-made training code. Every piece
is ours and every piece is written so a person can read what it does.

One rule sits above all others: **never say something works unless a test shows it working, and show the real
numbers.** That rule is why this page carries bad news as well as good.

---

## 2. What it can do now - tested on things it has NEVER seen

### Reading and picking things out - WORKS
Four different jobs: copy a digit; copy a digit from far back in the line; find the only real digit hidden among
fakes; find two real digits and say which is bigger.

**Every one of those: 100 correct out of 100.**

### Adding numbers, when allowed to write the working out - WORKS
2,000 brand-new sums it had never seen:

| how wide the numbers are | result on 2,000 new sums |
|---|---|
| 2 digits | 100 out of 100 |
| 3 digits | 100 out of 100 |
| 4 digits | 100 out of 100 |
| 5 digits | 100 out of 100 |
| 8 digits | 100 out of 100 |

For the 8-digit case there are ten thousand trillion possible pairs. It had seen only a small slice of them and
still got every new one right.

### Taking numbers away, including borrowing - WORKS
100 out of 100 on 2,000 new sums, at 3 digits and at 5 digits.

### The part that matters most: you can check its work
For 73 + 4 it writes **7, then 0, then 7** - units 7, nothing to carry, tens 7. That is 77, and you can check it by
hand. The explanation is not a story added afterwards. **The working out IS the answer.**

---

## 3. What it can NOT do - the honest list

**Answer in one jump: fails.** Without the working out it gets about 2 to 7 right out of 100. That is guessing.

**Think harder inside: no effect.** Running the network 1, 4 or 8 times before answering gave 7 out of 100 every
single time.

**Know which part of the input mattered: failed.** A part built for that reached about 77 out of 100 against a bar
of 90. Closed as failed, not quietly tuned until it passed.

**Know when it is right: it already does, for free.** Ranking its own right answers above its wrong ones works at
93 out of 100 with no extra parts. The special part I built for that job scored 85 and lost to the free one. Closed.

**Learn 10 times faster: not with any trick I tried.**

| idea | what happened |
|---|---|
| concentrate on the hard parts | much worse - fell from 100 to 4.5 |
| think several times before answering | no change at all |
| eight times more examples | helped a little, far short of the goal |
| more room inside | tiny gain, wobbling |
| feed the easy examples first | 1.6 to 2.7 times slower |

**Three numbers at once: unknown.** My test was broken, so both results were thrown away rather than keeping the
nice-looking one.

---

## 4. The one big thing we discovered

The machine could not add. Not because it was too small, not for lack of examples, not because it needed longer to
think - all three were tested and eliminated.

It could not add because **there was nowhere to keep the middle number while it worked.** Adding 47 and 38 means 7
and 8 make 15, so write 5 and keep 1 in your head. That "keep 1 in your head" is the middle number, and the machine
had nowhere to put it.

Once it was allowed to **write the middle number down**, one symbol at a time, it could add - perfectly, up to eight
digits - and subtract with borrowing too.

**Give it somewhere to write the working, and it can do the work. And because it writes it down, you can watch it.**

---

## 5. My own mistakes, all of them, and how each was caught

None were hidden. The lesson: **the second test caught every one. Reading my own code never did.**

1. A wrong counting setting made it "learn" to write blank padding - the score looked fantastic, the answers were
   rubbish. Caught by printing one real example.
2. A test that could never fail told me "no leak" when there was a leak. Caught by asking which direction the test
   could possibly detect.
3. A real bug: the network was copying information from one example into another. Caught, fixed, fix checked three ways.
4. Reading answers from the wrong place made 4- and 5-digit results look like 0%. The machine was right, my measuring
   was wrong. Caught because the printed example showed the right answer shifted by two.
5. A runner script that never checked for crashes reported "finished" while all three jobs had crashed.
6. A syntax check that only peeked instead of compiling missed a real error.
7. I started runs even when my own check said failed - twice.
8. A setting given twice made the program refuse to start.
9. The three-number test labelled the carry as yes/no instead of the actual amount, so its working out did not match
   the sum - it claimed 999+999+999 = 997. **Both results withdrawn.**
10. I quoted wrong commit numbers twice. Now a script reads the number back instead of me typing it.

---

## 6. Full checklist of the parts

| part | status | in plain words |
|---|---|---|
| Its own maths engine, every step double-checked | DONE | learning steps verified against a second method |
| Reading input, position, words in | DONE | works |
| Memory of the recent past | CLOSED - not needed | at this size it already sees everything; memory added nothing |
| The thinking block (attention) | DONE, certified three ways | the workhorse |
| "Which part mattered" | FAILED | 77/100 against a bar of 90; closed |
| Named concepts (readable middle layer) | STILL BROKEN | worked on pictures (81.95%); in the word model the named parts cancel out. Last broken layer. |
| "Am I right?" self-check | CLOSED - already free | the machine does it at 93/100 |
| Thinking longer inside (loops) | FAILED | no effect at any depth |
| Writing the working out | WORKS - the big win | 100% on new sums, 2 to 8 digits, add and subtract |
| Three numbers at once | UNKNOWN | test broken, results thrown away, re-running |
| Learning 10x faster | NOT ACHIEVED | five ideas, all measured, none worked |
| Local server / live viewer / standard export | NOT STARTED | owed |
| Computer freezes | NOT SOLVED | cause unknown; one crash dump needs the password |

---

## 7. What happens next, in order

1. Fix the three-number test properly and find out whether it works beyond two numbers. If it does not, the claim
   stays exactly "two numbers" and I will say so.
2. Fix the last broken layer - the named concepts that cancel each other out.
3. Then train everything together once and measure how good it really is, against a guessing baseline, against our
   earlier best, and against the picture model that scored 81.95%.

---

## 8. Where everything is saved right now

- Project folder: everything committed, nothing loose.
- Public copy on GitHub: up to date, checked against the server, personal details removed.
- A second copy on a different, healthier disk: 6,606 files checked one by one and confirmed identical - because the
  project disk is the damaged one that keeps crashing.
- Nothing is running except a small watchdog that writes down what happened if the computer freezes - because last
  time it froze, nothing recorded it, and that was my fault too.

---

## 9. The one-sentence version

**We built a small brain from scratch that can add and subtract multi-digit numbers perfectly - and prove it, step
by step, in a way a human can check - but only when it is allowed to write down its working; and it is not faster at
learning than it was this morning, because every shortcut I tried either made things worse or did nothing.**
