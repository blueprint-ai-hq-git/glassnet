# 2026-11-26 - the compute path moves NOTHING end-to-end: arithmetic was not the constraint after all

Source: tools/eval_compute_path.py, lens/runs/COMPUTE-PATH-EVAL.json. Same checkpoint as the 2.4% run
(`math_explain_v2_ep3.npz`), same prompts, same 120 greedy tokens. ONE thing changed: when the text so far ends in
"<number> <op> <number> =", the next token is COMPUTED by exact integer arithmetic instead of predicted.

    condition                                   MATH-500        MetaMathQA held-out
    arithmetic PREDICTED (previous run)         12/500 = 2.4%    6/300 = 2.0%
    arithmetic COMPUTED (this run)              12/500 = 2.4%    6/300 = 2.0%

The compute route fired 40 times across the 500 MATH-500 generations and 111 times across the 300 in-domain generations, so it was
not idle - and the number of correct final answers did not move by a single item.

## My prediction was wrong a second time, in one evening

I predicted in-domain would rise from 2.0% to at least 6% and MATH-500 from 2.4% to at least 3.0%, with the falsifier at "both stay
within 0.5 points". Both stayed EXACTLY where they were. **Falsifier FIRED.**

## What the two experiments together prove

Yesterday's probe established the model gets 4.6% of the sums it writes correct. This experiment establishes that repairing those
sums changes no final answer. Put together:

* the arithmetic is broken, AND
* the arithmetic is not what stands between this model and a right answer.

The binding constraint is the PLAN: choosing which quantities to combine, in what order, and recognising which number is the answer
at all. That is a harder problem than arithmetic, and it is not reachable by making the sums right.

## Caveat stated plainly

The compute route only fires on an explicit "a op b =" pattern, which is 40 of 500 MATH-500 generations - so most of each solution
is still model-written. This experiment therefore tests exactly what it claims ("does repairing the visible arithmetic change the
answer"), not "would a model that computes everything be better". A model whose EVERY arithmetic step was computed is a different
experiment and has not been run.

## What this rules out

Spending the next GPU budget on arithmetic drills, a bigger calculator surface, or more epochs of the same corpus. All three target
a constraint this experiment just showed is not binding. The next lever has to be step-level structure: supervision that teaches
which quantity to pick at each step, which is a data-shape problem before it is a model-size problem.
