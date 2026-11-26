# 2026-11-26 - the maths line hits the same ceiling in six configurations: 2-3% end-to-end answers

Tonight I ran six configurations of the maths solver. Every one is scored with the same scorer (`tools/eval_math500.py`) on the same
fixed sets: MATH-500 (500 problems, test only) and MetaMathQA held-out by original question (300 problems).

    arm                          what changed                        MATH-500     in-domain
    math_explain_v2              baseline, window 32                  12/500 2.4%   6/300 2.0%
    math_w128                    window 128 (also bigger model)        2/500 0.4%   1/300 0.3%
    math_budget                  10 epochs, lr 0.4 -> 0.12             7/500 1.4%   7/300 2.3%
    math_asst                    assistant-only loss                   11/500 2.2%  9/300 3.0%
    math_memorise (200 solutions) 100.00% train acc, loss 0.0053      4/200 2.0%   (same 200 it fitted)
    compute path                 arithmetic computed, not predicted    12/500 2.4%   6/300 2.0%

Every arm sits between 1.4% and 3.0%, and the spread from 2 to 12 correct answers out of 500 is what noise looks like. Nothing
moved the end-to-end number in a way that survives its own seed.

## The six hypotheses, and what killed each

    capacity / architecture   RULED OUT   100.00% train accuracy on 200 solutions, loss 0.0053
    arithmetic                RULED OUT   computing every visible sum changed no final answer
    window 32 too small       FALSIFIED   128 was worse; confounded by a fixed lr, my error
    vocabulary too small      RULED OUT   99.45% of number occurrences writable, 1.5% of numeric answers not
    training budget/schedule  FALSIFIED   7x the training, decaying lr: 2.0% -> 2.3%
    loss on the whole turn    FALSIFIED   assistant-only masking (81.8% of targets): 2.0% -> 3.0%

## What the evidence now says

The model has learned the FORMAT of a worked solution and none of its content. It writes LaTeX, it writes "the answer is :", and its
arithmetic is wrong 95.4% of the time. Making the sums right does not help, because the parts it chooses to combine are wrong.
Showing it more of the question does not help. Giving it the loss it should have had from the start does not help.

The ceiling is not a training defect. It is what next-token prediction over 149,936 solutions does at 15 million parameters on this
machine: it learns to sound like a solution.

## What this project can already do, measured, and is not using

    single-operation arithmetic     100.0% on 300 unseen problems, 9,084 parameters, shipped
    four-digit addition with carries 100.0% on 500 problems
    composition of two operations    69.7-74.6%

So the parts that work are proven and they are not connected to the part that fails. A solver built by COMPOSING the proven parts -
parse the problem into operations, run the arithmetic that is already 100% accurate, compose the results - would not depend on the
15M model doing something it has now failed to do six times.

That is a change of approach, not another training configuration, and it is the owner's call. It is the one question this evidence forces.
