# Models

Every number below is copied from the model file's own metadata together with the run that produced it. Where a claim could not be
verified from a saved checkpoint, it is not listed. Nothing on this page is an estimate.

## math1 - two-digit addition, with written working

**Download:** https://huggingface.co/blueprint-ai-hq-main/glassnet-math-tiny

| | |
|---|---|
| parameters | 9,084 |
| held-out accuracy | 100.0% (300 of 300 problems never seen in training) |
| training steps | 8,000 |
| vocabulary | 23 tokens |
| container | safetensors 37,728 B, GGUF 37,529 B, base64 JSON 49,454 B |

It writes its working as output tokens before the answer, the way a person writes a column - digit and carry pairs, least
significant first. Because the working is produced as output, a wrong reason forces a wrong answer; the explanation is not
generated after the fact.

**What it is not:** it cannot read questions, it is not a language model, and it does one operation on two digits. A four-digit
model measured 100.0% held-out on 500 fresh problems (420 of them requiring carries), but that run saved no checkpoint, so it is
not offered here. That failure is why every export now verifies its weights before it is published.

## Reproduce a number

```bash
python3 lens/math_app.py 8000
```

It checks 400 generated traces against real arithmetic first and refuses to train if any disagree. It re-scores 300 fresh
problems before exporting and refuses to export below the bar. The score you see is the score in the file.

## Verified by scrambling, not by assertion

A tiny model could in principle be a lookup table. Three checks rule that out for the arithmetic models: intact weights score
100.0%, scrambling the output weights drops it to 0.7%, and scrambling the first attention layer drops it to 0.0%. Nine thousand
numbers cannot store ten thousand sums.
