# Security policy

## What is in this repository

Source code, tests, documentation and training logs. **No credentials, no datasets, no model weights.**
Weights are published separately on HuggingFace behind gated access, never in git.

## Reporting a vulnerability

Email **blueprintai.help1@gmail.com** with:
1. What you found and where (file, line, or command)
2. How to reproduce it
3. What an attacker could do with it

Please do not open a public issue for anything that could be exploited before it is fixed. We will confirm
receipt, and if the report is valid we will credit you in the fix commit unless you ask us not to.

## Scope

In scope: the engine, the model code, the local server, the export path, the licence documents.
Out of scope: the accuracy of the model itself (that is a research limitation, documented in
[STATUS.md](STATUS.md), not a vulnerability).

## Known limitations that are not vulnerabilities

- The model is not safe for any production or safety-critical use.
- The local inspector has no authentication and binds to localhost - do not expose it to a network.
- The safety layer is designed but **not implemented** yet.
