# Privacy audit

Run 2026-11-06 against every tracked file. This is what we checked and what we found.

| Check | Result |
|---|---|
| API tokens (`github_pat_`, `ghp_`, `hf_`) | **none present** |
| Passwords or passphrases | **none present** |
| Private IP addresses | **none** - only `127.0.0.1` and `0.0.0.0`, which are loopback/any-interface |
| Email addresses | only the project contact and a `you@example.com` placeholder |
| Phone numbers | none; matches were decimal numbers inside training logs |
| Personal home directory in documentation | **scrubbed** - replaced with `~` and `<project root>` |
| Personal paths in development scripts | **still present in ~1,200 shell/Python scripts** (see below) |

## The one open item

Many experiment drivers were written during development against one machine and contain absolute paths
such as `/run/media/<user>/...`. They are functional locally and are **not** portable. This is a real
quality defect, and fixing it properly means replacing hardcoded paths with a `PROJECT_ROOT` environment
variable and retesting every driver - a mechanical but broad change that should land as one reviewed commit,
not smuggled in with a docs update.

**Until that lands, treat automation scripts as development artefacts, not as a public API.**

## What we do not collect

GlassNet does not phone home, does not upload your data, and has no telemetry. The live inspector binds to
your machine only. No dataset, checkpoint or training log leaves your disk unless you push it yourself.

## Reporting a privacy problem

Open an issue on this repository. If we published something that identifies a person, we will remove it
first and discuss afterwards.
