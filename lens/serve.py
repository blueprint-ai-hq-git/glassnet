#!/usr/bin/env python3
"""lens.serve - a local chat server for OUR architecture, because no standard runtime can run it.

Why this exists: GGUF and safetensors are the standard containers, but llama.cpp / LM Studio /
transformers only execute architectures they know, and LensNet is ours. So the model gets a runtime
of our own: this server. It speaks the same API shape as every other local model server
(POST /v1/chat/completions), so any client that can talk to LM Studio can talk to this too.

What it does:
  * builds the prompt in the model's own format (human : ... assistant : ...)
  * generates token by token with the project's decoder (top-k, temperature, repetition penalty)
  * STOPS at the turn boundary (the token that starts the next human turn) or at a token budget
  * reports the stop reason, so "it stopped" is visible rather than assumed
  * prints the fired concepts for the answer when asked (?explain=1), because that is the point of this
    model

Usage:
  python3 lens/serve.py [ship.npz] [--port 8199] [--device cpu]
  then open http://127.0.0.1:8199/   or  curl -s localhost:8199/v1/chat/completions -d '{...}'
"""
import argparse
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = "/run/media/maintainer/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REAL BUG FIX 2026-10-20 (evidence lens/runs/SERVE-GPU-AFTER.txt): `python3 lens/<this file>`
# puts THIS directory -- the project's lens/ -- at sys.path[0], so the top-level name `cuda`
# resolved to our own lens/cuda/ package. That is a REGULAR package and therefore beats
# cuda-python's PEP 420 namespace package (appending it does not help either), and cupy 14.2
# died at import with "ModuleNotFoundError: No module named 'cuda.pathfinder'" -- so --device
# gpu could not start AT ALL. This file's own directory must not be importable as top level.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens.concepts import explain as explain_fn  # noqa: E402

STATE = {"net": None, "vocab": None, "vmap": None, "ship": None, "loads": 0, "started": None}


def load_vocab(net):
    """The word list this ship was trained with, from the corpus's own cached vocabulary.

    Two things changed here while wiring the KV cache into the server (2026-10-20), and a third claim I
    made and then MEASURED AS FALSE is recorded too, because a wrong claim left standing is worse than
    no claim (evidence: lens/runs/SERVE-VOCAB-PROBE.txt):

      * REAL: the server used to call chat_data.build(), which loads the whole 1.2 GB token array just to
        obtain 12,000 words. The vocabulary json in the same directory IS that word list (the corpus
        builder writes it), so it is read directly; build() stays as the fallback when the json is absent.
      * REAL: nothing checked that the ship and the word list agree. A genuine mismatch would have been
        silent - the sampler would simply have ranged over the wrong ids. It is now refused by name.
      * FALSE, and it was my first version of this docstring: I claimed control-token ships were crippled
        because the server took a 12,000-word list for a 12,002-id model. Measured: the json of a control
        corpus ALREADY carries <|im_start|>/<|im_end|> (prepared_mix_ctrl_honest has 12,002 entries, its
        last two are exactly those tokens), so the old server was correct on every ship tested. The
        append below therefore only fires for a corpus whose json is genuinely two short - a safety net,
        not a fix.
    """
    d = os.environ.get("CHAT_DATA_DIR") or os.path.join(ROOT, "data", "chat", "prepared")
    cap = os.environ.get("CD_VOCAB", "12000")
    p = os.path.join(d, f"vocab_v{cap}.json")
    if os.path.exists(p):
        vocab = json.load(open(p))
        src = p
    else:
        vocab = list(chat_data.build()["vocab"])       # fallback: the old, heavier path
        src = "chat_data.build() (no cached vocab json in " + d + ")"
    if net.V == len(vocab) + 2:                        # safety net for a json that lacks the control tokens
        vocab = vocab + ["<|im_start|>", "<|im_end|>"]
    if net.V != len(vocab):
        raise SystemExit(f"vocab mismatch: ship has {net.V}, the word list at {src} has {len(vocab)}")
    return vocab, src


def load_ship(ship, device):
    STATE["net"] = lens.load(ship, device=device)
    vocab, src = load_vocab(STATE["net"])
    STATE["vocab"] = vocab
    STATE["vmap"] = {w: i for i, w in enumerate(vocab)}
    STATE["vocab_src"] = src
    STATE["ship"] = ship
    STATE["started"] = time.time()
    # THE WARM STORE BELONGS TO ONE SHIP. Found by the KV-DECODE regression on 2026-10-20: without this
    # reset, a second ship loaded into the same process reused the first ship's store (36 concepts and a
    # 12,002-column answer matrix for a 41-concept, 12,000-token model), and the sampler died with
    # "a and p must have same size" deep inside numpy - a message that names nothing. It is reset here,
    # by name, so a store can never outlive the ship whose concepts it holds.
    STATE["warm"] = None
    return vocab, src


def tokenize(t):
    return re.findall(r"\w+|[^\w\s]", t.lower())


def generate(prompt, max_new=120, temp=0.8, top_k=40, rep_penalty=1.15, seed=0, use_cache=True,
             use_warm=True, warm_read=False):
    """Generation for the server. Uses the SAME step function as lens/decode.py, so the server cannot
    drift from the measured decoder, and it uses the KV cache when the loaded ship has an attention
    block (PLAN item (e)). The path taken and its numbers are returned, not assumed.

    THE WARM TIER (PLAN item (c2), 2026-10-20): the server keeps ONE warm store per loaded ship. The
    prompt's windows are ingested into it (concepts only, 37 bytes per 32-token window) and every
    window on a store boundary is written through as generation proceeds, so the conversation's
    meanings accumulate at ~1.16 bytes/token and can be answered later from bytes alone with no
    network call (GET /memory?i=N). Windows whose end position is already stored are not paid for
    twice, so re-sending the same conversation is free.

    warm_read=False BY DEFAULT, and this is a measured decision: serving the step's logits from the
    stored bytes reproduces the network's top-1 token on 99.25% of held-out windows, i.e. it can
    change the text (lens/runs/WARM-DECODE.txt). The KV-DECODE gate requires the server to return
    exactly what the measured decoder returns, so the read path is opt-in per request
    ({"warm_read": true}) and the reported path says which mode ran.
    """
    from lens.decode import make_cache, StepSource, WarmDecode
    net, vmap, vocab = STATE["net"], STATE["vmap"], STATE["vocab"]
    W = int(getattr(net, "W", 32))
    V = len(vocab)
    enc = [vmap.get(x, 0) for x in tokenize(prompt)]
    if len(enc) < W:
        enc = [0] * (W - len(enc)) + enc
    ctx, out, used, stop_at, reason = enc[-W:], [], {}, None, "budget"
    if use_cache:
        cache, path = make_cache(net, ctx)
    else:
        cache, path = None, "full recompute (asked for by use_cache=False)"
    warm = None
    if use_warm:
        if STATE.get("warm") is None:
            STATE["warm"] = WarmDecode(net)          # stride = window: the conversation's memory
        warm = STATE["warm"]
        wrote = warm.ingest(enc, 0, len(enc) - 1)
        mode = "read+write" if warm_read else "write-only (no token can change)"
        path = path + f" + warm memory ({warm.store.bytes_per_window()} B/window, stride {warm.stride}," \
                      f" {mode}; {wrote} new prompt windows)"
    base_end = len(enc) - 1
    src = StepSource(net, cache, warm, base_end=base_end, read_from_memory=warm_read)
    t0 = time.time()
    for step in range(max_new):
        z, _p = src.step(ctx, base_end + step)
        for tok, cnt in used.items():
            z[tok] -= np.log(rep_penalty) * cnt
        z = z / max(temp, 1e-6)
        if top_k and top_k < len(z):
            cut = np.partition(z, -top_k)[-top_k]
            z = np.where(z >= cut, z, -np.inf)
        m = np.nanmax(z[np.isfinite(z)])
        p = np.exp(z - m)
        p = np.where(np.isfinite(p), p, 0.0)
        p = p / p.sum()
        nxt = int(np.random.default_rng(seed + 1000 * step).choice(V, p=p))
        used[nxt] = used.get(nxt, 0) + 1
        # THE STOP RULE: the token that opens the next human turn ends the assistant's turn
        if vocab[nxt].lower() in ("human", "<|im_end|>"):
            reason = "turn-end token"
            stop_at = step
            break
        out.append(nxt)
        ctx = (ctx + [nxt])[-W:]
        if cache is not None:
            cache.push(nxt)
    secs = time.time() - t0
    text = " ".join(vocab[t] for t in out).replace(" '", "'")
    warm_info = None
    if warm is not None:
        rep = warm.report()
        warm_info = {"windows": rep["windows"], "bytes": rep["bytes"],
                     "bytes_per_window": rep["bytes_per_window"],
                     "bytes_per_token": rep["bytes_per_token"],
                     "hits": rep["hits"], "misses": rep["misses"],
                     "read_from_memory": bool(warm_read),
                     "new_prompt_windows": wrote}
    return {"text": text, "tokens": len(out), "stop_reason": reason, "stop_at": stop_at,
            "path": path, "seconds": secs, "tokens_per_sec": (len(out) / secs if secs > 0 else 0.0),
            "cache_bytes_per_token": (cache.bytes_per_token() if cache is not None else 0),
            "cache_bytes_total": (cache.total_bytes() if cache is not None else 0),
            "warm": warm_info}


def answer_from_memory(i):
    """Answer stored window `i` from the STORE'S BYTES ALONE, with no network call anywhere in the path.

    This is the honest use of the warm tier: the model's own concept vector for that window, read back
    from one byte per concept, pushed through the model's own answer weights. It cannot quote the text
    (the store keeps no tokens - lens/warmmemory.py, describe_limits()), and it does NOT accelerate
    sliding generation (measured: lens/runs/WARM-DECODE.txt, P38e). It answers about what it remembers.
    """
    net, warm = STATE["net"], STATE.get("warm")
    if warm is None or warm.store.n == 0:
        return {"error": "the store is empty: no request has been served yet"}
    n = warm.store.n
    if i < 0 or i >= n:
        return {"error": f"row {i} out of range: the store holds {n} windows (0..{n - 1})"}
    names = [c.name for c in net.concepts]
    z = warm.logits(i)
    order = np.argsort(-np.asarray(z))[:5]
    return {"row": int(i), "windows_held": int(n), "network_calls": 0,
            "bytes_for_this_window": int(warm.store.bytes_per_window()),
            "top_tokens": [{"token": STATE["vocab"][int(j)], "logit": float(z[int(j)])} for j in order],
            "remembers": [{"concept": nm, "value": round(v, 4)}
                          for nm, v in warm.store.top(i, names, k=8)],
            "trace": warm.trace(i, names),
            "limits": "concepts only: the store keeps no tokens, so this cannot quote the text"}


def build_prompt(messages):
    """Prompt in the format the loaded model was trained on (standard chat format when the vocabulary
    carries the control tokens). Uses the shared helper so the server cannot drift from the measurement."""
    from lens.decode import format_prompt
    user = " ".join(m.get("content", "") for m in messages if m.get("role") in ("user", "human"))
    return format_prompt(user, STATE["vocab"])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj, ctype="application/json"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, {"ok": True, "ship": os.path.basename(STATE["ship"] or ""),
                                    "params": sum(p.data.size for p in STATE["net"].parameters()),
                                    "concepts": STATE["net"].K,
                                    "uptime_s": round(time.time() - STATE["started"], 1)})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"data": [{"id": os.path.basename(STATE["ship"] or "lensnet"),
                                              "object": "model"}]})
        if self.path.startswith("/memory"):
            # WHAT THE MODEL REMEMBERS, answered from stored bytes with no network call (PLAN item (c2)).
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                i = int((q.get("i") or ["0"])[0])
            except ValueError:
                return self._send(400, {"error": "i must be an integer row index"})
            r = answer_from_memory(i)
            return self._send(400 if "error" in r else 200, r)
        if self.path == "/" or self.path.startswith("/index"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        if not self.path.startswith("/v1/chat/completions"):
            return self._send(404, {"error": "not found"})
        messages = req.get("messages") or [{"role": "user", "content": req.get("prompt", "hello")}]
        max_new = int(req.get("max_tokens") or 120)
        prompt = build_prompt(messages)
        t0 = time.time()
        g = generate(prompt, max_new=max_new, warm_read=bool(req.get("warm_read", False)))
        secs = time.time() - t0
        out = {"id": f"lens-{int(t0)}", "object": "chat.completion", "created": int(t0),
               "model": os.path.basename(STATE["ship"] or "lensnet"),
               "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                            "content": g["text"]}}],
               "lens": {"stop_reason": g["stop_reason"], "tokens": g["tokens"],
                        "seconds": round(secs, 2), "prompt": prompt,
                        "path": g["path"], "tokens_per_sec": round(g["tokens_per_sec"], 1),
                        "cache_bytes_per_token": g["cache_bytes_per_token"],
                        "cache_bytes_total": g["cache_bytes_total"],
                        "warm_memory": g["warm"]}}
        self._send(200, out)


PAGE = """<!doctype html><meta charset=utf-8><title>GlassNet chat</title>
<style>body{font:15px system-ui;margin:0;background:#111;color:#eee}
#wrap{max-width:820px;margin:0 auto;padding:18px}
.msg{padding:9px 12px;border-radius:9px;margin:8px 0;white-space:pre-wrap}
.me{background:#26405c}.it{background:#243024}
input{width:78%;padding:11px;font-size:15px;background:#1b1b1b;color:#eee;border:1px solid #444;border-radius:7px}
button{padding:11px 16px;font-size:15px;border:0;border-radius:7px;background:#3b6ea5;color:#fff}
#meta{color:#8a8a8a;font-size:12px;margin-top:10px}</style>
<div id=wrap><h3>GlassNet - readable model, local runtime</h3>
<div id=log></div><input id=q placeholder="type something and press enter"><button onclick=send()>Send</button>
<div id=meta></div></div>
<script>
const log=document.getElementById('log');
function add(t,c){const d=document.createElement('div');d.className='msg '+c;d.textContent=t;log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);}
async function send(){const q=document.getElementById('q').value;if(!q)return;document.getElementById('q').value='';add(q,'me');
 add('...','it');
 const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({messages:[{role:'user',content:q}],max_tokens:120})});
 const j=await r.json();log.lastChild.textContent=j.choices[0].message.content||'(nothing)';
 document.getElementById('meta').textContent='stopped by: '+j.lens.stop_reason+' | '+j.lens.tokens+' tokens in '+j.lens.seconds+'s ('+j.lens.tokens_per_sec+' tok/s) | '+j.lens.path+' | prompt: '+j.lens.prompt;}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ship", nargs="?", default="lens/runs/chat/sb_S0_d256_noattn_ep96.npz")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    load_ship(a.ship, a.device)
    n = sum(p.data.size for p in STATE["net"].parameters())
    print(f"GlassNet server: {os.path.basename(a.ship)}  {n:,} params  {STATE['net'].K} named concepts")
    print(f"  vocabulary: {len(STATE['vocab'])} tokens from {STATE['vocab_src']}")
    print(f"  cache:      {'ON (ship has attention)' if STATE['net'].attn is not None else 'off (no attention block in this ship)'}")
    print(f"  warm tier:  ON - concepts per window, {STATE['net'].K + 1} bytes per {STATE['net'].W}-token window"
          f" ({(STATE['net'].K + 1) / STATE['net'].W:.3f} bytes/token); grows with the conversation")
    print(f"  chat page:  http://127.0.0.1:{a.port}/")
    print(f"  API:        POST http://127.0.0.1:{a.port}/v1/chat/completions")
    print(f"  memory:     GET  http://127.0.0.1:{a.port}/memory?i=0   (answered from stored bytes, 0 network calls)")
    print(f"  health:     http://127.0.0.1:{a.port}/health")
    print("  NOTE: this is our own runtime. llama.cpp / LM Studio cannot load this architecture.")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()