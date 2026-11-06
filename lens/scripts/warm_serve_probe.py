#!/usr/bin/env python3
"""warm_serve_probe.py - the warm tier over real HTTP, in the server (PLAN item (c2), prediction P38i).

Starts lens/serve.py on a free port, sends three chat requests whose prompts grow (so the conversation's
memory grows), then asks the server what it remembers - GET /memory?i=N - which is answered from the
stored bytes with NO network call in that path. Every number printed here names the response it came from.

Usage: CHAT_DATA_DIR=data/chat/prepared_mix_honest python3 lens/scripts/warm_serve_probe.py [ship] [port]
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(ROOT)

SHIP = sys.argv[1] if len(sys.argv) > 1 else "lens/runs/chat/cl_d256_clean_24_ep24.npz"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8231
BASE = f"http://127.0.0.1:{PORT}"


def post(path, obj, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


def get(path, timeout=60):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


print("=" * 100)
print("THE WARM TIER IN THE SERVER, OVER REAL HTTP (prediction P38i)")
print("=" * 100)
print(f"ship {os.path.basename(SHIP)}   port {PORT}   corpus dir {os.environ.get('CHAT_DATA_DIR')}")

env = dict(os.environ)
log = open("lens/runs/_tmp/warm_serve_probe.log", "w")
proc = subprocess.Popen([sys.executable, "lens/serve.py", SHIP, "--port", str(PORT)],
                        stdout=log, stderr=subprocess.STDOUT, env=env)
try:
    ready, t0 = False, time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            print(f"  the server died during startup, exit {proc.returncode}; its log:")
            print(open("lens/runs/_tmp/warm_serve_probe.log").read()[-2000:])
            raise SystemExit(1)
        try:
            s, h = get("/health", timeout=5)
            if s == 200:
                ready = True
                print(f"  /health 200 after {time.time() - t0:.1f} s: {h}")
                break
        except Exception:
            time.sleep(1.0)
    if not ready:
        raise SystemExit("the server did not become ready in 180 s")

    prompts = ["hello there how are you doing today",
               "hello there how are you doing today and what do you think about the weather",
               "hello there how are you doing today and what do you think about the weather and the work you do"]
    for k, p in enumerate(prompts, 1):
        # requests 1-2 use the DEFAULT (write-only: memory accumulates, no token can change); request 3
        # asks for the read path explicitly, so the difference between the two modes is visible here.
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 40}
        if k == 3:
            body["warm_read"] = True
        s, r = post("/v1/chat/completions", body)
        w = r["lens"]["warm_memory"]
        print(f"  request {k}: HTTP {s}  tokens {r['lens']['tokens']}  read_from_memory={w['read_from_memory']}")
        print(f"    path: {r['lens']['path']}")
        print(f"    warm memory now: windows {w['windows']}  bytes {w['bytes']:,}"
              f"  ({w['bytes_per_window']} B/window, {w['bytes_per_token']:.5f} B/token)"
              f"  new prompt windows {w['new_prompt_windows']}  hits {w['hits']}")
        last = {"windows": w["windows"], "bytes": w["bytes"], "bpw": w["bytes_per_window"]}

    s0, r0 = get("/memory?i=0")
    print()
    print(f"  GET /memory?i=0   -> HTTP {s0}   network_calls={r0.get('network_calls')}")
    print(f"    holds {r0.get('windows_held')} windows; this window costs {r0.get('bytes_for_this_window')} bytes")
    print(f"    remembers: {r0.get('trace')}")
    print(f"    top tokens from memory: {[t['token'] for t in r0.get('top_tokens', [])]}")
    print(f"    limits: {r0.get('limits')}")

    sN, rN = get(f"/memory?i={last['windows'] - 1}")
    print(f"  GET /memory?i={last['windows'] - 1} (the last window) -> HTTP {sN}  network_calls={rN.get('network_calls')}")
    print(f"    remembers: {rN.get('trace')}")
    sX, rX = get(f"/memory?i={last['windows'] + 5}")
    print(f"  GET /memory?i={last['windows'] + 5} (out of range) -> HTTP {sX}  {rX.get('error')}")

    ok_size = (last["bytes"] == last["windows"] * last["bpw"])
    print()
    print(f"  P38i BAR: HTTP 200 and network_calls == 0 on /memory  -> "
          f"{'PASS' if (s0 == 200 and r0.get('network_calls') == 0) else 'FAIL'}")
    print(f"  P38i BAR: bytes == windows x {last['bpw']} in the server's store ({last['bytes']} vs "
          f"{last['windows'] * last['bpw']})  -> {'PASS' if ok_size else 'FAIL'}")
    print(f"  P38i BAR: the store grew across the three requests (windows {last['windows']})  -> "
          f"{'PASS' if last['windows'] > 1 else 'FAIL'}")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    log.close()
    print("  server stopped")
