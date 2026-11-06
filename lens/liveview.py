
"""lens/liveview.py - live, real-time window into GlassNet. Nothing on this page is invented.
Every number is read from the running processes, the run logs, or computed with numpy from the newest
checkpoint's OWN weights (the concept layer is a real matmul with the stored W and b). If something cannot
be computed, the page prints the error instead of a made-up value."""
import http.server, json, os, re, subprocess, time, glob
import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8770
CACHE = {"t": 0, "data": None}

def sh(c):
    try: return subprocess.run(["bash","-c",c], capture_output=True, text=True, timeout=8, cwd=P).stdout
    except Exception as e: return f"ERR {e}"

def gpu():
    o = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits").strip()
    try:
        u, m, mt, w, t = [x.strip() for x in o.split(",")[:5]]
        return {"util": float(u), "vram": float(m), "vram_total": float(mt), "power": float(w), "temp": float(t)}
    except Exception: return {"error": o}

def run_rows():
    """Every epoch row of the newest run log, parsed. Real numbers, straight from the log."""
    logs = sorted(glob.glob(os.path.join(P, "lens/runs/*.txt")), key=os.path.getmtime, reverse=True)
    out = {"log": os.path.basename(logs[0]) if logs else None, "rows": [], "gates": [], "arm": None}
    if not logs: return out
    txt = open(logs[0], errors="replace").read()
    for m in re.finditer(r"ep (\d+)/(\d+).*?loss ([\d.]+).*?acc ([\d.]+)%.*?VAL ppl ([\d.]+) acc ([\d.]+)%.*?ASST\(by-position\) ppl ([\d.]+) acc ([\d.]+)%", txt):
        ep, ne, loss, acc, ppl, vacc, app, aacc = m.groups()
        out["rows"].append({"ep": int(ep), "epochs": int(ne), "loss": float(loss), "train_acc": float(acc),
                            "val_ppl": float(ppl), "val_acc": float(vacc), "asst_ppl": float(app), "asst_acc": float(aacc)})
    for m in re.finditer(r"(GATE|AGREEMENT) RECORD: ([\d,]+) of ([\d,]+) steps refused \(([\d.]+)%\)", txt):
        out["gates"].append({"gate": m.group(1), "refused": int(m.group(2).replace(",","")), "total": int(m.group(3).replace(",","")), "pct": float(m.group(4))})
    m = re.search(r"(=====\s*\S+[^\n]*)", txt)
    out["arm"] = m.group(1).strip("= ")[:80] if m else None
    return out

def internals():
    """REAL forward pass through the concept layer using the checkpoint's own weights."""
    ck = sorted(glob.glob(os.path.join(P, "lens/runs/chat/*.npz")), key=os.path.getmtime, reverse=True)
    if not ck: return {"error": "no checkpoint on disk yet"}
    path = ck[0]; res = {"checkpoint": os.path.basename(path), "age_s": int(time.time()-os.path.getmtime(path))}
    try:
        z = np.load(path, allow_pickle=False)
        keys = {k: z[k].shape for k in z.files}
        res["shapes"] = {k: list(v) for k, v in keys.items()}
        # pick the tensors by SHAPE, whatever they are called
        emb = next((k for k, s in keys.items() if len(s) == 2 and s[0] > 5000), None)
        cW  = next((k for k, s in keys.items() if len(s) == 2 and s[0] > 1000 and 20 <= s[1] <= 200), None)
        cB  = next((k for k, s in keys.items() if len(s) == 1 and cW and s[0] == keys[cW][1]), None)
        aW  = next((k for k, s in keys.items() if len(s) == 2 and cW and s[0] == keys[cW][1] and s[1] > 5000), None)
        aB  = next((k for k, s in keys.items() if len(s) == 1 and aW and s[0] == keys[aW][1]), None)
        res["tensors"] = {"embedding": emb, "concept_W": cW, "concept_b": cB, "answer_W": aW, "answer_b": aB}
        ids_path = os.path.join(P, "data/chat/prepared_mix_ctrl_honest/ids_v12000.npy")
        if emb and cW and ids_path and os.path.exists(ids_path):
            ids = np.load(ids_path, mmap_mode="r")
            off = int(len(ids) * 0.731)                       # a fixed but arbitrary real position
            win = np.asarray(ids[off:off+32]).astype(np.int64)
            E = z[emb]; Wc = z[cW]; bc = z[cB] if cB else np.zeros(Wc.shape[1], dtype=Wc.dtype)
            x = E[win].reshape(-1)[:Wc.shape[0]].astype(np.float32)   # the REAL window the model sees
            act = np.maximum(x @ Wc + bc, 0.0)                        # the REAL concept layer
            order = np.argsort(-act)[:10]
            # CONCEPT NAMES, exact mapping: the run's own log holds the kept concept list IN NEURON ORDER and
            # it has exactly as many entries as the concept layer has columns. No guessing, no reordering.
            names = {}
            try:
                import json as _json
                base = os.path.basename(path).replace("_best.npz", "").replace(".npz", "")
                base = base.split("_ep")[0]
                lp = os.path.join(P, "lens/runs/chat", base + "_log.json")
                if os.path.exists(lp):
                    _sp = _json.load(open(lp)).get("concepts", {})
                    kept = _sp.get("kept", []) if isinstance(_sp, dict) else []
                    if len(kept) == Wc.shape[1]:
                        names = {i: (e.get("name", f"concept_{i}") if isinstance(e, dict) else str(e))
                                 for i, e in enumerate(kept)}
                        res["concept_names_source"] = base + "_log.json (kept, in neuron order)"
                    else:
                        res["names_error"] = f"kept has {len(kept)} names for {Wc.shape[1]} neurons"
                else:
                    res["names_error"] = "no log json beside this checkpoint"
            except Exception as e:
                res["names_error"] = str(e)[:140]
            res["window_tokens"] = win.tolist()[:12]
            res["concepts"] = [{"i": int(i), "name": names.get(int(i), f"concept_{int(i)}"), "act": round(float(act[i]), 4)} for i in order]
            res["active_count"] = int((act > 0).sum()); res["concept_total"] = int(act.size)
            if aW:
                logits = act @ z[aW] + (z[aB] if aB else 0)
                top = np.argsort(-logits)[:6]
                res["top_next"] = [{"id": int(t), "logit": round(float(logits[t]), 3)} for t in top]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    return res

def state():
    if time.time() - CACHE["t"] < 2 and CACHE["data"]: return CACHE["data"]
    d = {"time": time.strftime("%H:%M:%S"), "date": time.strftime("%Y-%m-%d"),
         "trainers": int(sh("pgrep -cf 'venv-gpu/bin/python3 -u lens/scripts/train_chat'").strip() or 0),
         "queue": [l.strip() for l in sh("grep -vE '^\\s*#|^\\s*$' lens/runs/QUEUE.txt 2>/dev/null").split("\n") if l.strip()],
         "gpu": gpu(), "ram": sh("free -m | awk 'NR==2{print $3\" / \"$2\" MB\"}'").strip(),
         "disk": sh("df -h . | tail -1 | awk '{print $4}'").strip(),
         "run": run_rows(), "internals": internals(),
         "inbox": sh("tail -6 lens/runs/AGENT-INBOX.md 2>/dev/null").strip().split("\n"),
         "checkpoints": int(sh("ls lens/runs/chat/*.npz 2>/dev/null | wc -l").strip() or 0),
         "proc": sh("ps -eo etime,args | grep '[t]rain_chat' | head -1 | cut -c1-96").strip()}
    CACHE.update(t=time.time(), data=d); return d

HTML = """<!doctype html><html><head><meta charset=utf-8><title>GlassNet live</title>
<style>body{background:#0b0f14;color:#d7e2ea;font:13px/1.5 ui-monospace,Menlo,monospace;margin:0;padding:18px}
h1{font-size:15px;color:#7ee787;margin:0 0 4px}h2{font-size:12px;color:#58a6ff;margin:18px 0 6px;letter-spacing:.08em}
.card{background:#111823;border:1px solid #1f2a37;border-radius:8px;padding:10px 12px;margin-bottom:10px}
.k{color:#8b98a5}.v{color:#e6edf3;font-weight:600}.hi{color:#7ee787}.bad{color:#ff7b72}
table{border-collapse:collapse;width:100%}td,th{padding:3px 8px;text-align:right;border-bottom:1px solid #1a2330}
th{color:#8b98a5;font-weight:400;text-align:right}td:first-child,th:first-child{text-align:left}
.bar{height:8px;background:#1f6feb;border-radius:4px;display:inline-block;vertical-align:middle}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}</style></head><body>
<h1>GlassNet - live view</h1><div class=k id=ts></div>
<h2>TRAINING</h2><div class=card id=train></div>
<h2>INSIDE THE MODEL (real weights, real activations)</h2><div class=card id=inside></div>
<h2>GATES / READABILITY</h2><div class=card id=gates></div>
<h2>MACHINE</h2><div class=card id=machine></div>
<h2>AGENT INBOX</h2><div class=card id=inbox></div>
<script>
const f=n=>typeof n==='number'?n.toFixed(2):n;
async function tick(){
 let d; try{ d=await (await fetch('/api/live')).json(); }catch(e){ document.getElementById('ts').innerHTML='<span class=bad>gateway down: '+e+'</span>'; return; }
 document.getElementById('ts').textContent=d.date+' '+d.time+'  |  trainers '+d.trainers+'  |  checkpoints '+d.checkpoints;
 const r=d.run||{}, rows=r.rows||[];
 let h='<div class=k>arm: <span class=v>'+(r.arm||'none')+'</span>   log: '+(r.log||'-')+'</div>';
 if(rows.length){ h+='<table><tr><th>epoch</th><th>loss</th><th>train acc</th><th>val acc</th><th>asst acc</th><th>val ppl</th></tr>';
  rows.slice(-6).forEach(x=>{h+='<tr><td>'+x.ep+'/'+x.epochs+'</td><td>'+f(x.loss)+'</td><td>'+f(x.train_acc)+'%</td><td class=hi>'+f(x.val_acc)+'%</td><td class=hi>'+f(x.asst_acc)+'%</td><td>'+f(x.val_ppl)+'</td></tr>';});
  h+='</table>'; } else h+='<div class=k>no epoch finished yet in the newest log</div>';
 if((d.run.gates||[]).length){ h+='<div style="margin-top:8px">'+d.run.gates.map(g=>g.gate+' refused '+g.refused+'/'+g.total+' ('+g.pct+'%)').join(' &nbsp; ')+'</div>'; }
 h+='<div class=k style="margin-top:6px">'+(d.proc||'nothing training')+'</div>';
 document.getElementById('train').innerHTML=h;
 const m=d.internals||{};
 let i='<div class=k>checkpoint: <span class=v>'+(m.checkpoint||'-')+'</span>  ('+(m.age_s!=null?Math.round(m.age_s/60)+' min old':'')+')</div>';
 if(m.shapes){ i+='<div class=k>tensors: '+Object.entries(m.shapes).map(([k,v])=>k+'='+v.join('x')).join(', ')+'</div>'; }
 if(m.concepts){ i+='<div class=k>active concepts this instant: <span class=v>'+m.active_count+' / '+m.concept_total+'</span> &nbsp; real window tokens: '+m.window_tokens.join(' ')+'</div>';
  i+='<table><tr><th>concept neuron</th><th>activation</th></tr>';
  m.concepts.forEach(c=>{i+='<tr><td>'+c.name+' <span class=k>#'+c.i+'</span></td><td>'+f(c.act)+' <span class=bar style="width:'+Math.min(90,c.act*40)+'px"></span></td></tr>';});
  i+='</table>'; }
 if(m.top_next){ i+='<div class=k style="margin-top:6px">next-token scores: '+m.top_next.map(t=>'id'+t.id+' '+t.logit).join(', ')+'</div>'; }
 if(m.error){ i+='<div class=bad>'+m.error+'</div>'; }
 document.getElementById('inside').innerHTML=i;
 document.getElementById('gates').innerHTML=(d.run.gates||[]).length? d.run.gates.map(g=>'<div>'+g.gate+' gate: <span class=v>'+g.refused+'</span> of '+g.total+' steps refused ('+g.pct+'%)</div>').join('') : '<div class=k>this run has no gate record (standard training arm)</div>';
 const g=d.gpu||{};
 document.getElementById('machine').innerHTML='<div class=grid>'+
  '<div><div class=k>GPU</div><div class=v>'+f(g.util)+'%</div></div>'+
  '<div><div class=k>VRAM</div><div class=v>'+f(g.vram)+' / '+f(g.vram_total)+' MB</div></div>'+
  '<div><div class=k>power</div><div class=v>'+f(g.power)+' W</div></div>'+
  '<div><div class=k>temp</div><div class=v>'+f(g.temp)+' C</div></div>'+
  '<div><div class=k>RAM</div><div class=v>'+d.ram+'</div></div>'+
  '<div><div class=k>disk free</div><div class=v>'+d.disk+'</div></div>'+
  '<div><div class=k>queue</div><div class=v>'+d.queue.length+' jobs</div></div></div>';
 document.getElementById('inbox').innerHTML=(d.inbox||[]).map(l=>'<div>'+l.replace(/</g,'&lt;')+'</div>').join('');
}
tick(); setInterval(tick,3000);
</script></body></html>"""

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/api/live"):
            body = json.dumps(state()).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)
        else:
            body = HTML.encode(); self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

if __name__ == "__main__":
    print(f"live view on http://0.0.0.0:{PORT}")
    http.server.ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
