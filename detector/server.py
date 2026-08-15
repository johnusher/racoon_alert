#!/usr/bin/env python3
"""
Lightweight monitor server for the h32 detector.

Serves a live annotated MJPEG view (boxes drawn as animals/people appear), a recent-events
sidebar (snapshot thumbnails + clip links), and a recording indicator. Runs in a daemon
thread inside detect.py.
"""
import os, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>h32 · monitor</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0b0d10;--panel:#14181d;--line:#262d36;--txt:#e7ecf2;--dim:#8b97a6;--accent:#4cc2ff;--rec:#ff5b5b;--good:#3ad29f}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);
   font:14px/1.4 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
 header{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--line);background:#101317}
 .brand{font-weight:700}.brand b{color:var(--accent)} .sp{flex:1}
 .pill{display:inline-flex;align-items:center;gap:7px;padding:4px 11px;border-radius:999px;background:#1b2027;border:1px solid var(--line);font-size:12px;color:var(--dim)}
 .dot{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 8px var(--good)}
 .rec{color:var(--rec);font-weight:700;display:none} .rec.on{display:inline-flex;align-items:center;gap:6px}
 .rec .dot{background:var(--rec);box-shadow:0 0 8px var(--rec);animation:blink 1s infinite}
 @keyframes blink{50%{opacity:.3}}
 main{flex:1;display:flex;min-height:0}
 .view{flex:1;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
 .view img{max-width:100%;max-height:100%}
 aside{width:300px;border-left:1px solid var(--line);background:var(--panel);overflow-y:auto;padding:12px}
 aside h3{margin:4px 0 10px;font-size:12px;letter-spacing:.4px;color:var(--dim);text-transform:uppercase}
 .ev{display:flex;gap:10px;padding:8px;border:1px solid var(--line);border-radius:10px;margin-bottom:8px;background:#0f1317}
 .ev img{width:88px;height:50px;object-fit:cover;border-radius:6px;background:#000}
 .ev .m{flex:1;min-width:0}.ev .tag{font-weight:700}.ev .tag.RACCOON{color:#ffcf6a}.ev .tag.ANIMAL{color:#4cc2ff}.ev .tag.PERSON{color:#3ad29f}
 .ev .t{color:var(--dim);font-size:11px}.ev a{color:var(--accent);font-size:12px;text-decoration:none}
 .empty{color:var(--dim);font-size:12px;padding:8px}
</style></head><body>
 <header><span class=brand><b>h32</b> monitor</span>
   <span class="rec" id=rec><span class=dot></span>RECORDING</span><span class=sp></span>
   <span class=pill><span class=dot></span><span id=status>live</span></span></header>
 <main>
   <div class=view><img src="/stream.mjpg" alt="live"></div>
   <aside><h3>Recent events</h3><div id=events><div class=empty>none yet — watching…</div></div></aside>
 </main>
<script>
async function poll(){
  try{
    const r=await fetch('/events.json',{cache:'no-store'});const d=await r.json();
    document.getElementById('status').textContent=d.status||'live';
    document.getElementById('rec').classList.toggle('on',!!d.recording);
    const box=document.getElementById('events');
    if(!d.events.length){box.innerHTML='<div class=empty>none yet — watching…</div>';}
    else box.innerHTML=d.events.map(e=>`<div class=ev>
      <img src="/file/${e.snapshot}" onerror="this.style.visibility='hidden'">
      <div class=m><div class="tag ${e.tag}">${e.tag}</div><div class=t>${e.time} · ${e.detail||''}</div>
      ${e.clip?`<a href="/file/${e.clip}" target=_blank>▶ clip</a>`:''}</div></div>`).join('');
  }catch(e){document.getElementById('status').textContent='detector offline';}
}
poll();setInterval(poll,3000);
</script></body></html>"""


class MonitorServer:
    def __init__(self, port, events_dir, fps=3):
        self.port = port
        self.events_dir = events_dir
        self.fps = fps
        self._jpeg = b""
        self._lock = threading.Lock()
        self.events = []          # newest first
        self.status = "starting"
        self.recording = False
        self.httpd = None

    def update_frame(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def add_event(self, tag, snapshot, detail="", clip=None):
        self.events.insert(0, {"tag": tag, "snapshot": snapshot, "detail": detail,
                               "clip": clip, "time": time.strftime("%H:%M:%S")})
        self.events = self.events[:25]

    def set_clip(self, snapshot, clip):
        for e in self.events:
            if e["snapshot"] == snapshot:
                e["clip"] = clip; break

    def start(self):
        server = self
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/":
                    self._send(200, "text/html", PAGE.encode())
                elif p == "/events.json":
                    body = json.dumps({"status": server.status, "recording": server.recording,
                                       "events": server.events}).encode()
                    self._send(200, "application/json", body)
                elif p == "/stream.mjpg":
                    self._stream()
                elif p.startswith("/file/"):
                    self._file(os.path.basename(p[len("/file/"):]))
                else:
                    self._send(404, "text/plain", b"not found")
            def _send(self, code, ctype, body):
                self.send_response(code); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                try: self.wfile.write(body)
                except Exception: pass
            def _file(self, name):
                path = os.path.join(server.events_dir, name)
                if not os.path.isfile(path): return self._send(404, "text/plain", b"no file")
                ctype = "video/mp4" if name.endswith(".mp4") else "image/jpeg"
                with open(path, "rb") as f: body = f.read()
                self._send(200, ctype, body)
            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                try:
                    while True:
                        frame = server.get_frame()
                        if frame:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame); self.wfile.write(b"\r\n")
                        time.sleep(1.0 / max(1, server.fps))
                except Exception:
                    pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self
