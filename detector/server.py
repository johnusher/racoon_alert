#!/usr/bin/env python3
"""
Data server for the h32 detector's live monitor.

The monitor UI itself is web/monitor.html, served by go2rtc on :1984 so that it is
same-origin with the WebRTC stream and can show the real camera video and audio.
This server supplies what go2rtc cannot: the current detections (as coordinates, for
the page to draw over the video), the camera-signal state, the recent-events list, and
the snapshot/clip files. Responses are CORS-open because the page is on another port.

The annotated MJPEG is still served at /stream.mjpg as a fallback (and is only encoded
while somebody is actually watching it). Runs in a daemon thread inside detect.py.
"""
import os, json, time, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset=utf-8><title>h32 · detector</title>
<meta http-equiv=refresh content="0; url={url}">
<style>body{{margin:0;height:100vh;display:flex;flex-direction:column;gap:14px;
 align-items:center;justify-content:center;background:#0b0d10;color:#e7ecf2;
 font:14px/1.6 -apple-system,BlinkMacSystemFont,system-ui,sans-serif}}
 a{{color:#4cc2ff}} .d{{color:#8b97a6;font-size:12.5px}}</style></head><body>
<div>The h32 monitor lives at <a href="{url}">{url}</a> — taking you there…</div>
<div class=d>This port is the detector's data feed:
 <a href="/state.json">state.json</a> · <a href="/stream.mjpg">annotated MJPEG</a></div>
</body></html>"""


class MonitorServer:
    def __init__(self, port, events_dir, fps=3, monitor_url="http://127.0.0.1:1984/monitor.html"):
        self.port = port
        self.events_dir = events_dir
        self.fps = fps
        self.monitor_url = monitor_url
        self._jpeg = b""
        self._lock = threading.Lock()
        self.events = []          # newest first
        self.status = "starting"
        self.online = False       # is the camera actually feeding the detector?
        self.signal = None        # why not, when it isn't
        self.recording = False
        self.boxes = []           # current detections, in source-frame coordinates
        self.faces = []           # current face hits: {box, who, score}
        self.frame_w = self.frame_h = 0
        self.last_frame = 0.0     # when the detector last got a frame (epoch seconds)
        self.mjpeg_clients = 0    # nobody watching → don't spend CPU encoding JPEGs
        self.httpd = None
        # ---- switches the monitor page can flip, live (see /set) ----
        # Runtime only, deliberately: config.json holds the defaults, and a restart
        # should come back up in the configured state rather than in whatever mood the
        # last session left it in. Detection is never affected — events still fire and
        # still appear in the sidebar; these only decide what an event *produces*.
        self.auto_record = True       # write the .jpg snapshot and the .mp4 clip
        self.email_alerts = True      # send the e-mail alert
        self.email_available = False  # is e-mail even configured? (greys the button out)
        self.on_switch = None         # detect.py hooks this to log a flip

    def switches(self):
        with self._lock:
            return {"auto_record": self.auto_record, "email_alerts": self.email_alerts,
                    "email_available": self.email_available}

    def set_switch(self, name, on):
        """Idempotent: sets an absolute value rather than flipping, so a repeated or
        prefetched request can never leave the button and the detector disagreeing."""
        if name not in ("auto_record", "email_alerts"):
            return None
        if name == "email_alerts" and not self.email_available:
            return None                       # nothing to turn on; don't lie to the UI
        with self._lock:
            before = getattr(self, name)
            setattr(self, name, bool(on))
        if before != bool(on) and self.on_switch:
            self.on_switch(name, bool(on))
        return bool(on)

    def update_frame(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def watching(self, delta):
        with self._lock:
            self.mjpeg_clients = max(0, self.mjpeg_clients + delta)

    def set_state(self, online, signal, recording, boxes, frame_w, frame_h, last_frame=0.0,
                  faces=()):
        with self._lock:
            self.online, self.signal, self.recording = online, signal, recording
            self.boxes, self.frame_w, self.frame_h = boxes, frame_w, frame_h
            self.last_frame = last_frame
            self.faces = list(faces)
            self.status = "live" if online else "no camera signal"

    def state(self):
        with self._lock:
            return {"status": self.status, "online": self.online, "signal": self.signal,
                    "recording": self.recording, "boxes": self.boxes,
                    "faces": self.faces,
                    "frame_w": self.frame_w, "frame_h": self.frame_h,
                    "last_frame": self.last_frame, "events": self.events,
                    "auto_record": self.auto_record, "email_alerts": self.email_alerts,
                    "email_available": self.email_available}

    def add_event(self, tag, snapshot, detail="", clip=None, who=None):
        self.events.insert(0, {"tag": tag, "snapshot": snapshot, "detail": detail,
                               "clip": clip, "who": who,
                               "time": time.strftime("%H:%M:%S")})
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
                    self._send(200, "text/html", PAGE.format(url=server.monitor_url).encode())
                elif p in ("/state.json", "/events.json"):
                    self._send(200, "application/json", json.dumps(server.state()).encode())
                elif p == "/set":
                    self._set(self.path.partition("?")[2])
                elif p == "/stream.mjpg":
                    self._stream()
                elif p.startswith("/file/"):
                    self._file(os.path.basename(p[len("/file/"):]))
                else:
                    self._send(404, "text/plain", b"not found")
            def _cors(self):
                # The monitor page is served by go2rtc on another port, so it reads us
                # cross-origin. Everything here is already public on loopback.
                self.send_header("Access-Control-Allow-Origin", "*")
            def _send(self, code, ctype, body):
                self.send_response(code); self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                try: self.wfile.write(body)
                except Exception: pass
            def _set(self, query):
                """/set?record=0&email=1 — the monitor's two switches.

                A GET, not a POST: the page is served from another port, and a GET
                needs no CORS preflight. Safe to do that here because it sets an
                absolute value (not a toggle) on a loopback-only server.
                """
                q = urllib.parse.parse_qs(query)
                for key, attr in (("record", "auto_record"), ("email", "email_alerts")):
                    if key in q:
                        on = q[key][0].strip().lower() not in ("0", "false", "off", "")
                        server.set_switch(attr, on)
                self._send(200, "application/json", json.dumps(server.switches()).encode())
            def _file(self, name):
                path = os.path.join(server.events_dir, name)
                if not os.path.isfile(path): return self._send(404, "text/plain", b"no file")
                ctype = "video/mp4" if name.endswith(".mp4") else "image/jpeg"
                with open(path, "rb") as f: body = f.read()
                self._send(200, ctype, body)
            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self._cors()
                self.send_header("Cache-Control", "no-store"); self.end_headers()
                server.watching(+1)                # tells detect.py it is worth encoding
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
                finally:
                    server.watching(-1)
        try:
            self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        except OSError as e:
            raise SystemExit(
                f"h32 detector: port {self.port} is already in use ({e.strerror}).\n"
                f"  Another detector is probably already running — "
                f"`pkill -f detector/detect.py`, or set monitor_port in config.json.") from None
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self
