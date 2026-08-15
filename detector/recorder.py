#!/usr/bin/env python3
"""
Circular-buffer recorder for the h32 camera.

Continuously segments the camera's RTSP (H.264 video + audio) into short .ts chunks,
keeping a rolling buffer. On an event trigger it assembles a clip covering
[trigger - preroll, trigger + postroll] — so we capture what happened BEFORE detection.
Audio is re-encoded to AAC so clips play everywhere.
"""
import os, time, glob, threading, subprocess, signal

class CircularRecorder:
    def __init__(self, rtsp, buffer_dir, events_dir,
                 buffer_secs=120, seg_secs=2, preroll=20, postroll=15):
        self.rtsp = rtsp
        self.buffer_dir = buffer_dir
        self.events_dir = events_dir
        self.buffer_secs = buffer_secs
        self.seg_secs = seg_secs
        self.preroll = preroll
        self.postroll = postroll
        self.proc = None
        self._janitor = None
        self._stop = threading.Event()
        os.makedirs(buffer_dir, exist_ok=True)
        os.makedirs(events_dir, exist_ok=True)

    def _launch(self):
        pattern = os.path.join(self.buffer_dir, "seg_%Y%m%d_%H%M%S.ts")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "warning",
               "-rtsp_transport", "tcp", "-i", self.rtsp,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "64k",
               "-f", "segment", "-segment_time", str(self.seg_secs),
               "-reset_timestamps", "1", "-strftime", "1", pattern]
        logpath = os.path.join(os.path.dirname(self.buffer_dir), "recorder.log")
        self._log = open(logpath, "ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log)
        self._proc_started = time.time()

    def start(self):
        for f in glob.glob(os.path.join(self.buffer_dir, "seg_*.ts")):
            try: os.remove(f)
            except OSError: pass
        self._launch()
        threading.Thread(target=self._maintain_loop, daemon=True).start()
        return self

    def _maintain_loop(self):
        """Watchdog + janitor: restart ffmpeg if it dies OR stalls (no new segments); prune old."""
        stall = max(6, 3 * self.seg_secs)
        while not self._stop.is_set():
            now = time.time()
            segs = glob.glob(os.path.join(self.buffer_dir, "seg_*.ts"))
            newest = max((os.path.getmtime(f) for f in segs), default=0)
            dead = self.proc is None or self.proc.poll() is not None
            stalled = (not dead) and (now - self._proc_started > stall) and (now - newest > stall)
            if dead or stalled:
                if stalled and self.proc:
                    try: self.proc.kill()
                    except Exception: pass
                self._launch()
            cutoff = now - self.buffer_secs
            for f in segs:
                try:
                    if os.path.getmtime(f) < cutoff: os.remove(f)
                except OSError:
                    pass
            self._stop.wait(self.seg_secs)

    def save_event(self, name, preroll=None, postroll=None):
        """Assemble a clip around 'now'. Blocks for postroll seconds. Returns clip path or None."""
        pre = self.preroll if preroll is None else preroll
        post = self.postroll if postroll is None else postroll
        t0 = time.time()
        time.sleep(post + self.seg_secs)   # let post-roll segments finish writing
        lo, hi = t0 - pre, t0 + post
        segs = sorted(f for f in glob.glob(os.path.join(self.buffer_dir, "seg_*.ts"))
                      if lo <= os.path.getmtime(f) <= hi + self.seg_secs)
        if not segs:
            return None
        segs = [s for s in segs if os.path.exists(s) and os.path.getsize(s) > 0]
        if not segs:
            return None
        listfile = os.path.join(self.events_dir, f"{name}.txt")
        with open(listfile, "w") as fh:
            for s in segs:
                fh.write(f"file '{s}'\n")
        out = os.path.join(self.events_dir, f"{name}.mp4")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", listfile,
                        "-c", "copy", out], check=False)
        try: os.remove(listfile)
        except OSError: pass
        return out if os.path.exists(out) else None

    def stop(self):
        self._stop.set()
        if self.proc:
            self.proc.send_signal(signal.SIGINT)
            try: self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired: self.proc.kill()


if __name__ == "__main__":
    import sys, json
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    rtsp = cfg.get("rtsp_camera_direct") or cfg.get("rtsp_main", "rtsp://***REMOVED-CREDS***@***REMOVED-IP***:554/realmonitor?channel=0&stream=0.sdp")
    base = os.path.dirname(os.path.abspath(__file__))
    rec = CircularRecorder(rtsp, os.path.join(base, "buffer"), os.path.join(base, "events"),
                           buffer_secs=60, preroll=8, postroll=4)
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("starting buffer; recording 14s then saving an event clip…")
        rec.start(); time.sleep(14)
        path = rec.save_event("selftest")
        rec.stop()
        print("event clip:", path)
        if path:
            subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type,codec_name,duration", "-of", "default=nw=1", path])
    else:
        print("Recording to circular buffer. Ctrl-C to stop.")
        rec.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            rec.stop()
