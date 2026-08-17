#!/usr/bin/env python3
"""
Circular-buffer recorder for the h32 camera.

Continuously segments the camera's RTSP (H.264 video + audio) into short .ts chunks,
keeping a rolling buffer. On an event trigger it assembles a clip covering
[trigger - preroll, trigger + postroll] — so we capture what happened BEFORE detection.
Audio is re-encoded to AAC so clips play everywhere.
"""
import os, time, glob, threading, subprocess, signal


def stray_recorder_pids(ps_output, buffer_dir, me=None):
    """Which pids in `ps -axo pid=,command=` output are ffmpegs writing to OUR buffer?

    Kept a pure function so the matching can be tested without spawning anything: it
    decides what gets SIGINT'd, and a sloppy match here would kill somebody else's
    ffmpeg. Matches on the buffer path, never on the word "ffmpeg" alone.
    """
    out = []
    for line in ps_output.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if not pid.isdigit() or (me is not None and int(pid) == me):
            continue
        if "ffmpeg" in cmd and buffer_dir in cmd:
            out.append(int(pid))
    return out


# How long a freshly launched ffmpeg gets to deliver its FIRST segment before the
# watchdog gives up on it. Measured 2026-08-17: the Victure takes 3 s to over 6 s to start
# a new RTSP session, and the old rule — "no new segment for 6 s", counted from launch —
# killed slow opens just before they produced. Each kill leaves a stale session on the
# camera, which makes the next open slower still, so the recorder churned for 25 minutes
# with an EMPTY buffer (no clip for 20:54:46, `clip: FAILED` at 21:07:21).
STARTUP_GRACE_SECS = 20.0


def restart_reason(now, proc_started, newest, dead, stall, grace=STARTUP_GRACE_SECS):
    """Why the watchdog should relaunch ffmpeg — "dead", "stalled", or None to leave it.

    `newest` is the mtime of the newest segment in the buffer (0 if none). A segment
    written BEFORE this launch is not evidence that this launch is delivering, so until
    the first segment of the current process appears the yardstick is `grace`, not
    `stall`; once it is flowing, `stall` seconds without a new segment is a stall.
    """
    if dead:
        return "dead"
    delivered = newest >= proc_started
    window = stall if delivered else grace
    if now - proc_started > window and now - newest > window:
        return "stalled"
    return None


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

    def reap_strays(self, wait=3.0):
        """Kill any ffmpeg still writing into OUR buffer directory, and wait for it to go.

        A detector that was force-killed leaves its recorder ffmpeg orphaned — re-parented
        to launchd, still holding an RTSP connection to the camera, and still writing
        seg_%Y%m%d_%H%M%S.ts into this very directory. Two writers sharing one naming
        pattern interleave, and save_event() concatenates whatever falls in the window, so
        clips come out spliced from two unsynchronised streams. Seen for real on
        2026-08-17: an orphan from 07:18 was still writing alongside the live recorder at
        07:46, alternating segments every two seconds.

        It has to happen BEFORE the buffer is wiped, or the orphan simply refills it.
        """
        try:
            ps = subprocess.run(["ps", "-axo", "pid=,command="],
                                capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        pids = stray_recorder_pids(ps, self.buffer_dir, me=os.getpid())
        for pid in pids:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    break                                  # already gone
                deadline = time.time() + wait / 3
                while time.time() < deadline:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        break                              # it died
                    time.sleep(0.1)
                else:
                    continue                               # still alive → next signal
                break
        if pids:
            print(f"[recorder] reaped {len(pids)} stray ffmpeg(s) writing to the buffer: "
                  f"{', '.join(str(p) for p in pids)}", flush=True)
        return pids

    def start(self):
        self.reap_strays()                                 # before the wipe, or it refills
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
            why = restart_reason(now, self._proc_started, newest, dead, stall)
            if why:
                if why == "stalled" and self.proc:
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
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(base))
    import h32env                                          # camera URL from local.env
    cfg = h32env.detector_config(os.path.join(base, "config.json"))
    rtsp = cfg["rtsp_camera_direct"] or cfg["rtsp_main"]
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
