#!/usr/bin/env python3
"""
Link watch — is the camera actually reachable, and how well?

Why this exists rather than a WiFi signal bar: **these cameras do not report their own
signal strength.** The 2.5K Vimtag advertises `Dot11Configuration=false` in its ONVIF
capabilities and answers `GetDot11Status` with nothing at all, so there is no RSSI to
read. Drawing a signal-bars icon would mean inventing a number.

What CAN be measured is the thing that actually matters: whether packets get from here
to the camera, and how long they take. That covers the whole path — camera radio, the
access point, the LAN — which is what "the camera keeps dropping out" is really about.
It also distinguishes the two failures that look identical on the monitor:

  * the camera is off the network entirely (ping fails, ARP goes incomplete), versus
  * the camera is reachable but the video stalls — which points at the stream, the
    session limit or the decoder, NOT the radio.

Runs one subprocess ping every `period` seconds on a daemon thread; that is ~nothing
next to MegaDetector's 84 ms/frame, and it deliberately does NOT touch the camera's
RTSP or ONVIF ports — probing those on a fragile camera is how the Victure got rebooted
once already (see TODO.md).
"""
import subprocess
import threading
import time
from collections import deque


def ping_once(host, timeout=2.0):
    """Round-trip time in ms, or None if there was no reply.

    Uses a subprocess timeout rather than ping's own -W, whose units differ between
    macOS (milliseconds) and Linux (seconds) — the Pi is the eventual target.
    """
    try:
        p = subprocess.run(["ping", "-c", "1", "-n", host],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if p.returncode != 0:
        return None
    for tok in p.stdout.split():
        if tok.startswith("time="):
            try:
                return float(tok[5:])
            except ValueError:
                return None
    return None


class LinkMonitor:
    """Rolling reachability for one camera.

    `quality` is deliberately coarse — good / fair / poor / down. A number that jitters
    between 8 ms and 15 ms invites tuning that changes nothing; what a person needs from
    across the room is whether this camera is fine, struggling, or gone.
    """

    GOOD_RTT, FAIR_RTT = 60.0, 150.0      # ms
    GOOD_LOSS, FAIR_LOSS = 2.0, 20.0      # percent over the window

    def __init__(self, host, period=5.0, window=12, pinger=ping_once):
        self.host = host or ""
        self.period = period
        self.window = window
        self._ping = pinger
        self._samples = deque(maxlen=window)   # rtt in ms, or None for a lost ping
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self.dropouts = 0                      # up -> down transitions since start
        self.down_since = None                 # epoch, or None while it is up
        self.last_ok = None                    # epoch of the last successful reply

    # ---- sampling ----

    def sample(self, now=None):
        """Take one measurement. Called by the thread; called directly by tests."""
        now = time.time() if now is None else now
        rtt = self._ping(self.host) if self.host else None
        with self._lock:
            was_up = self._up_locked()
            self._samples.append(rtt)
            if rtt is None:
                # Only count a dropout once per outage, not once per lost ping, or a
                # five-minute outage reads as sixty separate incidents.
                if was_up and not self._up_locked():
                    self.dropouts += 1
                    self.down_since = now
            else:
                self.last_ok = now
                self.down_since = None
        return rtt

    def _up_locked(self):
        """Up = at least one reply in the window. One lost ping is not an outage."""
        if not self._samples:
            return True                        # no evidence yet: do not cry wolf
        return any(s is not None for s in self._samples)

    # ---- thread ----

    def start(self):
        if self._thread or not self.host:
            return self
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._run:
            try:
                self.sample()
            except Exception:
                pass                           # a link watcher must never take detection down
            time.sleep(self.period)

    def stop(self):
        self._run = False

    # ---- reporting ----

    def status(self):
        with self._lock:
            samples = list(self._samples)
            dropouts, down_since, last_ok = self.dropouts, self.down_since, self.last_ok
        if not self.host:
            return {"host": "", "quality": "unknown", "up": None, "rtt_ms": None,
                    "loss_pct": None, "samples": 0, "dropouts": 0, "down_secs": None}

        got = [s for s in samples if s is not None]
        up = bool(got) if samples else None
        loss = round(100.0 * (len(samples) - len(got)) / len(samples), 1) if samples else None
        rtt = round(sum(got) / len(got), 1) if got else None

        if not samples:
            quality = "unknown"
        elif not got:
            quality = "down"
        elif loss > self.FAIR_LOSS or rtt > self.FAIR_RTT:
            quality = "poor"
        elif loss > self.GOOD_LOSS or rtt > self.GOOD_RTT:
            quality = "fair"
        else:
            quality = "good"

        return {"host": self.host, "quality": quality, "up": up, "rtt_ms": rtt,
                "loss_pct": loss, "samples": len(samples), "dropouts": dropouts,
                "down_secs": round(time.time() - down_since) if down_since else None,
                "last_ok": last_ok}

    def describe(self):
        s = self.status()
        if s["quality"] == "unknown":
            return "link: no host to watch" if not s["host"] else "link: measuring…"
        if s["quality"] == "down":
            d = f", down {s['down_secs']}s" if s["down_secs"] is not None else ""
            return f"link: {s['host']} DOWN ({s['dropouts']} dropout(s){d})"
        return (f"link: {s['host']} {s['quality']} — {s['rtt_ms']}ms, "
                f"{s['loss_pct']}% loss, {s['dropouts']} dropout(s)")
