#!/usr/bin/env python3
"""
Make a noise on THIS machine, right now.

Separate from notify.py (e-mail, minutes) and from talk.py (the camera's own speaker,
which is a Victure-only path — the VIMTAG on the gate has no proven audio-out, so the
gate camera cannot speak). What is left for something that has to be heard immediately is
the Mac the detector is running on.

It is deliberately dumb: a system sound repeated, then `say`. No new dependency, no
network, no cloud — which matters, because the whole point of h32 is that nothing leaves
the LAN, and an alarm that phones a push service would be the one thing that does.

Rate-limited, because an alarm that machine-guns is an alarm you turn off. Non-blocking,
because the detection loop must not stall behind a sound: a stalled loop is a camera that
stops watching the gate, which is the exact opposite of what this is for.
"""
import shutil
import subprocess
import threading
import time

SOUND = "/System/Library/Sounds/Sosumi.aiff"
# English, not the German the camera greets visitors in. talk.py says "Hallo." to whoever
# is at the gate, in Berlin, and that is right; this is shouted across the room at the
# household, and a German voice reading an English alarm is harder to catch, not easier.
VOICE = "Daniel"


def _run(cmd):
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass                                    # a silent alarm must never take detection down


class Alarm:
    """Immediate local noise, rate-limited and off the detection thread."""

    def __init__(self, enabled=True, sound=SOUND, voice=VOICE, repeat=3,
                 min_gap_secs=20.0, runner=_run):
        self.enabled = enabled
        self.sound, self.voice, self.repeat = sound, voice, repeat
        self.min_gap = min_gap_secs
        self._run = runner
        self._last = {}
        self.available = bool(shutil.which("afplay") or shutil.which("say"))

    def fire(self, text, key="alarm", now=None):
        """Sound the alarm and speak `text`. Returns True if it actually fired."""
        now = time.time() if now is None else now
        if not (self.enabled and self.available):
            return False
        if now - self._last.get(key, 0.0) < self.min_gap:
            return False                        # still ringing from the last one
        self._last[key] = now
        threading.Thread(target=self._play, args=(text,), daemon=True).start()
        return True

    def _play(self, text):
        for _ in range(max(1, self.repeat)):
            self._run(["afplay", self.sound])
        if text:
            self._run(["say", "-v", self.voice, text])

    def describe(self):
        if not self.enabled:
            return "off"
        if not self.available:
            return "off (no afplay/say — not a Mac?)"
        return f"{self.repeat}x {self.sound.rsplit('/', 1)[-1]} then say ({self.voice}), ≤1/{self.min_gap:.0f}s"
