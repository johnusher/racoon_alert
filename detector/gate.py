#!/usr/bin/env python3
"""
Is the gate open or shut?

A gate that never moves in the frame is a STATE question, not a detection question —
MegaDetector has no gate class and never will, and training a classifier for one bit
would need footage of an open gate that nobody has recorded yet. So this reads the
pixels of one fixed polygon, the gate's aperture, and asks a much narrower question:
is that patch still full of vertical bars?

That measure was picked by trying three on real frames (see test_gate.py for the table).
Counting the bars fails and so does looking for their regular spacing — a hedge has just
as many vertical edges as a wrought-iron gate and beats it on periodicity. What works is
the plain ratio of vertical to total edge energy: 0.68 on the gate, 0.43-0.47 on every
bar-free patch tried. It is also the cheapest of the three, and being a RATIO it
normalises illumination away, which is what killed the obvious alternative of diffing
against a stored "closed" photograph — a shadow crossing the gate defeats that, and
shadows cross this gate every sunny afternoon.

Only the UPPER band of the aperture is read, and that is not an optimisation. The whole
point of this module is to catch a 2-year-old opening the gate, and a 2-year-old standing
at the gate occludes the bottom of it — never the top. An adult occludes both, which is
why an ambiguous reading HOLDS the last state instead of guessing.

Two more rules that exist so this cannot cry wolf:
  • the first confident reading is adopted SILENTLY — restarting the detector in front of
    an open gate must not fire "the gate just opened";
  • a change needs min_frames of agreement, so one odd frame is never an opening.

⚠️ `closed_above` is provisional until somebody opens the gate in front of the camera:
the "open" end of the scale has only ever been measured on stand-in patches (pavement,
foliage, hedge), never on the real thing. `h32 gate calibrate` measures it properly.
"""
import cv2
import numpy as np

from roi import scale_poly

DAY_MIN_SAT = 12.0          # mean HSV saturation below this = infra-red = night


def is_daylight(frame, min_sat=DAY_MIN_SAT):
    """Is this a colour frame, or has the camera switched to infra-red?

    The gate rules are daytime-only — the toddler is indoors after dark, and the
    measure's night behaviour is unmeasured. Reading the picture rather than the clock
    means this follows the actual light instead of drifting with the season.
    """
    if frame is None or frame.size == 0:
        return False
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) >= min_sat


class GateWatcher:
    """Tracks one gate's open/closed state across frames.

    aperture   — the gate opening, [[x, y], ...] in zone_space
    read_band  — the fraction of the aperture's height to read, from the top; the
                 default upper 45% is what a small child cannot block
    closed_above / deadband — score above (closed_above + deadband) is closed, below
                 (closed_above - deadband) is open, and BETWEEN THEM nothing changes
    min_frames / min_secs — how much agreement a change of state needs
    """

    def __init__(self, aperture, zone_space=None, closed_above=0.57, deadband=0.04,
                 read_band=(0.0, 0.45), min_frames=4, min_secs=1.0):
        self.aperture = [list(p) for p in aperture] if aperture else None
        self.zone_space = list(zone_space) if zone_space else None
        self.closed_above, self.deadband = closed_above, deadband
        self.read_band = tuple(read_band)
        self.min_frames, self.min_secs = min_frames, min_secs
        self.state = None                  # None = unknown, then "closed" / "open"
        self.last_score = None
        self.changed_at = 0.0
        self._want, self._streak = None, 0
        self._cache = {}

    # ---- geometry ----
    def _mask(self, shape):
        """The read band of the aperture, as a boolean mask for this frame size."""
        h, w = shape[:2]
        key = (w, h)
        if key not in self._cache:
            pts = np.array(scale_poly(self.aperture, self.zone_space or [w, h], [w, h]),
                           np.int32)
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            span = max(1, y1 - y0)
            band = np.zeros((h, w), np.uint8)
            top = int(y0 + span * self.read_band[0])
            bot = int(y0 + span * self.read_band[1])
            cv2.fillPoly(band, [pts], 255)
            band[:max(0, top), :] = 0
            band[max(0, bot):, :] = 0
            # Pull in from the outline: the polygon's own edge is a step in the image
            # and would be counted as gate structure.
            band = cv2.erode(band, np.ones((7, 7), np.uint8))
            self._cache[key] = band.astype(bool)
        return self._cache[key]

    # ---- the measure ----
    def score(self, frame):
        """Vertical share of the edge energy in the read band. None if unreadable."""
        if frame is None or frame.size == 0 or not self.aperture:
            return None
        m = self._mask(frame.shape)
        if m.sum() < 200:                              # too small to mean anything
            return None
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
                             (0, 0), 1.2)
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))[m]
        gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))[m]
        tot = gx.mean() + gy.mean()
        if tot < 1e-3:                                 # a flat grey patch says nothing
            return None
        return float(gx.mean() / tot)

    # ---- state ----
    def update(self, frame, now):
        """Fold one frame in. Returns 'opened'/'closed' on a confirmed change, else None."""
        s = self.score(frame)
        self.last_score = s
        if s is None:
            return None
        if s >= self.closed_above + self.deadband:
            want = "closed"
        elif s <= self.closed_above - self.deadband:
            want = "open"
        else:
            return None                                # ambiguous: hold, do not guess
        if want == self.state:
            self._streak = 0
            return None
        self._streak = self._streak + 1 if want == self._want else 1
        self._want = want
        if self._streak < self.min_frames:
            return None
        first = self.state is None
        if not first and (now - self.changed_at) < self.min_secs:
            return None
        self.state, self.changed_at, self._streak = want, now, 0
        # Adopting the state we started in is not an event.
        return None if first else ("opened" if want == "open" else "closed")

    def describe(self):
        if not self.aperture:
            return "no gate configured"
        s = f"{self.last_score:.2f}" if self.last_score is not None else "—"
        band = f"{int(self.read_band[0] * 100)}-{int(self.read_band[1] * 100)}%"
        return (f"{self.state or 'unknown'} (score {s}, closed above "
                f"{self.closed_above:.2f}±{self.deadband:.2f}, top {band} of the aperture)")
