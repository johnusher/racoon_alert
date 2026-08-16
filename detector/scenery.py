#!/usr/bin/env python3
"""
Scenery filter — tells living things apart from garden furniture.

MegaDetector reliably fires low-confidence `person` boxes at static objects: on this
camera a stone bench scored 0.30–0.51 and produced 56 bogus PERSON events in a day.
Confidence alone cannot separate them — the real person who walked past at 00:32
scored 0.39 in the frame that triggered his event, *below* the bench.

What does separate them is movement. Measured over the recorded clips:

    object            detections   box displacement*   verdict
    bench  09:26          4 / 108        0.000         same 4 pixel coords, 31 min apart
    bench  08:55          6 / 108        0.009
    raccoon 03:53         2 / 120        0.062         real
    person  00:32        26 / 127        0.030 … 1.4   real

    *max distance the box centre travels from where the track started,
     divided by sqrt(box area) — i.e. movement in units of the object's own size.

So two gates, either of which alone would have killed every false event of that day:

  movement gate  a detection may only TRIGGER an event once its track has actually
                 moved. Unproven detections are still reported and still count toward
                 the hit window — they just cannot fire on their own. That distinction
                 matters: the raccoon was only ever seen in two frames, so a gate that
                 swallowed the first one would have lost it.

  scenery memory a location that keeps producing detections that never move becomes
                 furniture, and is then suppressed outright. Remembered across
                 restarts, forgotten once it stops being confirmed, and never applied
                 to a spot where something has genuinely moved.

Both gates step aside for a confident detection (`conf_certain`), so an unmistakable
person is never held back, wherever they stand.
"""
import json, math, os, time


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _scale(b):
    """Object size in pixels — the yardstick displacement is measured against."""
    return max(1.0, math.sqrt(max(1, (b[2] - b[0])) * max(1, (b[3] - b[1]))))


def _displacement(box, ref):
    (cx, cy), (rx, ry) = _centre(box), _centre(ref)
    return math.hypot(cx - rx, cy - ry) / _scale(ref)


class SceneryFilter:
    """Sorts detections into: act on it, believe it but wait, or it is furniture.

    apply() returns (confirmed, unproven, suppressed).
      confirmed — has moved (or is confident enough to take on trust); may fire an event
      unproven  — real as far as we know, but has not moved yet; shown and counted,
                  cannot fire on its own
      suppressed— known scenery; carries the reason so the monitor can grey it out
                  rather than making detections vanish with no explanation
    """

    def __init__(self, path=None, iou_match=0.85, track_iou=0.30, min_move=0.02,
                 track_gap_secs=3.0, static_after_secs=180.0, min_sightings=5,
                 forget_secs=1800.0, conf_certain=0.70, conf_override=0.25,
                 autosave_secs=60.0):
        self.path = path
        self.iou_match = iou_match      # tight: is this the same fixed spot?
        self.track_iou = track_iou      # loose: is this the same moving object?
        self.min_move = min_move
        self.track_gap_secs = track_gap_secs
        self.static_after_secs = static_after_secs
        self.min_sightings = min_sightings
        self.forget_secs = forget_secs
        self.conf_certain = conf_certain
        self.conf_override = conf_override
        self.autosave_secs = autosave_secs
        self.anchors = []          # long-lived: candidate furniture
        self.tracks = []           # short-lived: what is on screen right now
        self._dirty = False
        self._saved_at = 0.0
        self.load()

    # ---- persistence ------------------------------------------------------
    # Anchors are pruned by apply(), never by load(): the file may be older than
    # the process and only a real timestamp from the caller can age it correctly.

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        defaults = {"first_seen": 0.0, "last_seen": 0.0, "sightings": 0, "max_conf": 0.0,
                    "moved_at": 0.0, "static": False, "static_conf": 0.0}
        for a in data.get("anchors", []):
            if "box" not in a or "cls" not in a:
                continue
            self.anchors.append({**defaults, **a})

    def save(self):
        if not self.path:
            return
        tmp = f"{self.path}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump({"anchors": self.anchors}, fh, indent=1)
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError:
            pass

    # ---- matching ---------------------------------------------------------

    def _match_track(self, cls, box, now):
        best, best_iou = None, self.track_iou
        for t in self.tracks:
            if t["cls"] != cls or now - t["last_seen"] > self.track_gap_secs:
                continue
            v = _iou(box, t["box"])
            if v >= best_iou:
                best, best_iou = t, v
        return best

    def _match_anchor(self, cls, box):
        best, best_iou = None, self.iou_match
        for a in self.anchors:
            if a["cls"] != cls:
                continue
            v = _iou(box, a["box"])
            if v >= best_iou:
                best, best_iou = a, v
        return best

    def _prune(self, now):
        before = len(self.anchors)
        self.anchors = [a for a in self.anchors if now - a["last_seen"] <= self.forget_secs]
        if len(self.anchors) != before:
            self._dirty = True
        self.tracks = [t for t in self.tracks
                       if now - t["last_seen"] <= max(self.track_gap_secs, 10.0)]

    # ---- the gates --------------------------------------------------------

    def apply(self, dets, now=None):
        """dets: [(cls, conf, box)] → (confirmed, unproven, suppressed), the last as
        [(cls, conf, box, reason)]."""
        now = time.time() if now is None else now
        self._prune(now)
        confirmed, unproven, suppressed = [], [], []

        for cls, conf, box in dets:
            anchor = self._match_anchor(cls, box)
            if anchor is None:
                anchor = {"cls": cls, "box": list(box), "first_seen": now, "last_seen": now,
                          "sightings": 0, "max_conf": 0.0, "moved_at": 0.0,
                          "static": False, "static_conf": 0.0}
                self.anchors.append(anchor)
            anchor["last_seen"] = now
            anchor["sightings"] += 1
            anchor["max_conf"] = max(anchor["max_conf"], conf)
            self._dirty = True

            certain = conf >= self.conf_certain
            if anchor["static"] and not certain and conf <= anchor["static_conf"] + self.conf_override:
                suppressed.append((cls, conf, box, "scenery"))
                continue

            track = self._match_track(cls, box, now)
            if track is None:
                track = {"cls": cls, "box": list(box), "first_box": list(box),
                         "last_seen": now, "moved": False}
                self.tracks.append(track)
            track["box"] = list(box)
            track["last_seen"] = now
            if not track["moved"]:
                track["moved"] = certain or _displacement(box, track["first_box"]) >= self.min_move

            if track["moved"]:
                confirmed.append((cls, conf, box))
                anchor["moved_at"] = now        # something real happens here
                anchor["static"] = False
            else:
                unproven.append((cls, conf, box))

            self._promote(anchor, now)

        if self._dirty and self.autosave_secs and now - self._saved_at >= self.autosave_secs:
            self._saved_at = now
            self.save()
        return confirmed, unproven, suppressed

    def _promote(self, anchor, now):
        """A spot detected on and off for ages without anything ever moving there is furniture.

        The clock runs from the last time something did move through it, so a person
        walking past the bench un-learns it for a while rather than for good.
        """
        if anchor["static"]:
            return
        since = max(anchor["first_seen"], anchor["moved_at"])
        if (anchor["sightings"] >= self.min_sightings
                and now - since >= self.static_after_secs):
            anchor["static"] = True
            anchor["static_conf"] = anchor["max_conf"]
            self._dirty = True

    # ---- introspection ----------------------------------------------------

    def describe(self):
        st = [a for a in self.anchors if a["static"]]
        return f"{len(st)} scenery spot(s) learned, {len(self.anchors)} tracked"
