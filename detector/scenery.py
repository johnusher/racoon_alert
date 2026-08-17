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

--- the rock (2026-08-16, later the same day) -------------------------------------

The bench above has one convenient property: its box is *pixel-identical* frame to
frame. The rock in the corner of the same garden does not. It is big and irregular
and MegaDetector does not quite agree with itself about where its edges are, so the
box breathes by up to 28px while the rock, being a rock, does not move at all:

    object            box displacement*   verdict
    bench  09:26          0.000           a fixed threshold catches this
    rock   16:17          0.028
    rock   16:18          0.054
    rock   16:14          0.064           …and this is where a fixed threshold dies
    raccoon 03:53         0.062           real — and it moves LESS than the rock wobbles

That is the whole difficulty: the rock's wobble is larger than the raccoon's real
movement, so no single `min_move` can separate them. Worse, the wobble was
self-perpetuating — reading as movement, it both cleared the gate AND reset the
"nothing has moved here" clock, so the spot could never be written off as furniture;
and it pushed the box past `iou_match`, so one boulder sprawled into 30 anchors, each
born fresh and permissive. One of them was created 30 seconds before the first bad
event.

So the movement gate is no longer a constant. Each spot learns the wobble it actually
shows while nothing is happening there, and movement at that spot must beat its own
wobble. A spot we have only just noticed still gets the permissive `min_move` — which
is precisely what keeps the raccoon, seen twice in its life, coming through.

--- what this gate CANNOT do (2026-08-16 22:48) -------------------------------------

"Has it moved?" is a proxy for "is it alive?", and the proxy has a hole. A friend
walked outside, stared straight at the camera, and got no alert. MegaDetector caught
them in two frames about ten seconds apart at 0.31/0.35 — and because `track_gap_secs`
is 3.0, the second detection began a brand-new track whose `first_box` is itself. The
displacement measured is therefore 0.000 however far the person really walked, so
nothing here can ever confirm them. Their box had in fact travelled 0.187 of its own
size, NINE times `min_move`, and this file logged it as wobble.

No threshold fixes that: `min_move` cannot go below the 0.000 that was measured, and
raising `track_gap_secs` to bridge the gap re-opens the rock, whose gaps are the same
length. Trusting confidence instead means dropping `conf_certain` to 0.30, which is
squarely inside the bench's range. test_scenery.py section 9 pins all three dead ends.

The way out is to stop using a proxy: detect.py asks SpeciesNet whether the thing is
actually a person (`promote_unproven`), and furniture reads as `blank` so it still
cannot get through. Movement remains the cheap first answer — this gate runs on every
frame and the classifier does not.
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

    def __init__(self, path=None, iou_match=0.60, track_iou=0.30, min_move=0.02,
                 track_gap_secs=3.0, static_after_secs=180.0, min_sightings=5,
                 forget_secs=1800.0, conf_certain=0.70, conf_override=0.25,
                 jitter_slack=2.5, jitter_learn_secs=5.0, jitter_cap=0.25,
                 home_alpha=0.05, jitter_decay=0.99, autosave_secs=60.0):
        self.path = path
        self.iou_match = iou_match      # is this the same fixed spot? (must tolerate wobble)
        self.track_iou = track_iou      # loose: is this the same moving object?
        self.min_move = min_move
        self.jitter_slack = jitter_slack        # how far past its own wobble a spot must move
        self.jitter_learn_secs = jitter_learn_secs
        self.jitter_cap = jitter_cap            # a wobble bigger than this is not wobble
        self.home_alpha = home_alpha            # how fast a spot's resting box is re-learned
        self.jitter_decay = jitter_decay        # so a spot that settles down is trusted again
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
                    "moved_at": 0.0, "static": False, "static_conf": 0.0,
                    "home": None, "jitter": 0.0}
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

    # ---- what counts as movement here -------------------------------------

    def _gate(self, anchor, now):
        """How far a box must travel from where its track began to count as alive.

        A spot we have only just noticed is judged by `min_move`: we have no idea yet
        how much its box breathes, and being generous is what lets the raccoon — seen
        twice in its life — through. A spot we have watched long enough to know is
        judged against the wobble it actually shows, so the rock has to beat the rock.
        """
        if (anchor["sightings"] < self.min_sightings
                or now - anchor["first_seen"] < self.jitter_learn_secs):
            return self.min_move
        # The wobble is measured from where the box rests, so a swing from one extreme
        # to the other is twice that — hence the slack on top.
        return max(self.min_move, anchor["jitter"] * self.jitter_slack)

    def _settle(self, anchor, box, moved):
        """Learn where this spot rests and how much its box breathes while resting.

        Only detections that did NOT count as movement teach it. Otherwise a person
        walking across the anchor would inflate its wobble envelope on the way past
        and leave the spot blind behind them.
        """
        home = anchor["home"] or [float(v) for v in box]
        if not moved:
            home = [h + (b - h) * self.home_alpha for h, b in zip(home, box)]
            wobble = _displacement(box, home)
            anchor["jitter"] = min(self.jitter_cap,
                                   max(wobble, anchor["jitter"] * self.jitter_decay))
        anchor["home"] = home
        # Match on where the box rests, not on wherever we happened to first see it —
        # otherwise the wobble walks the box out of its own anchor and it starts again.
        anchor["box"] = [int(round(v)) for v in home]
        self._dirty = True

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
                          "static": False, "static_conf": 0.0,
                          "home": [float(v) for v in box], "jitter": 0.0}
                self.anchors.append(anchor)
            anchor["last_seen"] = now
            anchor["sightings"] += 1
            anchor["max_conf"] = max(anchor["max_conf"], conf)
            self._dirty = True

            certain = conf >= self.conf_certain
            if anchor["static"] and conf <= anchor["static_conf"] + self.conf_override:
                self._settle(anchor, box, moved=False)   # keep learning where it rests
                if certain:
                    # Confident — but at a spot already written off as furniture, and a
                    # global confidence bar cannot arbitrate that: MegaDetector holds
                    # 0.69-0.80 on the stone trough it has been wrong about all morning,
                    # so `certain` alone handed that furniture a permanent override and
                    # it kept firing even once learned. Neither act on it nor bin it —
                    # leave it unproven, which is the one state detect.py refers to
                    # SpeciesNet (promote_unproven). That fires on a human or a named
                    # species and on nothing else, so a person who really is standing at
                    # the trough is still announced while the trough itself is not.
                    unproven.append((cls, conf, box))
                else:
                    suppressed.append((cls, conf, box, "scenery"))
                continue

            track = self._match_track(cls, box, now)
            if track is None:
                track = {"cls": cls, "box": list(box), "first_box": list(box),
                         "last_seen": now, "moved": False}
                self.tracks.append(track)
            track["box"] = list(box)
            track["last_seen"] = now
            displaced = (_displacement(box, track["first_box"])
                         >= self._gate(anchor, now))
            if not track["moved"]:
                track["moved"] = certain or displaced

            if track["moved"]:
                confirmed.append((cls, conf, box))
                # Only real displacement un-learns a spot. A confident detection is
                # evidence that something is THERE, never that it moved — and on this
                # camera MegaDetector is confidently WRONG about fixed objects: it held
                # `person` 0.69-0.80 on the stone trough at [1263,568,1653,1070] that had
                # not shifted a pixel in 100 minutes (2026-08-17 07:37-07:47, 15 events).
                # Letting that reset the clock restarted static_after_secs every few
                # seconds, so the spot could never be written off as furniture: 2470
                # sightings, jitter 0.013, and still `static: false` in scenery.json.
                if displaced:
                    anchor["moved_at"] = now    # something real happens here
                    anchor["static"] = False
            else:
                unproven.append((cls, conf, box))

            self._settle(anchor, box, track["moved"])
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
        worst = max((a["jitter"] for a in self.anchors), default=0.0)
        return (f"{len(st)} scenery spot(s) learned, {len(self.anchors)} tracked, "
                f"worst wobble {worst:.3f}")
