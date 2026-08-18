#!/usr/bin/env python3
"""
Named regions of the picture, and who is standing in them.

The third member of a family: roi.py says which detections are real at all, scenery.py
says which are furniture, and this says WHERE a real one is. The schema has been sitting
in web/cameras.json since the multi-camera work — `zones`, `rules`, `zone_space` — with
nothing reading it; this is the reader for `zones`.

A detection is in a zone when its FEET are, not its centre. That is the same convention
as roi.py and for the same reason: the bottom-centre of a box is where the thing is
actually standing, and a person leaning over a gate has a box that overlaps half the
scene while their feet stay put.

Polygons are in the camera's `zone_space` and scaled to the live frame, so a camera
turned down to a lower resolution keeps its zones. `rules` are deliberately NOT
implemented here. There are exactly two behaviours wanted — the gate opening, and the
gate opening with a small child at it — and both live in detect.py where they can be
read and audited in one place. A rule DSL for two rules would be harder to check and
easier to get subtly wrong, which is the wrong trade for something meant to catch a
toddler reaching the street.
"""
import cv2
import numpy as np

from roi import scale_poly


class Zones:
    """The named polygons for one camera."""

    def __init__(self, zones=None, zone_space=None):
        self.zones = [z for z in (zones or []) if z.get("poly")]
        self.zone_space = list(zone_space) if zone_space else None
        self._cache = {}

    def __bool__(self):
        return bool(self.zones)

    def __len__(self):
        return len(self.zones)

    def polygons(self, frame_wh):
        """[(id, name, points)] in this frame's own pixels."""
        key = tuple(frame_wh)
        if key not in self._cache:
            space = self.zone_space or list(frame_wh)
            self._cache[key] = [(z.get("id", f"zone{i}"), z.get("name", z.get("id", "")),
                                 scale_poly(z["poly"], space, frame_wh))
                                for i, z in enumerate(self.zones)]
        return self._cache[key]

    @staticmethod
    def foot(box):
        """Where the thing is standing: bottom centre of the box."""
        return (int((box[0] + box[2]) / 2), int(box[3]))

    def at(self, box, frame_wh):
        """Ids of every zone this detection is standing in."""
        if not self.zones:
            return []
        pt = self.foot(box)
        return [zid for zid, _, pts in self.polygons(frame_wh)
                if cv2.pointPolygonTest(np.array(pts, np.int32), pt, False) >= 0]

    def contains(self, zone_id, box, frame_wh):
        return zone_id in self.at(box, frame_wh)

    def describe(self):
        if not self.zones:
            return "no zones"
        return ", ".join(z.get("id", "?") for z in self.zones)
