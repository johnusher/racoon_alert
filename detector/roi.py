#!/usr/bin/env python3
"""
Frame masks: the parts of the picture the detector must never look at.

This is NOT scenery.py. Scenery LEARNS the garden furniture MegaDetector keeps calling a
person — a bench or a stone trough is a real thing in the watched scene, a real animal
can stand in front of it, and so a learned spot keeps an override for the day it is
genuinely wrong. A mask is the opposite claim: this region is not part of the scene at
all and nothing can ever legitimately be there, so there is nothing to learn and no
override to keep.

The gate camera is why. It sits indoors on a desk shooting through a window at a steep
angle, so the lower-left third of the sensor is window sill. MegaDetector reads that big
smooth diagonal slab as `animal` at 0.27-0.52 — over the 0.20 animal threshold — and on
2026-08-18 it fired ANIMAL every 30 s from dawn onwards. Scenery could not save it and
never will: the box drifts with the light (its top edge wandered 185→317 px over the
day), every drift resets the static clock, and after 346 sightings the anchor was still
`static: false`. SpeciesNet read it `blank` 0.64-0.98, which is what stopped the sill
being NAMED, but a generic ANIMAL event fired anyway. A mask ends it in one line, and it
also keeps the sill away from scenery's confidence override — the slab reached 0.52 on
its own, and glare could plausibly take it past a learned spot's override bar.

A box is judged by its FOOT — bottom centre, where the thing would actually be standing
— exactly as the zone rules in web/cameras.json are. A raccoon on the path behind the
sill is still detected; the sill itself never is.

Polygons are `[[x, y], ...]` in the camera's `zone_space` (web/cameras.json), NOT in raw
frame pixels, and are scaled to whatever the stream is currently sending — the same
promise web/cameras.json already makes for its zones. That matters because these cameras
get turned down to a lower resolution to survive the 2.4 GHz cell (see TODO.md), and a
raw-pixel polygon would then be masking the wrong part of the picture with nothing in
the log to say so. Note the sill polygon below would have got away with it: its apex is
on the origin, and a line through the origin maps to itself when both axes scale by the
same factor, so it is very nearly invariant across 16:9 resolutions. That is luck, not
design — any polygon drawn around a THING rather than a corner lands somewhere else
entirely (see test_roi.py §8). Scaling is per axis, exact for a resolution change at the
same aspect ratio; a camera switched to a different aspect ratio has a different field
of view and its polygons have to be redrawn regardless.

A polygon may run outside the frame, and the sill one deliberately does. A foot point
lands on the very bottom row of the sensor often enough (y == frame height) that a
polygon ending exactly there decides the case on a boundary tie.
"""
import cv2
import numpy as np


def scale_poly(points, frm, to):
    """Scale `[[x, y], ...]` from the space it was drawn in to the space in use."""
    (fw, fh), (tw, th) = frm, to
    if (fw, fh) == (tw, th):
        return [[int(x), int(y)] for x, y in points]
    sx, sy = tw / fw, th / fh
    return [[int(round(x * sx)), int(round(y * sy))] for x, y in points]


class FrameMask:
    """Which detections survive, given an optional keep-region and an ignore-region.

    roi     — if given, a detection is kept only when its foot is INSIDE this polygon.
    exclude — a detection is dropped when its foot is inside this polygon.

    Both are in `zone_space` and are scaled per frame size on first use. A detection
    standing exactly on a polygon's edge counts as inside it, for both.
    """

    def __init__(self, roi=None, exclude=None, zone_space=None):
        self.roi = [list(p) for p in roi] if roi else None
        self.exclude = [list(p) for p in exclude] if exclude else None
        self.zone_space = list(zone_space) if zone_space else None
        self._cache = {}

    def __bool__(self):
        return bool(self.roi or self.exclude)

    def polygons(self, frame_wh):
        """(roi, exclude) as plain point lists in this frame's own pixels."""
        key = tuple(frame_wh)
        if key not in self._cache:
            space = self.zone_space or list(frame_wh)
            self._cache[key] = (scale_poly(self.roi, space, frame_wh) if self.roi else None,
                                scale_poly(self.exclude, space, frame_wh) if self.exclude else None)
        return self._cache[key]

    def arrays(self, frame_wh):
        """The same polygons as int32 arrays, for cv2 to draw or test against."""
        return tuple(np.array(p, np.int32) if p else None for p in self.polygons(frame_wh))

    def allows(self, box, frame_wh):
        """Is this detection in a part of the frame we are willing to believe?"""
        if not self:
            return True
        roi, exclude = self.arrays(frame_wh)
        foot = (int((box[0] + box[2]) / 2), int(box[3]))
        if roi is not None and cv2.pointPolygonTest(roi, foot, False) < 0:
            return False
        if exclude is not None and cv2.pointPolygonTest(exclude, foot, False) >= 0:
            return False
        return True

    def describe(self):
        n_roi = len(self.roi) if self.roi else 0
        n_ex = len(self.exclude) if self.exclude else 0
        if not self:
            return "whole frame"
        parts = []
        if n_roi:
            parts.append(f"roi {n_roi} pts")
        if n_ex:
            parts.append(f"ignoring {n_ex} pts")
        space = "x".join(str(v) for v in (self.zone_space or []))
        return ", ".join(parts) + (f" @{space}" if space else "")
