#!/usr/bin/env python3
"""
Tests for the frame masks (roi.py).

Written against a real false positive: the gate VIMTAG sits indoors on a desk shooting
through a window at a steep angle, so a wedge of the sensor is window sill. MegaDetector
reads that smooth diagonal slab as `animal` — above the 0.20 animal threshold — and on
2026-08-18 it fired ANIMAL every 30 s all day.

The polygon under test is READ FROM web/cameras.json, not copied into here, so the thing
being checked is the thing that actually ships. The boxes and ground points below are
measured off real frames from the camera, and they are the part that goes stale: they
describe where the sill and the path were on the day. That is not a flaw in the test, it
is the point of it — the camera was moved on 2026-08-18 between one restart and the next,
the sill shrank to a third of its width and got LOUDER as it shrank (0.27-0.52 before the
move, 0.70-0.82 after), and the polygon drawn an hour earlier was suddenly covering the
brick path and the flower bed. If this file fails after the camera is touched, re-measure
the frames and redraw the polygon; do not widen the tolerances.

The two things that must hold:
  • the mask eats every box the detector really produced over the sill, and eats none of
    the ground an animal could really stand on;
  • the polygon is stated in zone_space and SCALED to the live frame, so turning the
    camera's resolution down does not move the mask.

Run:  ../.venv/bin/python detector/test_roi.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
import h32env
from roi import FrameMask, scale_poly

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def box_at(foot, w=200, h=300):
    """A box whose foot point (bottom centre) is exactly `foot`."""
    x, y = foot
    return [x - w // 2, y - h, x + w // 2, y]


# ---- the shipped mask, and the frames it was drawn against -------------------------
GATE = h32env.detector_config(camera="gate")
SILL, SPACE = GATE.get("exclude_roi"), GATE.get("zone_space")
# Every box MegaDetector put on the sill in the 17:45 view, run over raw buffer segments
# with the mask off (detector/buffer/gate/seg_2026081817464*.ts).
SILL_BOXES = [[0, 319, 913, 1440], [0, 312, 930, 1439], [0, 321, 915, 1440],
              [0, 318, 915, 1440], [0, 314, 912, 1440]]
# Ground in those same frames where an animal genuinely could stand. The first three sit
# closest to the mask edge and are the ones that matter.
GROUND = {"path just right of the sill": (700, 1200), "path near the sill, low": (750, 1400),
          "brick path bottom": (900, 1350), "path in front of the gate": (900, 950),
          "the gate foot": (1150, 1000), "flower bed": (1050, 1250),
          "far pavement": (1500, 900), "grass right": (1300, 1300)}


print("\n1. no mask configured — nothing is filtered")
m = FrameMask()
check("empty mask allows anything", m.allows([0, 0, 10, 10], (2560, 1440)))
check("empty mask is falsy", not m)

print("\n2. exclude_roi drops a box, and it is the FOOT that decides")
m = FrameMask(exclude=[[0, 0], [100, 0], [100, 100], [0, 100]], zone_space=[200, 200])
check("foot inside the mask is dropped", not m.allows(box_at((50, 90), 40, 40), (200, 200)))
check("foot outside the mask is kept", m.allows(box_at((150, 90), 40, 40), (200, 200)))
check("a box whose middle is in the mask but whose feet are not is KEPT",
      m.allows([20, 20, 80, 150], (200, 200)), "feet at (50,150), below the mask")
check("a box whose middle is outside but whose feet are in is DROPPED",
      not m.allows([20, -80, 80, 90], (200, 200)), "feet at (50,90), inside the mask")

print("\n3. roi keeps only what stands inside it")
m = FrameMask(roi=[[0, 0], [100, 0], [100, 100], [0, 100]], zone_space=[200, 200])
check("foot inside the roi is kept", m.allows(box_at((50, 90), 40, 40), (200, 200)))
check("foot outside the roi is dropped", not m.allows(box_at((150, 90), 40, 40), (200, 200)))

print("\n4. polygons are scaled from zone_space to the live frame")
check("scale_poly halves both axes",
      scale_poly([[0, 0], [100, 200]], [200, 400], [100, 200]) == [[0, 0], [50, 100]])
check("scale_poly is a no-op at the same size",
      scale_poly([[7, 9], [11, 13]], [200, 400], [200, 400]) == [[7, 9], [11, 13]])
check("scale_poly carries negative coordinates through",
      scale_poly([[-100, -100]], [2560, 1440], [1280, 720]) == [[-50, -50]])
m = FrameMask(exclude=[[0, 0], [100, 0], [100, 100], [0, 100]], zone_space=[200, 200])
check("the same spot is masked on a half-size frame",
      not m.allows(box_at((25, 45), 20, 20), (100, 100)), "(50,90) in zone space")
check("and the spot beyond it is still not",
      m.allows(box_at((75, 45), 20, 20), (100, 100)), "(150,90) in zone space")

print("\n5. the mask web/cameras.json actually ships, on the resolution it was drawn at")
check("the gate camera has a mask at all", bool(SILL) and bool(SPACE),
      f"exclude_roi={SILL} zone_space={SPACE}")
m = FrameMask(exclude=SILL, zone_space=SPACE)
leaked = [b for b in SILL_BOXES if m.allows(b, (2560, 1440))]
check("every sill box the detector produced is suppressed", not leaked, f"leaked: {leaked}")
masked = [k for k, pt in GROUND.items() if not m.allows(box_at(pt), (2560, 1440))]
check("no real ground is masked", not masked, f"wrongly masked: {masked}")
# The margins are the early warning. If either collapses, the camera has moved.
foot = (int((SILL_BOXES[0][0] + SILL_BOXES[0][2]) / 2), SILL_BOXES[0][3])
import cv2, numpy as np                                     # noqa: E402  (only for margins)
poly = np.array(m.polygons((2560, 1440))[1], np.int32)
inside = cv2.pointPolygonTest(poly, foot, True)
clear = min(abs(cv2.pointPolygonTest(poly, pt, True)) for pt in GROUND.values())
check("the sill's foot sits well inside the mask", inside > 60, f"{inside:.0f}px")
check("the nearest real ground sits well outside it", clear > 60, f"{clear:.0f}px")

print("\n6. the same mask survives the camera being turned down to a lower resolution")
for w, h in [(1920, 1080), (1280, 720)]:
    sx, sy = w / 2560, h / 1440
    leaked = [b for b in SILL_BOXES
              if m.allows([int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)], (w, h))]
    check(f"sill still suppressed at {w}x{h}", not leaked, f"leaked: {leaked}")
    masked = [k for k, (x, y) in GROUND.items()
              if not m.allows(box_at((int(x * sx), int(y * sy)), int(200 * sx), int(300 * sy)), (w, h))]
    check(f"ground still detected at {w}x{h}", not masked, f"wrongly masked: {masked}")

print("\n7. 6 passes for a weaker reason than it looks — say so out loud")
# The sill mask is a wedge in the corner of a 16:9 frame, so scaling both axes by the
# same factor very nearly maps it to itself. It would have survived with no scaling at
# all. Do not read 6 as evidence that raw-pixel polygons are safe — 8 is that test.
unscaled = FrameMask(exclude=SILL, zone_space=[1280, 720])
sx, sy = 1280 / 2560, 720 / 1440
leaked = [b for b in SILL_BOXES
          if unscaled.allows([int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)],
                             (1280, 720))]
check("the sill mask happens to be near scale-invariant (a corner wedge)", not leaked,
      "so 6 is necessary but not sufficient — 8 is the real test")

print("\n8. a polygon NOT in the corner needs the scaling, or it lands somewhere else")
SPOT = [[1900, 1100], [2100, 1100], [2100, 1300], [1900, 1300]]   # e.g. a bird feeder
right = FrameMask(exclude=SPOT, zone_space=[2560, 1440])
wrong = FrameMask(exclude=SPOT, zone_space=[1280, 720])           # same points, raw pixels
foot = box_at((1000, 600))                                        # the spot at 1280x720
check("scaled: the spot is masked on the smaller frame", not right.allows(foot, (1280, 720)))
check("unscaled: the very same spot is missed", wrong.allows(foot, (1280, 720)),
      "the mask sits off the bottom-right of a 1280x720 frame entirely")

print("\n9. scaled polygons are cached per frame size and never mutate the source")
m2 = FrameMask(exclude=SILL, zone_space=SPACE)
check("repeat calls agree", m2.polygons((1920, 1080))[1] == m2.polygons((1920, 1080))[1])
check("the drawn-at size comes back untouched", m2.polygons((2560, 1440))[1] == SILL)
check("the source polygon is not mutated by scaling", m2.exclude == SILL)

print("\n10. the other cameras are not masked by the gate's sill")
west = h32env.detector_config(camera="west")
wm = FrameMask(west.get("roi"), west.get("exclude_roi"), west.get("zone_space"))
check("west has no mask", not wm, wm.describe())
check("west therefore filters nothing", wm.allows([0, 319, 913, 1080], (1920, 1080)))

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all roi tests passed")
