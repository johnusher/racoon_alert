#!/usr/bin/env python3
"""
Tests for the frame masks (roi.py).

Written against a real false positive: the gate VIMTAG sits indoors on a desk shooting
through a window at a steep angle, so the lower-left third of the sensor is window sill.
MegaDetector reads that big smooth diagonal slab as `animal` 0.27-0.52 — above the 0.20
animal threshold — and on 2026-08-18 it fired ANIMAL every 30 s all day.

The two things that must hold:
  • the sill polygon eats every box the detector actually produced over the sill, and
    eats none of the ground an animal could really stand on;
  • the polygon is stated in zone_space and SCALED to the live frame, so lowering the
    camera's resolution does not silently move the mask off the sill.

Run:  ../.venv/bin/python detector/test_roi.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


# The real mask, and the real numbers it was measured against (see roi.py).
SILL = [[-100, -100], [-100, 1600], [1383, 1600]]
SPACE = [2560, 1440]
# every box MegaDetector put on the sill on 2026-08-18 (events/gate/events.log + the
# scenery anchor it could never learn)
SILL_BOXES = [[3, 255, 1290, 1438], [1, 199, 1288, 1440], [1, 185, 1288, 1439],
              [10, 286, 1271, 1437], [23, 311, 1382, 1439], [39, 317, 1396, 1438],
              [28, 313, 1383, 1437], [23, 304, 1379, 1437]]
# ground in the same frames where an animal genuinely could stand
GROUND = {"path at the gate foot": (1700, 1050), "brick path near": (1500, 1250),
          "path bottom right": (1600, 1420), "grass mid": (1400, 1350),
          "path left of the gate": (1300, 1150), "soil by the sill edge": (1350, 1440),
          "far path": (1900, 900), "the bench": (1750, 1150)}


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
check("a frame the size of zone_space is not scaled at all",
      m.polygons((200, 200))[1] == [[0, 0], [100, 0], [100, 100], [0, 100]])

print("\n5. the real sill mask, on the resolution it was drawn at (2560x1440)")
m = FrameMask(exclude=SILL, zone_space=SPACE)
bad = [b for b in SILL_BOXES if m.allows(b, (2560, 1440))]
check("every sill box the detector produced is suppressed", not bad, f"leaked: {bad}")
kept = [k for k, pt in GROUND.items() if not m.allows(box_at(pt), (2560, 1440))]
check("no real ground is masked", not kept, f"wrongly masked: {kept}")

print("\n6. the same mask survives the camera being turned down to 1080p")
for w, h in [(1920, 1080), (1280, 720)]:
    sx, sy = w / 2560, h / 1440
    bad = [b for b in SILL_BOXES
           if m.allows([int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)], (w, h))]
    check(f"sill still suppressed at {w}x{h}", not bad, f"leaked: {bad}")
    kept = [k for k, (x, y) in GROUND.items()
            if not m.allows(box_at((int(x * sx), int(y * sy)), int(200 * sx), int(300 * sy)), (w, h))]
    check(f"ground still detected at {w}x{h}", not kept, f"wrongly masked: {kept}")

print("\n7. 6 passes for a weaker reason than it looks — say so out loud")
# The sill triangle has its apex on the origin, and a line through the origin maps to
# itself when both axes are scaled by the same factor. So THIS polygon would have
# survived 2560x1440 → 1920x1080 even with no scaling at all; do not read 6 as evidence
# that raw-pixel polygons are safe. Any polygon not anchored at the origin — 8 — is not.
unscaled = FrameMask(exclude=SILL, zone_space=[1280, 720])
sx, sy = 1280 / 2560, 720 / 1440
leaked = [b for b in SILL_BOXES
          if unscaled.allows([int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)],
                             (1280, 720))]
check("the sill mask happens to be scale-invariant (apex on the origin)", not leaked,
      "so 6 is necessary but not sufficient — 8 is the real test")

print("\n8. a polygon NOT anchored at the origin needs the scaling, or it lands elsewhere")
SPOT = [[1900, 1100], [2100, 1100], [2100, 1300], [1900, 1300]]   # e.g. a bird feeder
right = FrameMask(exclude=SPOT, zone_space=SPACE)
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

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all roi tests passed")
