#!/usr/bin/env python3
"""
Tests for named regions (zones.py).

Same foot-point convention as roi.py, and for the same reason: somebody leaning over the
gate has a box that sprawls across the picture while their feet stay where they are. The
zone that matters here is the ground in front of the gate, so that "a small person at the
gate" cannot be satisfied by a small person twenty metres up the road.

Run:  ../.venv/bin/python detector/test_zones.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
import h32env
from zones import Zones

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


SQ = [{"id": "gate", "name": "at the gate", "poly": [[100, 100], [300, 100], [300, 300], [100, 300]]}]
SPACE = [400, 400]


def box_at(foot, w=60, h=200):
    x, y = foot
    return [x - w // 2, y - h, x + w // 2, y]


print("\n1. it is the feet that decide")
z = Zones(SQ, SPACE)
check("feet inside", z.at(box_at((200, 250)), (400, 400)) == ["gate"])
check("feet outside", z.at(box_at((350, 250)), (400, 400)) == [])
check("a tall body whose head is far above the zone still counts",
      z.at([170, 20, 230, 250], (400, 400)) == ["gate"], "head at row 20, feet at 250")
check("a body overlapping the zone but standing outside it does not",
      z.at([120, 120, 380, 380], (400, 400)) == [], "feet at (250,380), below the zone")

print("\n2. zones scale with the frame, like every other polygon here")
check("the same spot at half size", z.at(box_at((100, 125), 30, 100), (200, 200)) == ["gate"])
check("and the spot beyond it", z.at(box_at((175, 125), 30, 100), (200, 200)) == [])

print("\n3. empty is a real answer, not a crash")
e = Zones()
check("no zones configured is falsy", not e)
check("and matches nothing", e.at(box_at((200, 250)), (400, 400)) == [])
check("contains() is safe too", not e.contains("gate", box_at((200, 250)), (400, 400)))

print("\n4. several zones, and overlap is allowed")
two = Zones(SQ + [{"id": "path", "poly": [[150, 200], [400, 200], [400, 400], [150, 400]]}], SPACE)
check("a spot in both is reported in both",
      sorted(two.at(box_at((200, 250)), (400, 400))) == ["gate", "path"])
check("a spot in one is reported once", two.at(box_at((350, 350)), (400, 400)) == ["path"])
check("contains picks one out", two.contains("path", box_at((350, 350)), (400, 400)))

print("\n5. the gate camera's shipped zone")
cfg = h32env.detector_config(camera="gate")
gz = Zones(cfg.get("zones"), cfg.get("zone_space"))
check("the gate camera has a zone called 'gate'", "gate" in [z.get("id") for z in gz.zones],
      gz.describe())
# The zone must contain the ground at the gate and not the whole frame, or "at the gate"
# means nothing. These two points are read off the 17:45 frame.
at_gate = box_at((1050, 880), 120, 300)
far_off = box_at((300, 1300), 120, 300)
check("someone standing at the gate is in it", gz.contains("gate", at_gate, (2560, 1440)))
check("someone down by the house is not", not gz.contains("gate", far_off, (2560, 1440)))

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all zone tests passed")
