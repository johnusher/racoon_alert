#!/usr/bin/env python3
"""
Tests for the gate calibration (calibrate_gate.py).

The two things worth testing are the ones that decide whether the number it writes is
trustworthy, and whether writing it can wreck the registry:

  • the threshold must be pitched between the two clusters' NEAREST EDGES, not their
    means — the worst closed reading and the best open one are the boundary that has to
    hold, and means hide a class that is far noisier than the other;
  • overlapping clusters must be refused outright rather than split down the middle,
    because a threshold inside the overlap is one that fires on a shut gate.

Writing is a surgical text edit rather than a JSON round-trip, so that the registry's
comments and hand-laid formatting survive; that it stays parseable is asserted here.

Run:  ../.venv/bin/python detector/test_calibrate_gate.py
"""
import json
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
from calibrate_gate import recommend, apply_to_registry

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


print("\n1. the threshold sits between the nearest edges, not the means")
closed = [0.70, 0.68, 0.72, 0.67]          # worst closed reading: 0.67
opened = [0.45, 0.44, 0.47, 0.43]          # best open reading:    0.47
thr, band, gap = recommend(closed, opened)
check("it lands inside the gap", 0.47 < thr < 0.67, f"{thr:.3f}")
check("halfway between the edges", abs(thr - 0.57) < 1e-6, f"{thr:.3f}")
check("the gap is edge-to-edge", abs(gap - 0.20) < 1e-6, f"{gap:.3f}")
check("the deadband is a quarter of the gap", abs(band - 0.05) < 1e-6, f"{band:.3f}")

print("\n2. one noisy class does not drag the threshold onto the other")
# The closed readings are tight; the open ones have a long tail upward. Using the means
# would put the boundary at 0.575 — ABOVE that tail's 0.60, i.e. inside the open class.
noisy_open = [0.40, 0.42, 0.45, 0.60]
thr2, _, _ = recommend([0.70, 0.71, 0.69], noisy_open)
check("the threshold clears the worst open reading", thr2 > max(noisy_open), f"{thr2:.3f}")
check("and stays under the worst closed one", thr2 < min([0.70, 0.71, 0.69]), f"{thr2:.3f}")

print("\n3. overlapping classes are refused, not averaged")
_, _, gap3 = recommend([0.50, 0.55], [0.52, 0.58])
check("the gap comes out non-positive so the caller can refuse", gap3 <= 0, f"{gap3:.3f}")

print("\n4. writing the number keeps the registry a registry")
REG = json.dumps({"cameras": [
    {"id": "west", "detect": {}},
    {"id": "gate", "detect": {"gate": {"aperture": [[0, 0]], "closed_above": 0.57,
                                       "deadband": 0.04, "calibrated": False}}},
], "_comment": "hand-written prose that must survive"}, indent=2)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "cameras.json")
    open(p, "w").write(REG)
    ok, err = apply_to_registry("gate", 0.612, 0.037, path=p)
    check("it reports success", ok, str(err))
    out = open(p).read()
    got = json.loads(out)
    g = got["cameras"][1]["detect"]["gate"]
    check("closed_above is the new number", abs(g["closed_above"] - 0.612) < 1e-9, str(g))
    check("deadband is the new number", abs(g["deadband"] - 0.037) < 1e-9, str(g))
    check("it is now marked calibrated", g["calibrated"] is True, str(g))
    check("the prose survived", "must survive" in out)
    check("the other camera is untouched", got["cameras"][0] == {"id": "west", "detect": {}})

print("\n5. it refuses a camera it cannot find rather than writing somewhere else")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "cameras.json")
    open(p, "w").write(REG)
    ok, err = apply_to_registry("south", 0.6, 0.03, path=p)
    check("unknown camera is refused", not ok, str(err))
    check("and the file is unchanged", open(p).read() == REG)

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all gate-calibration tests passed")
