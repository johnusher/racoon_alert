#!/usr/bin/env python3
"""
Tests for child-vs-adult by size (stature.py).

The gate is the ruler. Everything else about telling a 2-year-old from an adult needs
camera calibration — the literature does it with a ground-plane homography, or with the
head-to-body ratio from a pose model — but a person standing AT the gate is at the same
depth as the gate, so perspective cancels and no calibration survives being needed. Head
above the gate's top rail: adult. Head below it: child.

The numbers this has to separate are unusually kind. A garden gate is about 1.1 m; a
2-year-old is about 87 cm (WHO, 3rd-97th percentile 82-93 cm); an adult about 170 cm. So
the child is comfortably under the rail and the adult is comfortably over it, and the
threshold sits in a gap nothing has to be clever to find.

What lands in "unsure" is roughly a 0.9-1.3 m person — a 3-to-7-year-old. That is the
correct answer for this system: it is built to spot ONE toddler, and an older child at
the gate is a case it should decline rather than guess at.

⚠️ The known weakness is a crouching or bending adult, who is genuinely child-shaped to
any size-only measure. The fix if it ever fires is a pose model (head-to-body ratio),
which is scale-invariant and sees the crouch — deliberately not built until the camera
is mounted and we know whether it actually happens.

Run:  ../.venv/bin/python detector/test_stature.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stature import classify

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


# The gate as measured in the frame: top rail at row 672, its foot on the ground at 872.
TOP, GATE_H, GROUND = 672, 200, 872
PX_PER_M = GATE_H / 1.10                                  # a ~1.10 m garden gate


def person(height_m, ground=GROUND):
    """A person of this real height standing at the gate."""
    top = int(round(ground - height_m * PX_PER_M))
    return [900, top, 1000, ground]


def verdict(box, **kw):
    return classify(box, TOP, GATE_H, **kw)[0]


print("\n1. the household, at the gate")
for who, m, want in [("the 2-year-old (87cm)", 0.87, "child"),
                     ("a small 2-year-old (82cm)", 0.82, "child"),
                     ("a large 2-year-old (93cm)", 0.93, "child"),
                     ("an adult (170cm)", 1.70, "adult"),
                     ("a short adult (155cm)", 1.55, "adult"),
                     ("a tall adult (190cm)", 1.90, "adult")]:
    got = verdict(person(m))
    check(f"{who} reads {want}", got == want, f"got {got}, head row {person(m)[1]}")

print("\n2. it declines rather than guesses in the middle")
for who, m in [("a 4-year-old (~100cm)", 1.00), ("a 6-year-old (~115cm)", 1.15)]:
    got = verdict(person(m))
    check(f"{who} is unsure", got == "unsure", f"got {got}")

print("\n3. the boundary is the rail, and the margin brackets it")
check("head exactly on the rail is unsure", verdict([900, TOP, 1000, GROUND]) == "unsure")
check("just under the rail is still unsure",
      verdict([900, TOP + 10, 1000, GROUND]) == "unsure")
check("well under the rail is a child",
      verdict([900, TOP + 40, 1000, GROUND]) == "child")
check("well over the rail is an adult",
      verdict([900, TOP - 40, 1000, GROUND]) == "adult")

print("\n4. it refuses when it cannot see the head")
check("a box cut off by the top of the frame is unsure",
      classify([900, 0, 1000, GROUND], TOP, GATE_H, frame_h=1440)[0] == "unsure",
      "head is out of shot, so the height is a lower bound only")
check("a box starting 1px in is still cut off",
      classify([900, 1, 1000, GROUND], TOP, GATE_H, frame_h=1440)[0] == "unsure")
check("a box clear of the edge is judged normally",
      classify(person(0.87), TOP, GATE_H, frame_h=1440)[0] == "child")

print("\n5. it refuses when the gate has not been measured")
check("no gate height means no verdict", classify(person(0.87), TOP, 0)[0] == "unsure")
check("no top row means no verdict", classify(person(0.87), None, GATE_H)[0] == "unsure")

print("\n6. the margin has a ceiling, and it is the child's clearance")
# The two sides are NOT symmetric. Against a 1.10 m gate the toddler's head sits 0.23 m
# below the rail (42px here) while an adult's is 0.60 m above it (109px). So the adult
# side is generous and the CHILD side is what limits the margin: at 0.15 the toddler
# clears by 12px, and at 0.30 the margin (0.33 m) is wider than his clearance and he
# goes unsure. Anything above ~0.20 turns this feature off without saying so.
check("at the shipped margin the toddler clears", verdict(person(0.87), margin=0.15) == "child")
check("a doubled margin swallows him — this is the ceiling",
      verdict(person(0.87), margin=0.30) == "unsure",
      "0.30*1.10m = 0.33m > his 0.23m clearance")
check("the adult side has room to spare", verdict(person(1.70), margin=0.30) == "adult")

print("\n6b. how tall a gate can get before it lies about adults")
# Solve it rather than guess: an adult of height A against a gate of height G reads as a
# child when (G - A) > margin*G, i.e. G > A/(1 - margin) = 1.70/0.85 = 2.0 m. Below that
# the margin — which scales WITH the gate — absorbs the error into "unsure" first. So a
# normal garden or even a head-height gate is safe, and only a 2 m+ gate inverts.
for gate_m, want in [(1.10, "adult"), (1.70, "unsure"), (1.90, "unsure"), (2.50, "child")]:
    gh = int(gate_m * PX_PER_M)
    got = classify(person(1.70), GROUND - gh, gh)[0]
    check(f"a {gate_m:.2f}m gate calls an adult {want}", got == want, f"got {got}")
check("the shipped 1.10m gate is well inside the safe range", True,
      "inversion needs a gate over 2.0m — re-check if the gate is ever replaced")

print("\n7. the detail says what it saw, for the log")
v, detail = classify(person(0.87), TOP, GATE_H)
check("detail mentions the rail", "rail" in detail or "gate" in detail, detail)

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all stature tests passed")
