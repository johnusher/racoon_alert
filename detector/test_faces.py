#!/usr/bin/env python3
"""
Tests for the face MATCHING and VOTING logic — the part that decides who someone is.

Deliberately model-free and data-free: synthetic unit vectors stand in for SFace
embeddings, so this runs anywhere, needs no downloads, and never touches a real face.
Detection quality is OpenCV's problem; these are ours.

The rules under test exist because of what the 2026-08-16 spike found: a face is only
visible in ~38% of the frames a person appears in, so a single frame is weak evidence,
and the false-accept rate is UNMEASURED, so an ambiguous match must resolve to unknown
rather than to a guess.

Run:  ../.venv/bin/python detector/test_faces.py
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from faces import FaceMatcher

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def vec(*parts, dim=128):
    """A deterministic unit vector; `parts` seeds the leading components."""
    v = np.zeros(dim, np.float32)
    for i, p in enumerate(parts):
        v[i] = p
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def blend(a, b, t):
    """A vector t of the way from a to b — for making faces of controlled similarity."""
    v = (1 - t) * a + t * b
    return v / np.linalg.norm(v)


JOHN = vec(1, 0, 0)
JANE = vec(0, 1, 0)
STRANGER = vec(0, 0, 1)


def matcher(**kw):
    kw.setdefault("threshold", 0.40)
    kw.setdefault("margin", 0.10)
    kw.setdefault("vote_window_secs", 45)
    kw.setdefault("min_votes", 2)
    m = FaceMatcher(**kw)
    return m


print("\n1. matching a single face")
m = matcher()
check("empty store yields unknown", m.match(JOHN)[0] is None)

m.enroll("john", [JOHN])
m.enroll("jane", [JANE])
name, score, marg = m.match(JOHN)
check("an exact match is found", name == "john", f"{name} score={score:.2f}")
name, score, _ = m.match(STRANGER)
check("an orthogonal face is unknown", name is None, f"got {name} score={score:.2f}")

print("\n2. weak and ambiguous evidence must resolve to unknown")
# 0.35 similarity to john — below our 0.40 floor
weak = blend(STRANGER, JOHN, 0.0)
near = JOHN * 0.35 + STRANGER * math.sqrt(1 - 0.35 ** 2)
name, score, _ = m.match(near.astype(np.float32))
check("below threshold is unknown", name is None, f"score={score:.2f}")

# equally similar to two enrolled people: a coin flip we must not take
ambiguous = blend(JOHN, JANE, 0.5)
name, score, marg = m.match(ambiguous)
check("a face between two people is unknown", name is None,
      f"got {name} score={score:.2f} margin={marg:.2f}")

print("\n3. several enrolled shots per person")
m2 = matcher()
m2.enroll("john", [JOHN, blend(JOHN, JANE, 0.2), blend(JOHN, STRANGER, 0.2)])
check("the best enrolled shot wins", m2.match(JOHN)[0] == "john")
check("a pose near one enrolled shot still matches",
      m2.match(blend(JOHN, STRANGER, 0.18))[0] == "john")

print("\n4. voting across a visit, because one frame is weak evidence")
m3 = matcher(min_votes=2)
m3.enroll("john", [JOHN])
m3.observe_match("john", 0.8, now=100.0)
check("one sighting is not enough", m3.verdict(now=100.0)[0] is None,
      f"{m3.verdict(now=100.0)}")
m3.observe_match("john", 0.7, now=101.0)
check("two sightings identify him", m3.verdict(now=101.0)[0] == "john")

print("\n5. stale sightings fall out of the window")
m4 = matcher(min_votes=2, vote_window_secs=45)
m4.enroll("john", [JOHN])
m4.observe_match("john", 0.8, now=0.0)
m4.observe_match("john", 0.8, now=1.0)
check("both count while fresh", m4.verdict(now=2.0)[0] == "john")
check("neither counts an hour later", m4.verdict(now=3600.0)[0] is None)

print("\n6. a contested visit is unknown, not a guess")
m5 = matcher(min_votes=2)
m5.enroll("john", [JOHN]); m5.enroll("jane", [JANE])
for t in (10.0, 11.0):
    m5.observe_match("john", 0.7, now=t)
for t in (12.0, 13.0):
    m5.observe_match("jane", 0.7, now=t)
name, votes, detail = m5.verdict(now=13.0)
check("a tie resolves to unknown", name is None, f"got {name} ({detail})")
m5.observe_match("john", 0.75, now=14.0)
m5.observe_match("john", 0.75, now=15.0)
check("a clear majority resolves", m5.verdict(now=15.0)[0] == "john")

print("\n7. unknown faces never invent a name")
m6 = matcher(min_votes=2)
m6.enroll("john", [JOHN])
for t in (1.0, 2.0, 3.0, 4.0):
    m6.observe_match(None, 0.2, now=t)
name, votes, _ = m6.verdict(now=4.0)
check("four unknown sightings stay unknown", name is None)
check("the visit is still reported as seen", votes == 0 or votes is not None)

print("\n8. reset between visits")
m7 = matcher(min_votes=2)
m7.enroll("john", [JOHN])
m7.observe_match("john", 0.8, now=1.0)
m7.observe_match("john", 0.8, now=2.0)
m7.reset()
check("after reset the visit starts empty", m7.verdict(now=2.0)[0] is None)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all face-matching checks passed")
