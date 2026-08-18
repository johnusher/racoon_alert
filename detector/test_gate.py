#!/usr/bin/env python3
"""
Tests for the gate state reader (gate.py).

The measure was chosen by trying three on real pixels from the gate camera, 2026-08-18
(six frames, the aperture against four bar-free patches that stand in for "open"):

    signature            gate (bars)   paved   pavement   foliage   hedge
    vertical anisotropy      0.678      0.469     0.444     0.460    0.426   ← separates
    bars per 100px           7.69       6.44      9.56      8.08    10.67    ← does not
    periodicity              7.44       5.01      5.03      4.55     7.50    ← does not

Counting bars and looking for their spacing both FAIL: a hedge has as many vertical
edges as a gate does, and scores higher periodicity than the gate. What survives is the
plain ratio of vertical to total edge energy, which is also the cheapest of the three and
the only one that normalises away illumination. Reading only the UPPER band of the
aperture scored the same (0.676), and that is what makes the measure survive the case it
exists for: a 2-year-old at the gate occludes the bottom of it, never the top.

The "open" column above was a PROXY, and on 2026-08-18 20:2x the real gate was finally
opened and shut in front of the camera. It held up — see section 10, which is the real
measurement and is now what `closed_above` is set from.

Run:  ../.venv/bin/python detector/test_gate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import GateWatcher, is_daylight

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def bars(w=400, h=300, period=18, tilt=0.0):
    """A picket of vertical bars with top and bottom rails — a closed gate.

    The rails matter: bars alone score a degenerate 1.000 because nothing in the patch
    is horizontal, which is not what a real gate does. With them this scores ~0.8, near
    the 0.678 measured on the real gate.
    """
    im = np.full((h, w, 3), 200, np.uint8)
    for y in range(h):
        off = int(y * tilt)
        for x in range(0, w, period):
            im[y, max(0, x + off):max(0, x + off) + period // 3] = 40
    im[int(h * 0.06):int(h * 0.11), :] = 40                  # top rail
    im[int(h * 0.86):int(h * 0.92), :] = 40                  # bottom rail
    return im


def rubble(w=400, h=300, seed=1):
    """Isotropic texture — what the street through an open gate looks like."""
    rng = np.random.default_rng(seed)
    return np.repeat(rng.integers(60, 210, (h, w, 1), dtype=np.uint8), 3, axis=2)


APERTURE = [[0, 0], [400, 0], [400, 300], [0, 300]]
SPACE = [400, 300]


def watcher(**kw):
    kw.setdefault("min_frames", 2)
    return GateWatcher(APERTURE, SPACE, **kw)


print("\n1. the measure separates bars from not-bars")
g = watcher()
sb, sr = g.score(bars()), g.score(rubble())
check("a picket of bars scores high", sb > 0.60, f"{sb:.3f}")
# 0.5 is the floor for anything isotropic; the real bar-free patches measured
# 0.426-0.469, i.e. slightly HORIZONTAL-dominant. What matters is the gap, not the level.
check("isotropic rubble scores about half", 0.45 < sr < 0.55, f"{sr:.3f}")
check("and they are far apart", sb - sr > 0.15, f"gap {sb - sr:.3f}")
check("a tilted gate still reads as bars", g.score(bars(tilt=0.25)) > 0.55,
      f"{g.score(bars(tilt=0.25)):.3f}")

print("\n2. the first reading is adopted silently — a restart must not alarm")
g = watcher()
check("state starts unknown", g.state is None)
for _ in range(4):
    ev = g.update(bars(), now=0.0)
check("settles on closed", g.state == "closed")
check("without emitting a transition", ev is None, "startup is not an event")

print("\n3. closed -> open is reported once, when it is confirmed")
g = watcher(min_frames=3, min_secs=0.0)
for i in range(4):
    g.update(bars(), now=i)
seen = [g.update(rubble(), now=10 + i) for i in range(5)]
check("the transition fires", "opened" in seen, str(seen))
check("exactly once", seen.count("opened") == 1, str(seen))
check("and the state sticks", g.state == "open")

print("\n4. it takes min_frames of agreement — one odd frame is not an opening")
g = watcher(min_frames=4, min_secs=0.0)
for i in range(5):
    g.update(bars(), now=i)
one = g.update(rubble(), now=10)
check("a single contrary frame changes nothing", one is None and g.state == "closed")
for i in range(3):
    g.update(bars(), now=11 + i)
check("and the streak resets when it agrees again", g.state == "closed")

print("\n5. the deadband holds state rather than guessing")
# A deadband straddling the actual score: neither side is ever confident enough.
sb = watcher().score(bars())
g = watcher(min_frames=2, closed_above=sb, deadband=0.10)
for i in range(4):
    g.update(bars(), now=i)
check("a score inside the deadband settles on nothing", g.state is None, str(g.state))
g2 = watcher(min_frames=2, closed_above=sb - 0.15, deadband=0.10)
for i in range(4):
    g2.update(bars(), now=i)
check("the same score outside it does settle", g2.state == "closed", str(g2.state))

print("\n6. only the upper band is read, so a short body cannot hide the gate")
# A textured occluder, because a real body is textured. A FLAT patch would not test
# anything: flat pixels have no edges at all, so they cannot drag the ratio either way.
tall = bars()
tall[200:, :] = rubble()[200:, :]                   # a toddler blocking the bottom third
g = watcher(read_band=(0.0, 0.45))
check("occluding the bottom leaves the reading intact", g.score(tall) > 0.60,
      f"{g.score(tall):.3f}")
g_full = watcher(read_band=(0.0, 1.0))
check("reading the whole aperture would have been dragged down",
      g_full.score(tall) < g.score(tall),
      f"whole {g_full.score(tall):.3f} vs upper band {g.score(tall):.3f}")

print("\n7. the aperture is in zone_space and scales with the frame")
half = bars(w=200, h=150, period=9)
g = watcher()
check("a half-size frame reads the same gate", g.score(half) > 0.60, f"{g.score(half):.3f}")

print("\n8. daylight vs the infra-red night")
colour = np.dstack([np.full((80, 80), 30, np.uint8), np.full((80, 80), 90, np.uint8),
                    np.full((80, 80), 200, np.uint8)])
grey = np.repeat(np.full((80, 80, 1), 120, np.uint8), 3, axis=2)
check("a colour frame is daylight", is_daylight(colour))
check("a monochrome IR frame is not", not is_daylight(grey))

print("\n10. the real gate, opened and shut (2026-08-18 20:2x, 345 live samples)")
# A trace of the score while the gate was opened, held open ~10s and shut again. It gave
# a control nobody thought to ask for: the middle row is the gate STILL SHUT with a person
# standing at it, which is the case that could have faked an opening. It does not — a body
# costs about 0.08 and an open gate costs 0.13, and the two do not meet.
#
#   closed, nobody in shot  0.658-0.667   n=318
#   closed, PERSON at it    0.578-0.663   n=12    <- the control
#   open (person also there) 0.502-0.537  n=15
#
# Timing, from the same trace: `open` was declared 2.8s after the score fell and `closed`
# 0.7s after it recovered, which is min_frames doing its job rather than dithering.
import json as _json, os as _os
_reg = _json.load(open(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                     "web", "cameras.json")))
_g = next(c["detect"]["gate"] for c in _reg["cameras"]
          if c["id"] == "gate" and (c.get("detect") or {}).get("gate"))
_thr, _band = _g["closed_above"], _g["deadband"]
def _verdict(x):
    return "closed" if x >= _thr + _band else ("open" if x <= _thr - _band else "hold")
check("the shipped threshold is a measurement, not a guess", _g.get("calibrated") is True,
      f"closed_above {_thr} ± {_band}")
for lo, hi in [(0.658, 0.667)]:
    check("a shut gate with nobody there reads closed",
          _verdict(lo) == _verdict(hi) == "closed", f"{lo}-{hi}")
for v in (0.502, 0.520, 0.537):
    check(f"an open gate reads open ({v})", _verdict(v) == "open")
# The control: never `open`. Holding is a correct answer — that is what the deadband buys.
for v in (0.578, 0.596, 0.629, 0.663):
    check(f"a person at a SHUT gate never reads open ({v})", _verdict(v) != "open",
          _verdict(v))
check("and the two dips land in the deadband, so the state is held",
      _verdict(0.578) == "hold" and _verdict(0.596) == "hold")

print("\n9. it says so when it cannot see")
g = watcher()
check("an empty frame scores nothing", g.score(np.zeros((0, 0, 3), np.uint8)) is None)
check("and reports unknown", g.describe().startswith("unknown") or "unknown" in g.describe(),
      g.describe())

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all gate tests passed")
