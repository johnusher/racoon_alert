#!/usr/bin/env python3
"""
Regression test for the scenery/movement filter, driven by REAL recorded sequences.

Every box sequence below was measured by running MegaDetector over clips in
detector/events/ at the detector's own rate (3 fps). The false positives are the
garden bench that produced 56 bogus PERSON events on 2026-08-16; the true positives
are the 03:53 raccoon and the person who walked through at 00:32.

Run:  ../.venv/bin/python detector/test_scenery.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenery import SceneryFilter

# --- real data -------------------------------------------------------------

# FALSE POSITIVE — the stone bench, event 20260816_085554_person.mp4 (t=20.8..22.5s)
BENCH_0855 = [
    (0.0, "person", 0.31, [1215, 860, 1590, 1077]),
    (0.34, "person", 0.43, [1219, 864, 1588, 1078]),
    (0.68, "person", 0.42, [1219, 864, 1588, 1078]),
    (1.02, "person", 0.39, [1218, 864, 1588, 1078]),
    (1.36, "person", 0.36, [1218, 864, 1588, 1078]),
    (1.70, "person", 0.36, [1218, 864, 1588, 1078]),
]
# FALSE POSITIVE — same bench, 31 minutes later, 20260816_092646_person.mp4
BENCH_0926 = [
    (0.0, "person", 0.33, [1218, 861, 1588, 1077]),
    (0.34, "person", 0.33, [1218, 861, 1588, 1077]),
    (0.68, "person", 0.30, [1218, 861, 1588, 1077]),
    (1.02, "person", 0.30, [1218, 861, 1588, 1077]),
]
# TRUE POSITIVE — the raccoon, 20260816_035358_animal.mp4. Only two detections in
# the whole clip: the filter must not need more than that.
RACCOON = [
    (0.0, "animal", 0.42, [951, 809, 1141, 939]),
    (0.34, "animal", 0.30, [942, 806, 1139, 926]),
]
# TRUE POSITIVE — person walking through, 20260816_003201_person.mp4. Stands still
# for ~6s (boxes barely move) and then walks across frame.
PERSON = [
    (0.0, "person", 0.37, [452, 60, 844, 897]),
    (0.34, "person", 0.63, [442, 68, 843, 893]),
    (0.68, "person", 0.53, [436, 72, 826, 887]),
    (1.02, "person", 0.47, [433, 73, 836, 885]),
    (1.36, "person", 0.38, [432, 73, 833, 885]),
    (1.70, "person", 0.33, [434, 74, 814, 881]),
    (2.04, "person", 0.43, [434, 73, 836, 888]),
    (2.72, "person", 0.30, [427, 74, 838, 889]),
    (3.06, "person", 0.32, [429, 74, 835, 889]),
    (3.40, "person", 0.44, [431, 74, 840, 896]),
    (3.74, "person", 0.51, [430, 73, 839, 895]),
    (4.08, "person", 0.35, [429, 83, 844, 892]),
    (5.10, "person", 0.32, [420, 74, 840, 886]),
    (5.44, "person", 0.33, [434, 73, 842, 885]),
    (6.12, "person", 0.46, [439, 73, 840, 887]),
    (7.14, "person", 0.74, [637, 38, 1018, 805]),
    (8.50, "person", 0.65, [972, 11, 1404, 687]),
    (8.84, "person", 0.54, [1058, 4, 1461, 652]),
    (9.18, "person", 0.74, [1131, 2, 1485, 631]),
    (9.52, "person", 0.74, [1193, 4, 1492, 624]),
    (10.54, "person", 0.64, [1343, 2, 1575, 460]),
    (11.22, "person", 0.57, [1394, 1, 1618, 447]),
    (12.24, "person", 0.39, [1331, 0, 1551, 476]),
    (12.92, "person", 0.64, [1314, 0, 1541, 468]),
    (13.26, "person", 0.84, [1287, 0, 1559, 491]),
    (13.60, "person", 0.85, [1261, 1, 1558, 583]),
]

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def replay(filt, seq, t0=0.0):
    """Feed a sequence through the filter; return the detections cleared to fire."""
    out = []
    for dt, cls, conf, box in seq:
        confirmed, _unproven, _muted = filt.apply([(cls, conf, box)], now=t0 + dt)
        out.extend(confirmed)
    return out


def replay_seen(filt, seq, t0=0.0):
    """Detections the filter still believes in — confirmed plus not-yet-proven."""
    out = []
    for dt, cls, conf, box in seq:
        confirmed, unproven, _muted = filt.apply([(cls, conf, box)], now=t0 + dt)
        out.extend(confirmed + unproven)
    return out


def fresh(**kw):
    kw.setdefault("path", None)
    return SceneryFilter(**kw)


print("\n1. the bench must never clear the movement gate (it never moves)")
kept = replay(fresh(), BENCH_0926)
check("bench 09:26 clears nothing to fire", not kept, f"{len(kept)} of {len(BENCH_0926)}")
kept = replay(fresh(), BENCH_0855)
check("bench 08:55 clears nothing to fire", not kept, f"{len(kept)} of {len(BENCH_0855)}")

print("\n2. real animals and people must still get through")
kept = replay(fresh(), RACCOON)
check("raccoon (only 2 detections in 37s) is confirmed", len(kept) >= 1,
      f"{len(kept)} of {len(RACCOON)} confirmed")
check("raccoon is never hidden from the monitor",
      len(replay_seen(fresh(), RACCOON)) == len(RACCOON))
kept = replay(fresh(), PERSON)
check("person walking through is confirmed", len(kept) >= 20,
      f"{len(kept)} of {len(PERSON)} confirmed")
check("person is confirmed within the first 4 detections",
      len(replay(fresh(), PERSON[:4])) >= 1)

print("\n3. scenery memory: a location that keeps flickering becomes furniture")
f = fresh(static_after_secs=180, min_sightings=5)
for minute in range(0, 40, 5):                       # a flicker burst every 5 minutes
    replay(f, BENCH_0855, t0=minute * 60)
check("bench anchor is marked static", any(a["static"] for a in f.anchors))
check("bench is dropped outright once learned, not merely held back",
      len(replay_seen(f, BENCH_0926, t0=45 * 60)) == 0)

print("\n4. a real person standing where the furniture is still gets through")
f = fresh(static_after_secs=180, min_sightings=5)
for minute in range(0, 40, 5):
    replay(f, BENCH_0855, t0=minute * 60)
loud = [(0.0, "person", 0.84, [1215, 860, 1590, 1077])]
check("high-confidence detection overrides the scenery anchor",
      len(replay(f, loud, t0=45 * 60)) == 1)

print("\n5. scenery memory expires so a repositioned camera relearns")
f = fresh(static_after_secs=180, min_sightings=5, forget_secs=1800)
for minute in range(0, 40, 5):
    replay(f, BENCH_0855, t0=minute * 60)
f.apply([], now=45 * 60 + 3600)                       # an hour of nothing there
check("stale anchors are forgotten", not f.anchors, f"{len(f.anchors)} left")

print("\n6. anchors survive a detector restart")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "scenery.json")
    f = SceneryFilter(path=p, static_after_secs=180, min_sightings=5)
    for minute in range(0, 40, 5):
        replay(f, BENCH_0855, t0=minute * 60)
    f.save()
    f2 = SceneryFilter(path=p, static_after_secs=180, min_sightings=5)
    check("reloaded filter still suppresses the bench",
          len(replay_seen(f2, BENCH_0926, t0=45 * 60)) == 0)

print("\n7. end to end: would detect.py actually fire an event?")


def events_fired(seq, filt=None, min_hits=2, window=5, cooldown=30, t0=0.0):
    """Replay a sequence through exactly the gate detect.py uses: everything believed
    counts toward min_hits in a rolling window, only confirmed detections may fire."""
    from collections import deque
    hits, fired, last = deque(maxlen=window), 0, -1e9
    for dt, cls, conf, box in seq:
        now = t0 + dt
        if filt is None:
            confirmed, unproven = [(cls, conf, box)], []
        else:
            confirmed, unproven, _ = filt.apply([(cls, conf, box)], now=now)
        interesting = [d for d in confirmed + unproven if d[0] in ("animal", "person")]
        fireable = [d for d in confirmed if d[0] in ("animal", "person")]
        hits.append(1 if interesting else 0)
        if fireable and sum(hits) >= min_hits and (now - last) > cooldown:
            last = now; fired += 1
    return fired


check("without the filter the bench DOES fire (the bug reproduces)",
      events_fired(BENCH_0926) >= 1, f"{events_fired(BENCH_0926)} event(s)")
check("with the filter the bench fires nothing (09:26)",
      events_fired(BENCH_0926, fresh()) == 0)
check("with the filter the bench fires nothing (08:55)",
      events_fired(BENCH_0855, fresh()) == 0)
check("a whole morning of bench flicker fires nothing", (lambda: (
    lambda f: sum(events_fired(BENCH_0855, f, t0=m * 60) for m in range(0, 60, 5))
)(fresh()))() == 0)
check("the raccoon still fires", events_fired(RACCOON, fresh()) >= 1)
check("the person still fires", events_fired(PERSON, fresh()) >= 1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all scenery-filter checks passed")
