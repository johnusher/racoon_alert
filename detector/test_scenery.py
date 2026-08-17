#!/usr/bin/env python3
"""
Regression test for the scenery/movement filter, driven by REAL recorded sequences.

Every box sequence below was measured by running MegaDetector over clips in
detector/events/ at the detector's own rate (3 fps). The false positives are the
garden bench that produced 56 bogus PERSON events on 2026-08-16, and the rock that
kept firing later the same day; the true positives are the 03:53 raccoon and the
person who walked through at 00:32.

The bench and the rock fail differently, which is the whole point of this file: the
bench's box is pixel-identical frame to frame, the rock's is not.

Run:  ../.venv/bin/python detector/test_scenery.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenery import SceneryFilter, _displacement

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
# FALSE POSITIVE — the stone trough / plant pot / bench cluster on the right of frame,
# 20260817_074519_animal.mp4. The box is frozen (x1 1256-1274, y2 1065-1075 over the
# whole clip) but MegaDetector's CONFIDENCE in it is not: it runs 0.23 to 0.80. That is
# what separates this from the bench and the rock — both of those stayed under 0.45, so
# `conf_certain` never fired on them and they could be learned as scenery. This one sits
# above 0.70 for seconds at a time while never moving a pixel.
TROUGH_0745 = [
    (0.00, "person", 0.69, [1260, 566, 1653, 1073]),
    (0.31, "person", 0.71, [1257, 565, 1653, 1073]),
    (0.62, "person", 0.70, [1257, 565, 1653, 1073]),
    (0.92, "person", 0.70, [1257, 565, 1653, 1073]),
    (1.23, "person", 0.71, [1257, 565, 1653, 1073]),
    (1.54, "person", 0.70, [1257, 565, 1653, 1073]),
    (1.85, "person", 0.71, [1257, 565, 1653, 1073]),
    (6.16, "person", 0.77, [1259, 567, 1654, 1074]),
    (6.47, "person", 0.80, [1256, 566, 1653, 1073]),
    (6.78, "person", 0.80, [1256, 566, 1653, 1073]),
    (7.09, "person", 0.79, [1256, 566, 1653, 1073]),
    (7.40, "person", 0.79, [1256, 566, 1653, 1073]),
    (7.70, "person", 0.79, [1256, 566, 1653, 1073]),
    (8.01, "person", 0.79, [1256, 566, 1653, 1073]),
    (8.32, "person", 0.49, [1267, 572, 1653, 1073]),
    (8.63, "person", 0.53, [1266, 570, 1653, 1074]),
    (8.94, "person", 0.54, [1266, 570, 1653, 1074]),
    (10.48, "person", 0.24, [1261, 567, 1654, 1073]),
    (11.40, "person", 0.33, [1260, 567, 1654, 1074]),
    (12.33, "person", 0.39, [1259, 567, 1654, 1075]),
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

# FALSE POSITIVE — the rock. A big irregular boulder: unlike the bench, MegaDetector
# does not agree with itself about where its edges are, so the box wobbles by up to
# 28px frame to frame. That wobble is what defeated the original movement gate — it is
# LARGER than the raccoon's real movement (0.064 vs 0.062 of the object's own size),
# which is why no single min_move threshold can separate them. These three clips are
# the events the rock fired on 2026-08-16 at 16:14:57, 16:17:52 and 16:18:39.
# FALSE POSITIVE — the rock, 20260816_161457_person.mp4 (30 detections)
ROCK_1614 = [
    (4.27, 'person', 0.4, [188, 517, 531, 775]),
    (4.63, 'person', 0.38, [189, 516, 531, 775]),
    (4.98, 'person', 0.39, [188, 517, 531, 775]),
    (5.34, 'person', 0.38, [189, 516, 531, 775]),
    (5.69, 'person', 0.38, [189, 516, 531, 774]),
    (6.05, 'person', 0.38, [189, 516, 531, 774]),
    (12.46, 'person', 0.42, [183, 496, 531, 775]),
    (12.81, 'person', 0.34, [190, 504, 523, 777]),
    (18.86, 'person', 0.35, [187, 498, 531, 775]),
    (19.22, 'person', 0.36, [187, 498, 532, 776]),
    (19.57, 'person', 0.36, [187, 498, 532, 776]),
    (19.93, 'person', 0.36, [187, 498, 532, 776]),
    (20.28, 'person', 0.36, [187, 498, 532, 776]),
    (21.0, 'person', 0.37, [174, 506, 538, 775]),
    (21.35, 'person', 0.34, [175, 505, 538, 775]),
    (21.71, 'person', 0.36, [175, 504, 539, 775]),
    (22.06, 'person', 0.36, [175, 505, 537, 775]),
    (22.42, 'person', 0.35, [175, 505, 536, 774]),
    (25.62, 'person', 0.31, [160, 491, 532, 776]),
    (25.98, 'person', 0.3, [160, 491, 532, 775]),
    (26.33, 'person', 0.32, [159, 491, 532, 775]),
    (26.69, 'person', 0.39, [185, 514, 528, 774]),
    (27.05, 'person', 0.42, [185, 513, 530, 774]),
    (27.4, 'person', 0.41, [185, 515, 529, 774]),
    (27.76, 'person', 0.41, [185, 520, 528, 774]),
    (28.11, 'person', 0.45, [184, 520, 527, 774]),
    (28.47, 'person', 0.34, [187, 499, 519, 779]),
    (28.83, 'person', 0.3, [184, 496, 540, 777]),
    (29.89, 'person', 0.31, [184, 495, 541, 777]),
    (30.25, 'person', 0.31, [184, 496, 541, 777]),
]
# FALSE POSITIVE — the rock, 20260816_161752_person.mp4 (23 detections)
ROCK_1617 = [
    (6.75, 'person', 0.31, [186, 506, 532, 775]),
    (7.11, 'person', 0.31, [186, 505, 532, 775]),
    (7.46, 'person', 0.31, [185, 504, 532, 775]),
    (14.21, 'person', 0.39, [187, 516, 540, 772]),
    (14.57, 'person', 0.4, [187, 495, 539, 773]),
    (14.92, 'person', 0.39, [187, 494, 539, 772]),
    (15.28, 'person', 0.39, [187, 495, 539, 773]),
    (15.63, 'person', 0.38, [188, 496, 539, 772]),
    (15.99, 'person', 0.36, [187, 495, 540, 772]),
    (19.19, 'person', 0.3, [193, 499, 534, 774]),
    (19.54, 'person', 0.32, [193, 499, 534, 774]),
    (19.9, 'person', 0.34, [193, 499, 533, 773]),
    (22.38, 'person', 0.43, [184, 506, 531, 776]),
    (22.74, 'person', 0.41, [177, 502, 533, 774]),
    (23.09, 'person', 0.41, [179, 502, 533, 774]),
    (23.45, 'person', 0.41, [180, 503, 533, 774]),
    (23.8, 'person', 0.41, [179, 504, 533, 774]),
    (24.16, 'person', 0.42, [182, 504, 532, 774]),
    (28.78, 'person', 0.31, [184, 497, 534, 775]),
    (29.13, 'person', 0.32, [184, 497, 533, 775]),
    (29.49, 'person', 0.32, [184, 497, 533, 775]),
    (29.84, 'person', 0.32, [184, 497, 533, 775]),
    (30.2, 'person', 0.32, [184, 497, 533, 775]),
]
# FALSE POSITIVE — the rock, 20260816_161839_person.mp4 (37 detections)
ROCK_1618 = [
    (4.27, 'person', 0.4, [181, 498, 541, 775]),
    (4.62, 'person', 0.41, [183, 507, 513, 777]),
    (4.98, 'person', 0.41, [183, 511, 512, 777]),
    (5.33, 'person', 0.41, [183, 510, 515, 778]),
    (5.69, 'person', 0.37, [183, 512, 516, 777]),
    (6.04, 'person', 0.37, [183, 513, 516, 777]),
    (18.49, 'person', 0.45, [183, 499, 534, 774]),
    (18.84, 'person', 0.46, [183, 498, 534, 774]),
    (19.2, 'person', 0.47, [183, 497, 534, 774]),
    (19.55, 'person', 0.47, [183, 498, 534, 774]),
    (19.91, 'person', 0.47, [183, 498, 534, 774]),
    (21.33, 'person', 0.33, [186, 511, 535, 771]),
    (21.69, 'person', 0.32, [186, 510, 535, 771]),
    (22.04, 'person', 0.32, [186, 508, 535, 771]),
    (22.4, 'person', 0.5, [178, 507, 534, 775]),
    (22.75, 'person', 0.46, [179, 506, 535, 775]),
    (23.11, 'person', 0.46, [179, 508, 535, 775]),
    (23.46, 'person', 0.47, [178, 508, 534, 775]),
    (23.82, 'person', 0.47, [178, 508, 534, 775]),
    (24.17, 'person', 0.48, [179, 509, 534, 775]),
    (26.66, 'person', 0.41, [181, 518, 516, 778]),
    (27.02, 'person', 0.43, [181, 517, 516, 777]),
    (27.37, 'person', 0.45, [181, 517, 515, 777]),
    (27.73, 'person', 0.41, [181, 517, 515, 777]),
    (28.09, 'person', 0.39, [181, 515, 516, 777]),
    (28.44, 'person', 0.46, [182, 509, 538, 776]),
    (28.8, 'person', 0.42, [183, 510, 538, 776]),
    (29.15, 'person', 0.43, [183, 510, 538, 776]),
    (29.51, 'person', 0.43, [183, 507, 538, 776]),
    (29.86, 'person', 0.43, [183, 507, 538, 776]),
    (30.22, 'person', 0.43, [183, 507, 538, 776]),
    (30.57, 'person', 0.35, [184, 507, 537, 776]),
    (30.93, 'person', 0.47, [183, 500, 539, 776]),
    (31.29, 'person', 0.45, [183, 502, 538, 776]),
    (31.64, 'person', 0.44, [183, 503, 538, 776]),
    (32.0, 'person', 0.45, [183, 503, 538, 776]),
    (32.35, 'person', 0.43, [183, 506, 538, 776]),
]

FAILS = []


# TRUE POSITIVE, MISSED — the friend who walked outside at 22:48 on 2026-08-16 and
# stared straight at the camera, and got no alert. Reconstructed from the anchor the
# detector persisted (scenery.json: cls=person sightings=2 max_conf=0.35 jitter=0.187)
# and its console trace (`[22:48:45] person:0.31   hits=2/5`, `4 tracked, worst wobble
# 0.187`). MegaDetector caught them in only two frames about ten seconds apart — and
# track_gap_secs is 3.0, so the second detection began a BRAND NEW track whose
# first_box is itself. Displacement from first_box is therefore 0.000 no matter how
# far the person actually moved, and 0.31 is far below conf_certain 0.70, so nothing
# can ever confirm them. Their box really had travelled 0.187 of its own size — NINE
# times min_move — and the filter still logged it as wobble.
#
# This is a structural limit of a movement gate, not a threshold that wants nudging:
# an intermittently-detected object can never accumulate displacement. Lowering
# min_move cannot fix it (the displacement measured is exactly 0.000) and raising
# track_gap_secs re-opens the door to the rock, whose gaps are the same size. The fix
# lives in detect.py instead: SpeciesNet identifies the person and promotes them.
FRIEND_2248 = [
    (0.0,  "person", 0.31, [325, 608, 641, 1077]),
    (10.2, "person", 0.35, [397, 612, 713, 1079]),
]


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


def fresh_bench(**kw):
    """A filter that has already learned the bench as scenery (as in section 3/4)."""
    kw.setdefault("static_after_secs", 180)
    kw.setdefault("min_sightings", 5)
    f = fresh(**kw)
    for minute in range(0, 40, 5):
        replay(f, BENCH_0855, t0=minute * 60)
    return f


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

print("\n8. the rock: a static object whose BOX moves even though the object does not")
ROCKS = {"16:14": ROCK_1614, "16:17": ROCK_1617, "16:18": ROCK_1618}

# Why the bench's fix was not enough for the rock, in one line of arithmetic.
rock_wobble = max(max(_displacement(b, seq[0][3]) for _, _, _, b in seq)
                  for seq in ROCKS.values())
raccoon_move = max(_displacement(b, RACCOON[0][3]) for _, _, _, b in RACCOON)
check("the rock's box wobbles further than the raccoon really moves",
      rock_wobble > raccoon_move,
      f"rock {rock_wobble:.3f} > raccoon {raccoon_move:.3f} — so no fixed min_move "
      f"can separate them")

# The real case: a detector that has been watching this garden, which is what it is
# doing every minute of every day. It must not fire at the rock at all.
warm = fresh()
check("a whole afternoon of rock flicker fires nothing", sum(
    events_fired(seq, warm, t0=m * 60)
    for m, seq in zip(range(0, 60, 5), list(ROCKS.values()) * 4)) == 0)
for i, (when, seq) in enumerate(ROCKS.items()):
    n = events_fired(seq, warm, t0=(90 + i * 10) * 60)
    check(f"…and it still fires nothing when the rock comes round again ({when})",
          n == 0, f"{n} event(s)")

# A filter with no memory of a spot cannot know its wobble before it has watched it,
# so it is allowed one event — and must then have learned. (This is the same
# generosity that lets the raccoon through on first sight; see _gate.)
for when, seq in ROCKS.items():
    f = fresh()
    first = events_fired(seq, f)
    again = events_fired(seq, f, t0=5 * 60) + events_fired(seq, f, t0=10 * 60)
    check(f"a cold filter fires at most once at the rock, then learns it ({when})",
          first <= 1 and again == 0, f"{first} then {again}")

# The rock only ever fired because its wobble read as life, which also kept resetting
# the "nothing has moved here" clock — so the spot could never be written off.
f = fresh(static_after_secs=180, min_sightings=5)
for m, seq in zip(range(0, 60, 5), list(ROCKS.values()) * 4):
    replay(f, seq, t0=m * 60)
check("the rock is eventually learned as scenery despite the wobble",
      any(a["static"] for a in f.anchors),
      f"{sum(1 for a in f.anchors if a['static'])} of {len(f.anchors)} anchors static")
check("one boulder does not sprawl into a crowd of anchors", len(f.anchors) <= 6,
      f"{len(f.anchors)} anchors")

# The gate must be adaptive, not just higher: a real visitor at a learned scenery spot
# still has to get through.
f = fresh(static_after_secs=180, min_sightings=5)
for m, seq in zip(range(0, 60, 5), list(ROCKS.values()) * 4):
    replay(f, seq, t0=m * 60)
walk = [(t, c, cf, [b[0] + int(t * 90), b[1], b[2] + int(t * 90), b[3]])
        for t, c, cf, b in ROCK_1618[:8]]          # same boulder box, walking right
check("something that genuinely walks out of the rock still fires",
      events_fired(walk, f, t0=60 * 60) >= 1)
check("the raccoon is unaffected by rock learning",
      events_fired(RACCOON, f, t0=90 * 60) >= 1)

print("\n9. the limit of a movement gate: someone who stands still (2026-08-16 22:48)")
# Documents WHY detect.py needs SpeciesNet's promote path. These assertions describe
# the movement gate's ceiling; if one ever starts failing the gate got better and the
# promote path may be able to relax, so check there before 'fixing' the test.
check("the friend really did move, by 9x min_move",
      round(_displacement(FRIEND_2248[1][3], FRIEND_2248[0][3]), 3) >= 0.18,
      f"displacement={_displacement(FRIEND_2248[1][3], FRIEND_2248[0][3]):.3f} vs min_move=0.02")
check("…but the gate confirms nothing, because each detection restarts the track",
      len(replay(fresh(), FRIEND_2248)) == 0)
check("…so no event fires — exactly the alert that went missing",
      events_fired(FRIEND_2248, fresh()) == 0)
check("they are still SEEN, so the monitor drew the box John saw",
      len(replay_seen(fresh(), FRIEND_2248)) == 2)
check("a lower min_move cannot rescue them (displacement really is 0.000)",
      events_fired(FRIEND_2248, fresh(min_move=0.001)) == 0)
check("only trusting confidence would rescue them — and that lets the bench back in",
      events_fired(FRIEND_2248, fresh(conf_certain=0.30)) >= 1
      and events_fired(BENCH_0926, fresh(conf_certain=0.30)) >= 1)

print("\n10. confidence is not motion: the trough that stayed unlearnable (2026-08-17)")
# The stone trough was detected 2470 times over 100 minutes, never moved a pixel
# (jitter 0.013, gate 0.032) — and was STILL `static: false` in scenery.json, so it
# fired 15 events between 07:37 and 07:47. The reason is that `conf_certain` did two
# jobs: it let a confident detection override a learned anchor (right, and section 4),
# and it also counted as MOVEMENT, which reset moved_at and cleared static (wrong).
# MegaDetector reaches 0.80 on this trough, so the 180s static clock was restarted every
# few seconds and the spot could never be written off. Confidence is evidence that
# something is THERE, never that it moved.
f = fresh(static_after_secs=180, min_sightings=5)
for minute in range(0, 12):
    replay(f, TROUGH_0745, t0=minute * 60)
trough = [a for a in f.anchors if a["max_conf"] >= 0.70]
check("the trough really is detected at high confidence", len(trough) == 1,
      f"{len(trough)} anchors over conf_certain")
check("…and really never moves", all(a["jitter"] < 0.032 for a in trough),
      f"jitter={trough[0]['jitter']:.4f}" if trough else "-")
check("the trough anchor is marked static", any(a["static"] for a in trough),
      f"static={[a['static'] for a in trough]}")
check("…so the low-confidence detections stop firing",
      len(replay(f, [d for d in TROUGH_0745 if d[2] < 0.70], t0=13 * 60)) == 0,
      f"{len(replay(f, [d for d in TROUGH_0745 if d[2] < 0.70], t0=14 * 60))} still confirmed")
# The other half of conf_certain — overriding a learned anchor — must be untouched.
check("a confident detection still overrides a QUIET anchor (section 4 unchanged)",
      len(replay(fresh_bench(), [(0.0, "person", 0.84, [1215, 860, 1590, 1077])],
                 t0=45 * 60)) == 1)
check("a real person walking through is still confirmed", len(replay(fresh(), PERSON)) >= 20)
check("the raccoon is still confirmed", len(replay(fresh(), RACCOON)) >= 1)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all scenery-filter checks passed")
