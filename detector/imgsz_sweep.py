#!/usr/bin/env python3
"""Replay the whole events archive through the detector at several imgsz values.

Written to answer one buying question — "can `imgsz` drop from 1280 so cheaper hardware
will do?" — and kept because the same replay is what you want after ANY change to the
detector that could move box coordinates or confidences: a different MegaDetector variant,
or the int8 quantisation that compiling for a Hailo NPU implies. Point it at the new model
and compare the tables against the ones in TODO.md.

    ./imgsz_sweep.py run                 # the expensive part: infer, cache one JSON per clip
    ./imgsz_sweep.py report              # the cheap part: read the cache, print the tables
    ./imgsz_sweep.py run --sizes 1280,640

`run` is resumable — a clip that already has a cache file is skipped, so it can be
interrupted. Detections are cached down to conf 0.10 so `report` can re-examine thresholds
without paying for inference again.

Everything matches detect.py: the same CLAHE enhance(), the same class filter, the same
frame rate, and the real SceneryFilter driven exactly as test_scenery.py drives it.
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import cv2

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from scenery import SceneryFilter, _displacement, _iou, _scale   # noqa: E402

CACHE = os.path.join(BASE, "sweep_cache")
EVENTS = os.path.join(BASE, "events")
FLOOR = 0.10                       # cache everything above this; filter in report
cfg = json.load(open(os.path.join(BASE, "config.json")))
CONF = cfg["conf"]
SC = cfg["scenery"]
IOU_MATCH = SC["iou_match"]

# The cases this system was built around — see README. Named so a regression in any of
# them is visible by name rather than buried in an average.
HARD = {
    "20260816_035307_animal.mp4": "raccoon 03:53:07 (species 0.978)",
    "20260816_035358_animal.mp4": "raccoon 03:53:58 (2 frames in 37s)",
    "20260816_003201_person.mp4": "person 00:32 (scored 0.39, below the bench)",
    "20260816_211807_person.mp4": "the 21:18 cat called person 0.74",
    "20260816_225946_person.mp4": "the friend who stood still and stared",
    "20260816_065109_animal.mp4": "cat 06:51 (species 0.91)",
    "20260817_065352_cat.mp4": "cat 06:53",
}


def clips():
    return sorted(f for f in os.listdir(EVENTS) if f.endswith(".mp4"))


def stem_secs(clip):
    """Seconds-of-archive from the filename, so clips can be replayed in real order."""
    d, t = clip[:8], clip[9:15]
    return (int(d[6:8]) * 86400 + int(t[0:2]) * 3600
            + int(t[2:4]) * 60 + int(t[4:6]))


def qualifying(dets):
    """The detections detect.py would have kept, at production thresholds."""
    return [d for d in dets if d["conf"] >= CONF.get(d["cls"], 1.0)]


# ---------------------------------------------------------------- run (inference)
def run(sizes):
    from ultralytics import YOLO
    import torch

    os.makedirs(CACHE, exist_ok=True)
    device = cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu"
    model = YOLO(os.path.join(BASE, cfg["model"]))
    names = model.names
    ignore = set(cfg.get("ignore_classes", []))
    det_fps = cfg.get("fps", 3)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))     # detect.py:48

    def enhance(img):                                               # detect.py:194
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def frames_of(path):
        cap = cv2.VideoCapture(path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        step = max(1, int(round(src_fps / det_fps)))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i / src_fps, frame
            i += 1
        cap.release()

    todo = [c for c in clips() if not os.path.exists(os.path.join(CACHE, c + ".json"))]
    print(f"device={device}  clips={len(clips())}  todo={len(todo)}  sizes={sizes}",
          flush=True)
    started = time.time()
    for n, c in enumerate(todo, 1):
        t0 = time.time()
        frames = list(frames_of(os.path.join(EVENTS, c)))
        out = {"clip": c, "n_frames": len(frames), "by_size": {}}
        for sz in sizes:
            per_frame = []
            for t, frame in frames:
                r = model(enhance(frame), imgsz=sz, conf=FLOOR,
                          device=device, verbose=False)[0]
                per_frame.append({"t": round(t, 3), "dets": [
                    {"cls": names[int(b.cls)], "conf": round(float(b.conf), 4),
                     "box": [int(x) for x in b.xyxy[0]]}
                    for b in r.boxes if names[int(b.cls)] not in ignore]})
            out["by_size"][str(sz)] = per_frame
        json.dump(out, open(os.path.join(CACHE, c + ".json"), "w"))
        el = time.time() - started
        print(f"[{n}/{len(todo)}] {c}  {len(frames)}f  {time.time()-t0:.1f}s"
              f"  (elapsed {el/60:.1f}m, eta {el/n*(len(todo)-n)/60:.1f}m)", flush=True)
    print("done", flush=True)


# ------------------------------------------------------------------------- report
def build_spots(frames):
    """Link qualifying detections across frames into spots — one place in the garden."""
    spots = []
    for fr in frames:
        for d in qualifying(fr["dets"]):
            best, best_iou = None, IOU_MATCH
            for s in spots:
                v = _iou(d["box"], s["last"])
                if v >= best_iou:
                    best, best_iou = s, v
            if best is None:
                spots.append({"first": d["box"], "last": d["box"],
                              "obs": [(fr["t"], d["conf"], d["box"])]})
            else:
                best["last"] = d["box"]
                best["obs"].append((fr["t"], d["conf"], d["box"]))
    for s in spots:
        s["n"] = len(s["obs"])
        xs = [o[2] for o in s["obs"]]
        s["median"] = [sorted(b[i] for b in xs)[len(xs) // 2] for i in range(4)]
    return spots


def wobble(obs):
    """scenery.py's _settle: EMA 'home' box, wobble = displacement from it."""
    home, peak = None, 0.0
    for _t, _c, box in obs:
        home = ([float(v) for v in box] if home is None
                else [h + (b - h) * 0.05 for h, b in zip(home, box)])
        peak = max(peak, _displacement(box, home))
    return peak


def new_filter():
    return SceneryFilter(
        path=None, iou_match=SC["iou_match"], track_iou=SC["track_iou"],
        min_move=SC["min_move"], jitter_slack=SC["jitter_slack"],
        jitter_learn_secs=SC["jitter_learn_secs"],
        static_after_secs=SC["static_after_secs"], min_sightings=SC["min_sightings"],
        forget_secs=SC["forget_secs"], conf_certain=SC["conf_certain"],
        conf_override=SC["conf_override"])


def report():
    data = {}
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(CACHE, f)))
            data[j["clip"]] = j
    if not data:
        sys.exit(f"no cache in {CACHE} — run `{sys.argv[0]} run` first")
    sizes = sorted(next(iter(data.values()))["by_size"], key=int, reverse=True)
    ref = sizes[0]
    print(f"clips analysed: {len(data)}   sizes: {', '.join(sizes)}   reference: {ref}\n")

    index = {(c, sz, round(fr["t"], 3)): qualifying(fr["dets"])
             for c, j in data.items() for sz in sizes for fr in j["by_size"][sz]}

    def match(clip, size, t, box):
        for d in index.get((clip, size, round(t, 3)), ()):
            if _iou(d["box"], box) >= 0.5:
                return d
        return None

    # Furniture is found the way the scenery filter finds it: the same box recurring
    # across many clips, hours apart, cannot be an animal. No hand labelling.
    clusters = []
    for c, j in data.items():
        for s in build_spots(j["by_size"][ref]):
            for cl in clusters:
                if _iou(cl["box"], s["median"]) >= IOU_MATCH:
                    cl["members"].append((c, s))
                    break
            else:
                clusters.append({"box": s["median"], "members": [(c, s)]})
    for cl in clusters:
        cs = {c for c, _ in cl["members"]}
        span = (max(stem_secs(c) for c in cs) - min(stem_secs(c) for c in cs)
                if len(cs) > 1 else 0)
        cl["clips"], cl["fixed"] = len(cs), len(cs) >= 5 and span > 1800
    fixed = [cl for cl in clusters if cl["fixed"]]
    transient = [cl for cl in clusters if not cl["fixed"]]
    fixed_boxes = [cl["box"] for cl in fixed]
    print(f"spots: {len(clusters)} places  ({len(fixed)} FIXED = furniture, "
          f"{len(transient)} TRANSIENT = live candidates)\n")

    def on_furniture(box):
        return any(_iou(box, fb) >= IOU_MATCH for fb in fixed_boxes)

    print("=" * 76)
    print(f"A. RECALL — of what {ref} detected, how much survives?")
    print("=" * 76)
    for kind, group in (("TRANSIENT (live)", transient), ("FIXED (furniture)", fixed)):
        tot, got = 0, defaultdict(int)
        for cl in group:
            for clip, s in cl["members"]:
                for t, _conf, box in s["obs"]:
                    tot += 1
                    for sz in sizes[1:]:
                        got[sz] += 1 if match(clip, sz, t, box) else 0
        print(f"  {kind:20s} {tot:5d} @{ref}   " + "   ".join(
            f"{sz}: {got[sz]:5d} ({100*got[sz]/max(1,tot):5.1f}%)" for sz in sizes[1:]))

    print("\n  recall by object size (box geometric-mean side, 1080p pixels):")
    for lo, hi, name in [(0, 150, "small  <150px"), (150, 300, "medium 150-300"),
                         (300, 600, "large  300-600"), (600, 1e9, "huge    >600px")]:
        tot, got = 0, defaultdict(int)
        for cl in transient:
            for clip, s in cl["members"]:
                for t, _conf, box in s["obs"]:
                    if not lo <= _scale(box) < hi:
                        continue
                    tot += 1
                    for sz in sizes[1:]:
                        got[sz] += 1 if match(clip, sz, t, box) else 0
        if tot:
            print(f"    {name:16s} n={tot:5d}   " + "   ".join(
                f"{sz}: {100*got[sz]/tot:5.1f}%" for sz in sizes[1:]))

    print("\n  detections a smaller imgsz INVENTS (nothing at the reference there):")
    for sz in sizes[1:]:
        onf, other, by_cls = 0, 0, defaultdict(int)
        for c in data:
            for fr in data[c]["by_size"][sz]:
                for d in qualifying(fr["dets"]):
                    if match(c, ref, fr["t"], d["box"]):
                        continue
                    by_cls[d["cls"]] += 1
                    if on_furniture(d["box"]):
                        onf += 1
                    else:
                        other += 1
        print(f"    {sz}: {onf+other:5d} new   {onf:5d} on a known fixed spot, "
              f"{other:5d} elsewhere   " + " ".join(f"{k}={v}" for k, v in sorted(by_cls.items())))

    print("\n  the cases this system was built around (frames holding the subject /")
    print("  best confidence; detections on a known fixed spot excluded):")
    for clip, what in HARD.items():
        if clip not in data:
            continue
        cells = []
        for sz in sizes:
            real = [[d for d in qualifying(fr["dets"]) if not on_furniture(d["box"])]
                    for fr in data[clip]["by_size"][sz]]
            mx = max([d["conf"] for r in real for d in r] or [0])
            cells.append(f"{sz}: {sum(1 for r in real if r):3d}f/{mx:.2f}")
        print(f"    {what:46s} " + "  ".join(cells))

    print("\n" + "=" * 76)
    print("B. CAN A LOWER conf THRESHOLD COMPENSATE?")
    print("=" * 76)
    ref_obs = [(clip, t, box) for cl in transient for clip, s in cl["members"]
               for t, _c, box in s["obs"]]
    print(f"  {'imgsz':>6} {'conf':>6}   {'live kept':>20}   {'on furniture':>13}")
    for sz in sizes:
        for th in (0.30, 0.25, 0.20, 0.15, 0.10):
            kept = sum(1 for clip, t, box in ref_obs
                       if any(_iou(d["box"], box) >= 0.5 and d["conf"] >= th
                              for fr in data[clip]["by_size"][sz]
                              if abs(fr["t"] - t) < 1e-6 for d in fr["dets"]))
            furn = sum(1 for c in data for fr in data[c]["by_size"][sz] for d in fr["dets"]
                       if d["conf"] >= th and on_furniture(d["box"]))
            print(f"  {sz:>6} {th:>6.2f}   {kept:6d} / {len(ref_obs):<6d}"
                  f" ({100*kept/max(1,len(ref_obs)):5.1f}%)   {furn:10d}")
        print()

    print("=" * 76)
    print("C. BOX WOBBLE on fixed spots — what the scenery filter has to beat")
    print("=" * 76)
    print(f"  (rock 0.064 vs raccoon 0.062; min_move={SC['min_move']}, "
          f"jitter_slack={SC['jitter_slack']})\n")
    for cl in sorted(fixed, key=lambda c: -c["clips"])[:6]:
        print(f"  spot {cl['box']}  ({cl['clips']} clips)")
        for sz in sizes:
            ws, ns = [], 0
            for clip, _s in cl["members"]:
                for s2 in build_spots(data[clip]["by_size"][sz]):
                    if _iou(s2["median"], cl["box"]) >= IOU_MATCH and s2["n"] >= 3:
                        ws.append(wobble(s2["obs"]))
                        ns += s2["n"]
            print(f"     {sz:>4}: median {sorted(ws)[len(ws)//2]:.3f}  max {max(ws):.3f}"
                  f"   ({ns} detections in {len(ws)} clips)" if ws
                  else f"     {sz:>4}: not detected at all")
        print()

    print("=" * 76)
    print("D. WARM REPLAY — one SceneryFilter across the archive, in time order")
    print("=" * 76)
    print("  The closest model of a detector that has been watching: it carries what it")
    print("  learned from one clip into the next. (The archive holds only the clips that")
    print("  fired, not the continuous video, so it learns LESS than the real thing.)\n")
    kind_of = {}
    for c in data:
        kinds = {("FIXED" if cl["fixed"] else "TRANSIENT")
                 for cl in clusters for clip, _ in cl["members"] if clip == c}
        kind_of[c] = ("live" if "TRANSIENT" in kinds
                      else "furniture" if kinds else "empty")
    warm, per_clip = defaultdict(lambda: defaultdict(int)), defaultdict(dict)
    for sz in sizes:
        filt = new_filter()
        for c in sorted(data, key=stem_secs):
            t0 = stem_secs(c) - cfg["preroll"]
            hits, fired, last = deque(maxlen=cfg["window"]), 0, -1e9
            for fr in data[c]["by_size"][sz]:
                now = t0 + fr["t"]
                confirmed, unproven, _ = filt.apply(
                    [(d["cls"], d["conf"], d["box"]) for d in qualifying(fr["dets"])],
                    now=now)
                seen = [d for d in confirmed + unproven if d[0] in ("animal", "person")]
                fireable = [d for d in confirmed if d[0] in ("animal", "person")]
                hits.append(1 if seen else 0)
                if fireable and sum(hits) >= cfg["min_hits"] and now - last > cfg["cooldown_secs"]:
                    last, fired = now, fired + 1
            warm[kind_of[c]][sz] += 1 if fired else 0
            warm[kind_of[c]][sz + "_n"] += 1
            per_clip[c][sz] = fired
    for k in ("live", "furniture", "empty"):
        if warm[k].get(ref + "_n"):
            print(f"  {k:10s} ({warm[k][ref+'_n']:2d} clips)  fires: "
                  + "  ".join(f"{sz}={warm[k][sz]:2d}" for sz in sizes))
    print("\n  clips where a smaller imgsz changes the outcome:")
    for c in sorted(per_clip, key=stem_secs):
        f = per_clip[c]
        if any(f[sz] != f[ref] for sz in sizes[1:]):
            print(f"    {c:36s} {kind_of[c]:10s} "
                  + "  ".join(f"{sz}={f[sz]}" for sz in sizes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("run", "report"))
    ap.add_argument("--sizes", default="1280,960,640",
                    help="comma-separated imgsz values, largest first (default 1280,960,640)")
    a = ap.parse_args()
    if a.mode == "run":
        run([int(s) for s in a.sizes.split(",")])
    else:
        report()
