#!/usr/bin/env python3
"""
Mine the saved event clips for labelled-reference material — crop + SpeciesNet embedding.

WHY THIS EXISTS

SpeciesNet names the species on a crop, and for the raccoon and the cat it does. For the
hedgehog that walked through at 03:42 on 2026-08-17 it does not, and no setting fixes it:

    western european hedgehog   0.00004 - 0.00012
    blank                       0.51    - 0.96

measured over that clip at tight/+50%/+150%/+400% padding, on raw / CLAHE / gamma 0.5 /
histogram-equalised / 4x-upscaled crops, and on the full frame. Its top guesses are all
New World species (virginia opossum, central american agouti, white-lipped peccary) —
the classifier head simply does not cover a night-IR European hedgehog on this camera.

The 1280-d pooled feature UNDERNEATH that head is still a descriptor trained on 65M
camera-trap images, and it comes off the same forward pass for free. So a species this
model cannot name can still be recognised by matching that feature against crops labelled
here. This tool is how those crops get collected.

WHY RETROACTIVELY

Waiting for new visits would take nights. The archive already holds every animal event as
a clip, so the material is here now — including the hedgehog. Reading it offline also
costs the live detector nothing.

Only .mp4 clips are read, never the _*.jpg snapshots: those have the detection overlay
(red box, banner) burned in, and a reference set should not learn our own annotations.

Crops land in the existing gallery (kind "animal") so they share its manifest, its
near-duplicate filter and its layout; the per-kind rate limit is dropped to 0 here
because that limit is a live flood guard, and offline we want every distinct pose.
Labelling them is the next step — see label_animals.py.

Usage:
    detector/harvest_refs.py                    # every clip in detector/events/
    detector/harvest_refs.py a.mp4 b.mp4        # just these
    detector/harvest_refs.py --fps 2            # sample harder (default 1/s)
    detector/harvest_refs.py --dry-run          # report what it would bank
"""
import argparse
import glob
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="*", help="clips to read (default: detector/events/*.mp4)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second")
    ap.add_argument("--classes", default="animal",
                    help="comma-separated MegaDetector classes to bank (default: animal)")
    ap.add_argument("--dry-run", action="store_true", help="report, bank nothing")
    args = ap.parse_args()

    import cv2
    import torch
    import h32env
    from ultralytics import YOLO
    from gallery import Gallery
    from speciesnet import SpeciesNetClassifier

    cfg = h32env.detector_config(os.path.join(BASE, "config.json"))
    spc = cfg.get("species", {})
    events_dir = os.path.join(BASE, "events")
    clips = args.clips or sorted(glob.glob(os.path.join(events_dir, "*.mp4")))
    if not clips:
        sys.exit(f"no clips in {events_dir}")

    sn = SpeciesNetClassifier(os.path.join(BASE, spc.get("model", "")),
                              os.path.join(BASE, spc.get("labels", "")),
                              spc.get("human_veto", 0.25), spc.get("human_min", 0.45),
                              spc.get("species_min", 0.50))
    if not sn.available:
        sys.exit(f"speciesnet unavailable: {sn.describe()}")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    md = YOLO(os.path.join(BASE, cfg["model"]))
    gal = Gallery(os.path.join(BASE, "gallery"),
                  min_gap_secs=0.0,                     # offline: keep every distinct pose
                  dedup_cos=cfg.get("gallery", {}).get("dedup_cos", 0.94),
                  max_per_kind=cfg.get("gallery", {}).get("max_animals", 1500))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    want = {c.strip() for c in args.classes.split(",") if c.strip()}
    conf_map = cfg["conf"]

    def enhance(f):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    banked = seen = 0
    for path in clips:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        step = max(1, int(round(fps / max(args.fps, 0.01))))
        event = os.path.basename(path)
        # The manifest timestamps are what the label tool groups by, and two clips must
        # never collide; the clip's own mtime plus the offset into it keeps them ordered
        # and unique without inventing a clock.
        t0 = os.path.getmtime(path)
        n_here = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step:
                i += 1
                continue
            r = md(enhance(frame), imgsz=cfg["imgsz"], conf=min(conf_map.values()),
                   device=dev, verbose=False)[0]
            for b in r.boxes:
                cls = md.names[int(b.cls)]
                conf = float(b.conf)
                if cls not in want or conf < conf_map.get(cls, 1.0):
                    continue
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if not crop.size:
                    continue
                seen += 1
                v = sn.classify(crop, embed=True)
                if v is None or v.embedding is None:
                    continue
                if args.dry_run:
                    n_here += 1
                    continue
                name = gal.add("animal", crop,
                               {"event": event, "clip_t": round(i / fps, 2),
                                "conf": round(conf, 2), "box": [x1, y1, x2, y2],
                                "top": v.top_label, "top_p": round(v.top_p, 3),
                                "src": "harvest_refs"},
                               embedding=v.embedding, now=t0 + i / fps)
                if name:
                    n_here += 1
            i += 1
        cap.release()
        banked += n_here
        if n_here:
            print(f"  {event:<38} {n_here:>3} crop(s)")

    verb = "would bank" if args.dry_run else "banked"
    print(f"\n{verb} {banked} crop(s) from {len(clips)} clip(s) "
          f"({seen} detection(s) classified; near-duplicates dropped at "
          f"cos>={gal.dedup_cos})")
    if not args.dry_run:
        print(f"gallery now holds {gal.count('animal')} animal crop(s) — "
              f"label them with detector/label_animals.py")


if __name__ == "__main__":
    main()
