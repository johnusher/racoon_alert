#!/usr/bin/env python3
"""
Enrol faces for the h32 detector, and measure whether the result can be trusted.

  enroll.py add <name> live [--secs 30]     grab faces from the camera as you walk about
  enroll.py add <name> <file|clip>...       ...or from stills / recorded clips
  enroll.py list                            who is enrolled, and from how many shots
  enroll.py remove <name>
  enroll.py test <file|clip>...             who does it think is there? (see below)

Enrol from CAMERA frames, not a phone selfie. Matching a daylight portrait against this
camera's night IR at a steep angle is a cross-domain problem and much harder than
like-for-like; the enrolled shots should look like what the detector will actually see.

`test` is the measurement that decides whether any of this is trustworthy. The 2026-08-16
spike showed the one person in the archive matches HIMSELF 10/10 — but with only one
identified human on file we could never measure whether he fails to match SOMEBODY ELSE.
So: enrol person A, then run `test` on a clip of person B. Every frame should come back
unknown. If B matches A, the false-accept rate is too high to act on and the thresholds
in config.json need raising before anything is wired to alerting.

Faces live in detector/faces_store.npz, which is GITIGNORED — this repo is public and
these are real people's biometrics.
"""
import os, sys, time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
import h32env
from faces import FaceIdentifier

cfg = h32env.detector_config(os.path.join(BASE, "config.json"))
FC = cfg.get("faces", {})
STORE = os.path.join(BASE, "faces_store.npz")
_md = None


def megadetector():
    """Loaded lazily — `list`/`remove` should not pay for it."""
    global _md
    if _md is None:
        import torch
        from ultralytics import YOLO
        dev = cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu"
        _md = (YOLO(os.path.join(BASE, cfg["model"])), dev)
    return _md


def identifier():
    ident = FaceIdentifier(
        os.path.join(BASE, "models"), STORE,
        min_face_px=FC.get("min_face_px", 45), det_score=FC.get("det_score", 0.7),
        threshold=FC.get("threshold", 0.40), margin=FC.get("margin", 0.10),
        vote_window_secs=FC.get("vote_window_secs", 45),
        min_votes=FC.get("min_votes", 2))
    if not ident.available:
        sys.exit("face models missing — run detector/get-face-models.sh")
    return ident


_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def people_in(frame):
    """Person boxes, found the same way detect.py finds them — CLAHE included, since
    without it MegaDetector loses most of the people in this camera's night IR."""
    model, dev = megadetector()
    img = frame
    if cfg.get("clahe", True):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    r = model(img, imgsz=cfg["imgsz"], conf=0.25, device=dev, verbose=False)[0]
    return [[int(x) for x in b.xyxy[0]] for b in r.boxes
            if model.names[int(b.cls)] == "person"]


def frames_from(source, every_s=0.4, secs=30):
    """Yield (label, frame) from a still, a clip, or the live camera."""
    if source == "live":
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(cfg["rtsp_main"])
        t0, last = time.time(), 0.0
        print(f"  live from {cfg['rtsp_main']} for {secs}s — walk into view, "
              f"look around, vary your angle…")
        while time.time() - t0 < secs:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time()
            if now - last >= every_s:
                last = now
                yield f"t{now - t0:.1f}s", frame
        cap.release()
        return
    if not os.path.exists(source):
        print(f"  !! no such file: {source}")
        return
    if source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = cv2.imread(source)
        if img is not None:
            yield os.path.basename(source), img
        return
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    step = max(1, int(round(fps * every_s)))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield f"{os.path.basename(source)}@{i / fps:.1f}s", frame
        i += 1
    cap.release()


def harvest(ident, sources, secs, verbose=True):
    """Faces from every source, gated on a person box (see faces.py for why)."""
    found = []
    for src in sources:
        for label, frame in frames_from(src, secs=secs):
            boxes = people_in(frame)
            if not boxes:
                # A deliberate portrait has no person box worth finding; fall back, but
                # only for stills — on camera frames an ungated search finds buckets.
                if src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    boxes = [[0, 0, frame.shape[1], frame.shape[0]]]
                else:
                    continue
            for box in boxes:
                for fb, emb, score, fh, _aligned in ident.faces_in_crop(frame, box):
                    found.append((label, emb, score, fh, frame, fb))
                    if verbose:
                        print(f"    {label:28} face {fh:5.0f}px  det={score:.2f}")
    return found


def cmd_add(args):
    if len(args) < 2:
        sys.exit("usage: enroll.py add <name> <live|file...> [--secs N]")
    name, sources = args[0], [a for a in args[1:] if not a.startswith("--")]
    secs = 30
    if "--secs" in args:
        secs = int(args[args.index("--secs") + 1])
    ident = identifier()
    print(f"enrolling '{name}' from {', '.join(sources)}")
    found = harvest(ident, sources, secs)
    if not found:
        sys.exit("  no faces found — nothing enrolled")

    # Drop shots that disagree with the rest: they are mis-detections, not hard poses.
    from faces import FaceMatcher
    keep, drop = FaceMatcher.consistent([f[1] for f in found],
                                        cut=ident.matcher.threshold)
    if drop:
        print(f"  dropped {len(drop)} shot(s) that did not look like the others "
              f"(mis-detections, not this person)")
    if len(keep) < 2:
        sys.exit("  too few consistent shots — try again with more/better footage")
    ident.matcher.enroll(name, [found[i][1] for i in keep])
    ident.save()

    sheet = os.path.join(BASE, f"faces_{name}.jpg")
    shots = [(i, found[i]) for i in keep[:24]] + [(i, found[i]) for i in drop[:8]]
    crops = []
    for i, f in shots:
        c = f[4][max(0, f[5][1]):f[5][3], max(0, f[5][0]):f[5][2]]
        if c.size:
            c = cv2.resize(c, (112, 112))
            if i in drop:                     # mark the rejects so the sheet explains itself
                cv2.rectangle(c, (0, 0), (111, 111), (60, 60, 220), 3)
                cv2.putText(c, "DROPPED", (4, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (60, 60, 220), 1)
            crops.append(c)
    if crops:
        cols = min(8, len(crops)); rows = (len(crops) + cols - 1) // cols
        out = np.zeros((rows * 112, cols * 112, 3), np.uint8)
        for k, c in enumerate(crops):
            r, cc = divmod(k, cols)
            out[r*112:(r+1)*112, cc*112:(cc+1)*112] = c
        cv2.imwrite(sheet, out)
    print(f"  enrolled {len(keep)} shot(s) of '{name}' -> {os.path.basename(STORE)}")
    if crops:
        print(f"  check they really are all the same person: {sheet}")
    print(f"  now enrolled: {ident.describe()}")


def cmd_list(_args):
    ident = identifier()
    if not ident.matcher.people:
        print("nobody enrolled yet — enroll.py add <name> live")
        return
    for n in ident.matcher.names():
        print(f"  {n:20} {ident.matcher.count(n):3} shot(s)")


def cmd_remove(args):
    if not args:
        sys.exit("usage: enroll.py remove <name>")
    ident = identifier()
    if args[0] not in ident.matcher.people:
        sys.exit(f"'{args[0]}' is not enrolled")
    ident.matcher.forget(args[0])
    ident.save()
    print(f"removed '{args[0]}' — now: {ident.describe()}")


def cmd_test(args):
    if not args:
        sys.exit("usage: enroll.py test <file|clip>...")
    ident = identifier()
    if not ident.matcher.people:
        sys.exit("nobody enrolled — nothing to test against")
    print(f"enrolled: {ident.describe()}")
    print(f"thresholds: match>={ident.matcher.threshold} margin>={ident.matcher.margin} "
          f"min_votes={ident.matcher.min_votes}\n")
    found = harvest(ident, args, secs=30, verbose=False)
    if not found:
        print("  no faces found at all")
        return
    tally = {}
    t = 0.0
    for label, emb, det, fh, _f, _b in found:
        name, score, gap = ident.matcher.match(emb)
        ident.matcher.observe_match(name, score, now=t); t += 1.0
        tally[name] = tally.get(name, 0) + 1
        print(f"  {label:28} {fh:5.0f}px -> {name or 'UNKNOWN':10} "
              f"score={score:.3f} margin={gap:+.3f}")
    verdict, votes, detail = ident.matcher.verdict(now=t)
    print(f"\n  per-face tally : " + ", ".join(f"{k or 'UNKNOWN'}={v}" for k, v in tally.items()))
    print(f"  visit verdict  : {verdict or 'UNKNOWN'}  ({detail})")
    print("\n  If this clip is somebody who is NOT enrolled, every line above should say")
    print("  UNKNOWN. Any other name is a false accept — raise `threshold`/`margin` in")
    print("  config.json before wiring identification to alerting.")


CMDS = {"add": cmd_add, "list": cmd_list, "remove": cmd_remove, "test": cmd_test}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    CMDS[sys.argv[1]](sys.argv[2:])
