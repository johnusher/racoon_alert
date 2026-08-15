#!/usr/bin/env python3
"""
h32 animal detector.

Polls the camera (via go2rtc), runs MegaDetector to find animals / people, applies
an optional pond ROI + confidence + temporal confirmation (robust to wind-shake and
fluttering leaves because it's pure object detection, no motion), and on a confirmed
event saves an annotated snapshot and a pre-roll clip (audio+video) via the recorder.

Special raccoon warning: MegaDetector reports generic 'animal'; classify_raccoon() is
the hook for a raccoon-specific stage (see README 'next steps').
"""
import os, sys, json, time, threading, urllib.request
from collections import deque
import numpy as np, cv2, torch
from ultralytics import YOLO
from recorder import CircularRecorder

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, "config.json")))
DEVICE = cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu"

model = YOLO(os.path.join(BASE, cfg["model"]))
NAMES = model.names                                   # {0:animal,1:person,2:vehicle}
conf_map = cfg["conf"]; min_conf = min(conf_map.values())
ignore = set(cfg.get("ignore_classes", []))
imgsz = cfg["imgsz"]; fps = cfg.get("fps", 3)
min_hits, window = cfg.get("min_hits", 2), cfg.get("window", 5)
cooldown = cfg.get("cooldown_secs", 30)
use_clahe = cfg.get("clahe", True)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
roi = np.array(cfg["roi"], np.int32) if cfg.get("roi") else None
exclude = np.array(cfg["exclude_roi"], np.int32) if cfg.get("exclude_roi") else None
events_dir = os.path.join(BASE, "events"); os.makedirs(events_dir, exist_ok=True)
logf = open(os.path.join(events_dir, "events.log"), "a")

rec = CircularRecorder(cfg["rtsp_main"], os.path.join(BASE, "buffer"), events_dir,
                       buffer_secs=cfg["buffer_secs"], seg_secs=cfg["seg_secs"],
                       preroll=cfg["preroll"], postroll=cfg["postroll"])

def grab():
    try:
        data = urllib.request.urlopen(cfg["frame_url"], timeout=5).read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def enhance(img):
    if not use_clahe: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def in_roi(box):
    pt = (int((box[0] + box[2]) / 2), int(box[3]))     # foot point (bottom-center)
    if roi is not None and cv2.pointPolygonTest(roi, pt, False) < 0: return False
    if exclude is not None and cv2.pointPolygonTest(exclude, pt, False) >= 0: return False
    return True

def classify_raccoon(crop):
    """Hook for a raccoon-specific classifier (Roboflow/CLIP). None = not yet implemented."""
    return None

def detect(img):
    r = model(img, imgsz=imgsz, conf=min_conf, device=DEVICE, verbose=False)[0]
    dets = []
    for b in r.boxes:
        cls = NAMES[int(b.cls)]; conf = float(b.conf); box = [int(x) for x in b.xyxy[0]]
        if cls in ignore or conf < conf_map.get(cls, 1.0) or not in_roi(box):
            continue
        dets.append((cls, conf, box))
    return dets, r

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    logf.write(line + "\n"); logf.flush()

def fire_event(img, r, dets, forced_tag=None):
    labels = {c for c, _, _ in dets}
    racc = None
    for c, cf, box in sorted((d for d in dets if d[0] == "animal"),
                             key=lambda d: -(d[2][2]-d[2][0])*(d[2][3]-d[2][1])):
        racc = classify_raccoon(img[box[1]:box[3], box[0]:box[2]]); break
    tag = forced_tag or ("RACCOON" if racc else ("ANIMAL" if "animal" in labels else "PERSON"))
    ts = time.strftime("%Y%m%d_%H%M%S"); name = f"{ts}_{tag.lower()}"
    cv2.imwrite(os.path.join(events_dir, f"{name}.jpg"), r.plot())
    status = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
    print(f"\n🔔 {tag}: {status}  → snapshot {name}.jpg, recording clip…")
    log(f"{tag}  {status}  snapshot={name}.jpg")
    threading.Thread(target=lambda n=name: log(f"clip: {os.path.basename(rec.save_event(n) or 'FAILED')}"),
                     daemon=True).start()
    return name

TEST = len(sys.argv) > 1 and sys.argv[1] == "test-event"
print(f"h32 detector: device={DEVICE}, model={cfg['model']}, {fps}fps"
      f"{' [TEST-EVENT]' if TEST else ''}. Ctrl-C to stop.")
rec.start()

if TEST:
    print("warming up buffer (8s)…"); time.sleep(8)
    img = grab()
    if img is None:
        print("no frame — is go2rtc (h32) running?"); rec.stop(); sys.exit(1)
    dets, r = detect(enhance(img))
    print(f"forcing a test event (real detections: {[(c, round(cf,2)) for c,cf,_ in dets]})")
    name = fire_event(img, r, dets or [("test", 1.0, [0, 0, 20, 20])], forced_tag="TEST")
    time.sleep(cfg["postroll"] + 5)
    print(f"done → events/{name}.jpg and events/{name}.mp4"); rec.stop(); sys.exit(0)

hits = deque(maxlen=window)
last_event = last_hb = 0.0
frames = 0
try:
    while True:
        t = time.time()
        img = grab()
        if img is None:
            print("\r[no frame — is go2rtc/h32 running?]     ", end="", flush=True)
            time.sleep(1); continue
        frames += 1
        dets, r = detect(enhance(img))
        interesting = [d for d in dets if d[0] in ("animal", "person")]
        hits.append(1 if interesting else 0)
        status = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
        print(f"\r[{time.strftime('%H:%M:%S')}] {status:44} hits={sum(hits)}/{len(hits)}   ", end="", flush=True)
        if t - last_hb > 15:                       # newline heartbeat so logs show progress
            last_hb = t
            print(f"\n[{time.strftime('%H:%M:%S')}] running: {frames} frames analyzed, latest: {status}")
        if interesting and sum(hits) >= min_hits and (t - last_event) > cooldown:
            last_event = t
            fire_event(img, r, interesting)
        dt = 1.0 / fps - (time.time() - t)
        if dt > 0: time.sleep(dt)
except KeyboardInterrupt:
    print("\nstopping…"); rec.stop(); logf.close()
