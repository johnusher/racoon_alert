#!/usr/bin/env python3
"""
h32 animal detector + live monitor + recorder.

Architecture (so the live video stays smooth even though the AI is slow):
  • capture thread  — reads the camera (go2rtc RTSP restream) continuously (~camera fps)
  • publish thread  — overlays the LATEST boxes on the LATEST frame and pushes the monitor
                      MJPEG at display_fps (smooth)
  • detection loop  — runs MegaDetector as fast as it can (slower), updating the shared boxes
                      + firing events (snapshot, pre-roll clip, optional email)

Pure object detection (no motion) → robust to wind camera-shake and fluttering leaves.
"""
import os, sys, json, time, threading, urllib.request, webbrowser
from collections import deque
import numpy as np, cv2, torch
from ultralytics import YOLO
from recorder import CircularRecorder
from server import MonitorServer
from notify import EmailNotifier

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, "config.json")))
DEVICE = cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu"

model = YOLO(os.path.join(BASE, cfg["model"]))
NAMES = model.names                                       # {0:animal,1:person,2:vehicle}
conf_map = cfg["conf"]; min_conf = min(conf_map.values())
ignore = set(cfg.get("ignore_classes", []))
imgsz = cfg["imgsz"]
det_fps = cfg.get("fps", 3); display_fps = cfg.get("display_fps", 12)
min_hits, window = cfg.get("min_hits", 2), cfg.get("window", 5)
cooldown = cfg.get("cooldown_secs", 30)
use_clahe = cfg.get("clahe", True)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
roi = np.array(cfg["roi"], np.int32) if cfg.get("roi") else None
exclude = np.array(cfg["exclude_roi"], np.int32) if cfg.get("exclude_roi") else None
events_dir = os.path.join(BASE, "events"); os.makedirs(events_dir, exist_ok=True)
logf = open(os.path.join(events_dir, "events.log"), "a")
COLORS = {"animal": (80, 80, 255), "person": (210, 180, 60), "vehicle": (120, 120, 120)}

rec = CircularRecorder(cfg["rtsp_main"], os.path.join(BASE, "buffer"), events_dir,
                       buffer_secs=cfg["buffer_secs"], seg_secs=cfg["seg_secs"],
                       preroll=cfg["preroll"], postroll=cfg["postroll"])
monitor = MonitorServer(cfg.get("monitor_port", 8090), events_dir, fps=display_fps)
notifier = EmailNotifier(os.path.join(BASE, "secrets.json"), cfg.get("email"))

# ---- shared state between threads ----
S = {"frame": None, "dets": [], "recording": False, "banner": None, "run": True}
LK = threading.Lock()

def enhance(img):
    if not use_clahe: return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def in_roi(box):
    pt = (int((box[0] + box[2]) / 2), int(box[3]))       # foot point
    if roi is not None and cv2.pointPolygonTest(roi, pt, False) < 0: return False
    if exclude is not None and cv2.pointPolygonTest(exclude, pt, False) >= 0: return False
    return True

def classify_raccoon(crop):
    """Hook for a raccoon-specific classifier (Roboflow/CLIP). None = not implemented yet."""
    return None

def detect(img):
    r = model(img, imgsz=imgsz, conf=min_conf, device=DEVICE, verbose=False)[0]
    dets = []
    for b in r.boxes:
        cls = NAMES[int(b.cls)]; conf = float(b.conf); box = [int(x) for x in b.xyxy[0]]
        if cls in ignore or conf < conf_map.get(cls, 1.0) or not in_roi(box):
            continue
        dets.append((cls, conf, box))
    return dets

def draw_overlay(img, dets, recording=False, banner=None):
    im = img.copy(); W = im.shape[1]
    if roi is not None: cv2.polylines(im, [roi], True, (0, 200, 120), 2)
    for c, cf, box in dets:
        col = COLORS.get(c, (80, 80, 255))
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), col, 2)
        cv2.putText(im, f"{c} {cf:.2f}", (box[0], max(18, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.rectangle(im, (0, 0), (W, 30), (15, 17, 20), -1)
    cv2.putText(im, f"h32 detector   {time.strftime('%Y-%m-%d %H:%M:%S')}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 235), 1)
    if recording:
        cv2.circle(im, (W - 96, 15), 7, (80, 80, 255), -1)
        cv2.putText(im, "REC", (W - 82, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    if banner:
        cv2.putText(im, f"{banner} DETECTED", (W // 2 - 150, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 200, 255), 3)
    return im

def publish(img, dets, recording=False, banner=None):
    im = draw_overlay(img, dets, recording, banner)
    if im.shape[1] > 1280:
        im = cv2.resize(im, (1280, int(1280 * im.shape[0] / im.shape[1])))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok: monitor.update_frame(buf.tobytes())

def log(msg):
    logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n"); logf.flush()

def fire_event(img, dets, forced_tag=None):
    labels = {c for c, _, _ in dets}
    racc = None
    for c, cf, box in sorted((d for d in dets if d[0] == "animal"),
                             key=lambda d: -(d[2][2]-d[2][0])*(d[2][3]-d[2][1])):
        racc = classify_raccoon(img[box[1]:box[3], box[0]:box[2]]); break
    tag = forced_tag or ("RACCOON" if racc else ("ANIMAL" if "animal" in labels else "PERSON"))
    detail = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
    ts = time.strftime("%Y%m%d_%H%M%S"); name = f"{ts}_{tag.lower()}"; snap = f"{name}.jpg"
    cv2.imwrite(os.path.join(events_dir, snap), draw_overlay(img, dets, banner=tag))
    print(f"\n🔔 {tag}: {detail}  → snapshot {snap}, recording clip…")
    log(f"{tag}  {detail}  snapshot={snap}")
    monitor.add_event(tag, snap, detail)
    notifier.maybe_alert(tag, detail, os.path.join(events_dir, snap))
    def _save():
        clip = rec.save_event(name)
        if clip: monitor.set_clip(snap, os.path.basename(clip))
        log(f"clip: {os.path.basename(clip) if clip else 'FAILED'}")
    threading.Thread(target=_save, daemon=True).start()
    return name

# ---- threads ----
def capture_loop():
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    src = cfg["rtsp_main"]
    cap = cv2.VideoCapture(src)
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    fails = 0
    while S["run"]:
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1; time.sleep(0.3)
            if fails % 15 == 0:
                cap.release(); cap = cv2.VideoCapture(src)
            continue
        fails = 0
        with LK: S["frame"] = frame
    cap.release()

def publish_loop():
    period = 1.0 / max(1, display_fps)
    while S["run"]:
        with LK:
            f = S["frame"]; d = S["dets"]; r = S["recording"]; b = S["banner"]
        if f is not None:
            publish(f, d, r, b)
        time.sleep(period)

# ---- run ----
TEST = len(sys.argv) > 1 and sys.argv[1] == "test-event"
url = f"http://127.0.0.1:{cfg.get('monitor_port', 8090)}/"
monitor.start(); monitor.status = "live"; rec.start()
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=publish_loop, daemon=True).start()
print(f"h32 detector: device={DEVICE}, model={cfg['model']}, detect~{det_fps}fps, display {display_fps}fps"
      f"{' [TEST-EVENT]' if TEST else ''}")
print(f"👁  LIVE MONITOR: {url}   (smooth video + boxes; records + alerts on detection)")
if cfg.get("open_browser", True) and not TEST and not os.environ.get("H32_NO_BROWSER"):
    try: webbrowser.open(url)
    except Exception: pass

for _ in range(60):                                        # wait for first frame
    with LK: ready = S["frame"] is not None
    if ready: break
    time.sleep(0.25)

if TEST:
    with LK: img = S["frame"]
    if img is None: print("no frame — is go2rtc (h32) running?"); S["run"] = False; rec.stop(); sys.exit(1)
    dets = detect(enhance(img))
    with LK: S["dets"], S["recording"], S["banner"] = dets, True, "TEST"
    print(f"forcing test event (real detections: {[(c, round(cf,2)) for c,cf,_ in dets]})")
    name = fire_event(img, dets or [("test", 1.0, [40, 40, 300, 300])], forced_tag="TEST")
    time.sleep(cfg["postroll"] + 5)
    print(f"done → events/{name}.jpg + events/{name}.mp4 (monitor: {url})"); S["run"] = False; rec.stop(); sys.exit(0)

hits = deque(maxlen=window)
last_event = last_hb = record_until = 0.0
frames = 0
try:
    while True:
        t = time.time()
        with LK: img = S["frame"]
        if img is None:
            monitor.status = "no camera frame"; time.sleep(0.2); continue
        frames += 1
        dets = detect(enhance(img))
        interesting = [d for d in dets if d[0] in ("animal", "person")]
        hits.append(1 if interesting else 0)
        recording = t < record_until
        banner = interesting[0][0].upper() if (recording and interesting) else None
        with LK: S["dets"], S["recording"], S["banner"] = dets, recording, banner
        monitor.recording = recording
        status = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
        print(f"\r[{time.strftime('%H:%M:%S')}] {status:44} hits={sum(hits)}/{len(hits)}   ", end="", flush=True)
        if t - last_hb > 15:
            last_hb = t; print(f"\n[{time.strftime('%H:%M:%S')}] running: {frames} detections done, latest: {status}  (monitor {url})")
        if interesting and sum(hits) >= min_hits and (t - last_event) > cooldown:
            last_event = t; record_until = t + cfg["postroll"] + cfg["seg_secs"]
            fire_event(img, interesting)
        dt = 1.0 / det_fps - (time.time() - t)
        if dt > 0: time.sleep(dt)
except KeyboardInterrupt:
    print("\nstopping…"); S["run"] = False; rec.stop(); logf.close()
