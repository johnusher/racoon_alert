#!/usr/bin/env python3
"""
h32 animal detector + live monitor + recorder.

Architecture (so the live video stays smooth even though the AI is slow):
  • capture thread  — reads the camera (go2rtc RTSP restream) continuously (~camera fps)
  • publish thread  — publishes the LATEST boxes for the monitor overlay, and an annotated
                      MJPEG fallback while anyone is watching one
  • detection loop  — runs MegaDetector as fast as it can (slower), updating the shared boxes
                      + firing events (snapshot, pre-roll clip, optional email)

Pure object detection (no motion) → robust to wind camera-shake and fluttering leaves.
Detections then pass the scenery filter (see scenery.py), which drops the static garden
furniture MegaDetector likes to call a person.

If the camera stops feeding us, the monitor says so instead of sitting on the last frame:
nothing is detected, no events fire, and the picture is replaced by a NO CAMERA SIGNAL card.
"""
import os, sys, json, time, threading, urllib.request, webbrowser
from collections import deque
import numpy as np, cv2, torch
from ultralytics import YOLO
from recorder import CircularRecorder
from server import MonitorServer
from notify import EmailNotifier
from scenery import SceneryFilter
from faces import FaceIdentifier
from gallery import Gallery
from species import SpeciesClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE))
import h32env                                             # camera/e-mail from local.env
import talk as talkmod                                    # send audio to the camera speaker

cfg = h32env.detector_config(os.path.join(BASE, "config.json"))
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
COLORS = {"animal": (80, 80, 255), "person": (210, 180, 60), "vehicle": (120, 120, 120),
          "cat": (80, 200, 255), "raccoon": (80, 80, 255)}   # species share the animal box
GREY = (110, 116, 124)                                    # suppressed-as-scenery boxes

sc = cfg.get("scenery", {})
scenery_on = sc.get("enabled", True)
scenery = SceneryFilter(os.path.join(BASE, "scenery.json"),
                        iou_match=sc.get("iou_match", 0.60),
                        track_iou=sc.get("track_iou", 0.30),
                        min_move=sc.get("min_move", 0.02),
                        jitter_slack=sc.get("jitter_slack", 2.5),
                        jitter_learn_secs=sc.get("jitter_learn_secs", 5.0),
                        static_after_secs=sc.get("static_after_secs", 180),
                        min_sightings=sc.get("min_sightings", 5),
                        forget_secs=sc.get("forget_secs", 1800),
                        conf_certain=sc.get("conf_certain", 0.70),
                        conf_override=sc.get("conf_override", 0.25))
signal_timeout = cfg.get("signal_timeout_secs", 8)        # no frames for this long = no signal

fc = cfg.get("faces", {})
faces_on = fc.get("enabled", True)
faces = FaceIdentifier(os.path.join(BASE, "models"), os.path.join(BASE, "faces_store.npz"),
                       min_face_px=fc.get("min_face_px", 45),
                       det_score=fc.get("det_score", 0.7),
                       threshold=fc.get("threshold", 0.40),
                       margin=fc.get("margin", 0.10),
                       vote_window_secs=fc.get("vote_window_secs", 45),
                       min_votes=fc.get("min_votes", 2))
known_suppresses = fc.get("known_suppresses_event", False)

# ---- species: split MegaDetector's "animal" into cat / raccoon / … ----
spc = cfg.get("species", {})
species_on = spc.get("enabled", True)
species = SpeciesClassifier(os.path.join(BASE, "species_refs.npz"),
                            min_sim=spc.get("min_sim", 0.55), margin=spc.get("margin", 0.04))
species_on = species_on and species.available and bool(species.matcher.refs)

# ---- crop harvester: collect faces + animal crops to learn from later ----
gc = cfg.get("gallery", {})
gallery_on = gc.get("enabled", True)
gallery = Gallery(os.path.join(BASE, "gallery"),
                  min_gap_secs=gc.get("min_gap_secs", 15),
                  dedup_cos=gc.get("dedup_cos", 0.94),
                  max_per_kind=max(gc.get("max_faces", 1500), gc.get("max_animals", 1500)))
# Faces run whenever we harvest, even before anyone is enrolled — that is how the
# household gets learned. Recognition (naming) needs enrollments; harvesting does not.
faces_available = faces.available
faces_on = faces_on and faces.available and bool(faces.matcher.people)

# ---- talk-back: greet a detected person through the camera speaker ----
tc = cfg.get("talk", {})
talk_on = tc.get("enabled", False) and bool(h32env.CAMERA_DEVID)
greet_pcm = None
if talk_on:
    try:                                                  # render the greeting once, up front
        greet_pcm = talkmod.say_to_pcm(tc.get("greet_text", "Hallo."),
                                       tc.get("greet_voice"))
    except Exception as e:
        print(f"    talk: greeting disabled ({e})"); talk_on = False
greet_on = tc.get("greet_on", "person")
greet_once = tc.get("greet_once", True)
greet_cooldown = tc.get("greet_cooldown_secs", 30)
_greet_state = {"done": False, "at": 0.0, "busy": False}
_greet_lock = threading.Lock()

def greet(tag_classes):
    """Play the greeting to the camera speaker on a background thread (never blocks
    detection). Fires on the configured class, once per run if greet_once."""
    if not talk_on or greet_pcm is None:
        return
    if greet_on not in tag_classes:
        return
    now = time.time()
    with _greet_lock:
        if _greet_state["busy"]: return
        if greet_once and _greet_state["done"]: return
        if now - _greet_state["at"] < greet_cooldown: return
        _greet_state["busy"] = True; _greet_state["at"] = now; _greet_state["done"] = True
    def _run():
        try:
            with talkmod.CameraTalk() as t:
                t.play(greet_pcm)
            print(f"\n[{time.strftime('%H:%M:%S')}] 🔊 greeted the camera ("
                  f"{tc.get('greet_text','Hallo.')!r})", flush=True)
        except Exception as e:
            print(f"\n[{time.strftime('%H:%M:%S')}] talk failed: {e}", flush=True)
        finally:
            with _greet_lock: _greet_state["busy"] = False
    threading.Thread(target=_run, daemon=True).start()

rec = CircularRecorder(cfg.get("rtsp_camera_direct") or cfg["rtsp_main"],
                       os.path.join(BASE, "buffer"), events_dir,
                       buffer_secs=cfg["buffer_secs"], seg_secs=cfg["seg_secs"],
                       preroll=cfg["preroll"], postroll=cfg["postroll"])
# The monitor page is served by go2rtc so it is same-origin with the WebRTC stream and
# gets the real video + audio; it pulls boxes and events from our own port.
MONITOR_URL = cfg.get("monitor_url") or "http://127.0.0.1:1984/monitor.html"
monitor = MonitorServer(cfg.get("monitor_port", 8090), events_dir, fps=display_fps,
                        monitor_url=MONITOR_URL)
notifier = EmailNotifier(os.path.join(BASE, "secrets.json"), cfg.get("email"))

# ---- shared state between threads ----
#   dets  — boxes we are acting on;  muted — boxes the scenery filter dropped (drawn grey)
#   frame_ts / signal — when the last frame arrived, and why none are arriving if they aren't
S = {"frame": None, "frame_ts": 0.0, "signal": "waiting for the first frame…",
     "dets": [], "muted": [], "faces": [], "recording": False, "banner": None, "run": True}
LK = threading.Lock()

def live_at(now, ts):
    """Is the camera feeding us? A frame older than signal_timeout means it is not."""
    return ts > 0 and (now - ts) <= signal_timeout

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

def detect(img):
    r = model(img, imgsz=imgsz, conf=min_conf, device=DEVICE, verbose=False)[0]
    dets = []
    for b in r.boxes:
        cls = NAMES[int(b.cls)]; conf = float(b.conf); box = [int(x) for x in b.xyxy[0]]
        if cls in ignore or conf < conf_map.get(cls, 1.0) or not in_roi(box):
            continue
        dets.append((cls, conf, box))
    return dets

def draw_overlay(img, dets, muted=(), recording=False, banner=None, face_hits=()):
    im = img.copy(); W = im.shape[1]
    for fb, who, sc in face_hits:
        col = (120, 230, 150) if who else (150, 200, 240)
        cv2.rectangle(im, (fb[0], fb[1]), (fb[2], fb[3]), col, 2)
        cv2.putText(im, f"{who or '?'} {sc:.2f}", (fb[0], max(14, fb[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if roi is not None: cv2.polylines(im, [roi], True, (0, 200, 120), 2)
    for c, cf, box, *_ in muted:                          # scenery: shown, never acted on
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), GREY, 1)
        cv2.putText(im, f"{c} {cf:.2f} scenery", (box[0], max(18, box[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
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

def no_signal_card(reason, last_ts, size):
    """What the monitor shows instead of a frozen frame with a ticking clock on it."""
    h, w = size
    im = np.full((h, w, 3), (24, 20, 18), np.uint8)
    cv2.rectangle(im, (0, 0), (w, 30), (15, 17, 20), -1)
    cv2.putText(im, f"h32 detector   {time.strftime('%Y-%m-%d %H:%M:%S')}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 235), 1)
    seen = time.strftime('%H:%M:%S', time.localtime(last_ts)) if last_ts else "never"
    ago = f" ({int(time.time() - last_ts)}s ago)" if last_ts else ""
    for text, scale, thick, col, dy in (
            ("NO CAMERA SIGNAL", 1.3, 3, (90, 90, 255), 0),
            (reason or "no video from the camera", 0.62, 1, (200, 205, 212), 58),
            (f"last frame: {seen}{ago}", 0.62, 1, (150, 156, 165), 98),
            ("detector still running — it reconnects on its own", 0.55, 1, (120, 126, 135), 140)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(im, text, ((w - tw) // 2, int(h * 0.42) + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick)
    return im

def push(im):
    if im.shape[1] > 1280:
        im = cv2.resize(im, (1280, int(1280 * im.shape[0] / im.shape[1])))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok: monitor.update_frame(buf.tobytes())

def log(msg):
    logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n"); logf.flush()

def fire_event(img, dets, forced_tag=None, who=None, who_detail=""):
    labels = {c for c, _, _ in dets}
    species_label = None                                  # cat / raccoon / … or None=unknown
    for c, cf, box in sorted((d for d in dets if d[0] == "animal"),
                             key=lambda d: -(d[2][2]-d[2][0])*(d[2][3]-d[2][1])):
        crop = img[max(0, box[1]):box[3], max(0, box[0]):box[2]]
        if species_on and crop.size:
            species_label, _sim = species.classify(crop)
        break
    tag = forced_tag or (species_label.upper() if species_label
                         else ("ANIMAL" if "animal" in labels else "PERSON"))
    detail = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
    if "person" in labels and faces_on:
        detail += f"  who={who or 'unknown'}"
    where = " ".join(f"{c}@{box}" for c, _, box in dets) or "-"
    ts = time.strftime("%Y%m%d_%H%M%S"); name = f"{ts}_{tag.lower()}"; snap = f"{name}.jpg"
    banner = f"{tag} · {who.upper()}" if who else tag
    cv2.imwrite(os.path.join(events_dir, snap), draw_overlay(img, dets, banner=banner))
    print(f"\n🔔 {tag}: {detail}  → snapshot {snap}, recording clip…")
    # Log the box coords too: a false positive that keeps firing from the same spot is
    # then obvious in events.log, which is how the bench in scenery.py was tracked down.
    log(f"{tag}  {detail}  {where}  snapshot={snap}"
        + (f"  faces[{who_detail}]" if who_detail else ""))
    monitor.add_event(tag, snap, detail, who=who)
    notifier.maybe_alert(tag, detail, os.path.join(events_dir, snap), who=who)
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
            fails += 1
            with LK:
                S["signal"] = (f"no video on {src}"
                               + (" — go2rtc up? camera powered?" if fails > 30 else ""))
            time.sleep(0.3)
            if fails % 15 == 0:                            # periodic reconnect attempt
                cap.release(); cap = cv2.VideoCapture(src)
            continue
        fails = 0
        with LK: S["frame"], S["frame_ts"], S["signal"] = frame, time.time(), None
    cap.release()

def publish_loop():
    """Publish boxes for the monitor overlay; encode the MJPEG only while someone watches."""
    period = 1.0 / max(1, display_fps)
    while S["run"]:
        now = time.time()
        with LK:
            f = S["frame"]; ts = S["frame_ts"]; why = S["signal"]
            d = S["dets"]; m = S["muted"]; r = S["recording"]; b = S["banner"]
            fh = S["faces"]
        live = live_at(now, ts)
        monitor.set_state(live, why, r,
                          [{"cls": c, "conf": round(cf, 2), "box": box, "scenery": False}
                           for c, cf, box in d] +
                          [{"cls": c, "conf": round(cf, 2), "box": box, "scenery": True,
                            "why": rsn} for c, cf, box, rsn in m],
                          f.shape[1] if f is not None else 0,
                          f.shape[0] if f is not None else 0, ts,
                          [{"box": fb, "who": who, "score": round(sc, 2)}
                           for fb, who, sc in fh])
        if monitor.mjpeg_clients:
            if live and f is not None:
                push(draw_overlay(f, d, m, r, b, fh))
            else:
                push(no_signal_card(why, ts, f.shape[:2] if f is not None else (720, 1280)))
        time.sleep(period)

# ---- run ----
TEST = len(sys.argv) > 1 and sys.argv[1] == "test-event"
url = MONITOR_URL
monitor.start(); rec.start()
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=publish_loop, daemon=True).start()
print(f"h32 detector: device={DEVICE}, model={cfg['model']}, detect~{det_fps}fps, display {display_fps}fps"
      f"{' [TEST-EVENT]' if TEST else ''}")
print(f"    scenery filter: {'on' if scenery_on else 'OFF'} — {scenery.describe()}")
print(f"    face id: {'on' if faces_on else 'off'} — {faces.describe()}"
      + ("   (known people SUPPRESS events)" if faces_on and known_suppresses else ""))
print(f"    species: {'on' if species_on else 'off'} — {species.describe()}")
print(f"    gallery: {'on' if gallery_on else 'off'} — harvesting crops to learn from"
      f" ({gallery.describe()})" if gallery_on else "    gallery: off")
print(f"    talk: {'on' if talk_on else 'off'}"
      + (f" — says {tc.get('greet_text','Hallo.')!r} ({tc.get('greet_voice','default')}) "
         f"on {'the first' if greet_once else 'each'} {greet_on}"
         f"{'' if greet_once else f', ≤1/{greet_cooldown}s'}"
         if talk_on else " (set H32_CAMERA_DEVID in local.env)"))
print(f"👁  LIVE MONITOR: {url}   (live video + boxes; records + alerts on detection)")
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
        with LK: img, ts, why = S["frame"], S["frame_ts"], S["signal"]
        if not live_at(t, ts):                             # camera gone: detect nothing, fire nothing
            with LK:
                S["dets"], S["muted"], S["faces"], S["banner"] = [], [], [], None
                S["recording"] = t < record_until          # the recorder reads the camera
            hits.clear()                                   # directly, so REC may still be true
            if t - last_hb > 15:
                last_hb = t
                print(f"\n[{time.strftime('%H:%M:%S')}] NO CAMERA SIGNAL — "
                      f"{why or 'no frames'}; paused until the feed returns", flush=True)
            time.sleep(0.5); continue
        frames += 1
        raw = detect(enhance(img))
        # confirmed = has moved, may fire.  unproven = believed, shown and counted, but
        # not allowed to fire by itself.  muted = known scenery, ignored entirely.
        confirmed, unproven, muted = scenery.apply(raw, now=t) if scenery_on else (raw, [], [])
        dets = confirmed + unproven
        interesting = [d for d in dets if d[0] in ("animal", "person")]
        fireable = [d for d in confirmed if d[0] in ("animal", "person")]
        hits.append(1 if interesting else 0)
        # Faces are only looked for INSIDE a person box the scenery filter has cleared —
        # an ungated search finds the plastic bucket (see faces.py). Run them whenever we
        # recognise OR harvest, so the household gets learned even before anyone's enrolled.
        persons = [b for c, _, b in dets if c == "person"]
        run_faces = (faces_on or gallery_on) and faces_available and persons
        face_hits = faces.observe(img, persons, now=t) if run_faces else []
        # Harvest crops to learn from later (rate-limited + deduped inside the gallery).
        if gallery_on:
            for aligned, emb, name in getattr(faces, "harvest", []):
                gallery.add("face", aligned, {"who": name}, embedding=emb, now=t)
            for c, cf, box in confirmed:
                if c == "animal":
                    a = img[max(0, box[1]):box[3], max(0, box[0]):box[2]]
                    if a.size:
                        gallery.add("animal", a, {"conf": round(cf, 2), "box": box}, now=t)
        recording = t < record_until
        banner = interesting[0][0].upper() if (recording and interesting) else None
        with LK:
            S["dets"], S["muted"], S["recording"], S["banner"] = dets, muted, recording, banner
            S["faces"] = face_hits
        status = " ".join(f"{c}:{cf:.2f}" for c, cf, _ in dets) or "-"
        if muted: status += f" [{len(muted)} scenery]"
        if face_hits: status += " " + " ".join(f"<{n or '?'}>" for _, n, _ in face_hits)
        print(f"\r[{time.strftime('%H:%M:%S')}] {status:44} hits={sum(hits)}/{len(hits)}   ", end="", flush=True)
        if t - last_hb > 15:
            last_hb = t; print(f"\n[{time.strftime('%H:%M:%S')}] running: {frames} detections done, "
                               f"latest: {status}  ({scenery.describe()})  (monitor {url})")
        if fireable and sum(hits) >= min_hits and (t - last_event) > cooldown:
            who, _votes, who_detail = faces.verdict(t) if faces_on else (None, 0, "")
            is_person = any(c == "person" for c, _, _ in fireable)
            if known_suppresses and who and is_person:
                last_event = t                             # recognised: stay quiet
                print(f"\n[{time.strftime('%H:%M:%S')}] known person ({who}) — event suppressed"
                      f"  [{who_detail}]", flush=True)
            else:
                last_event = t; record_until = t + cfg["postroll"] + cfg["seg_secs"]
                fire_event(img, fireable, who=who, who_detail=who_detail)
                greet({c for c, _, _ in fireable})         # "Hallo." on a real person
            faces.reset()                                  # next visit votes on its own
        dt = 1.0 / det_fps - (time.time() - t)
        if dt > 0: time.sleep(dt)
except KeyboardInterrupt:
    print("\nstopping…"); S["run"] = False; rec.stop(); scenery.save(); logf.close()
