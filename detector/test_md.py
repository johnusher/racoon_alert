#!/usr/bin/env python3
"""Load MegaDetector v6 weights directly via ultralytics; test on raccoon frames."""
import glob, cv2, torch
from ultralytics import YOLO

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO("***REMOVED-PATH***/Documents/h32/detector/models/MDV6-yolov9-c.pt")
print("model classes:", model.names)

best_f, best_n, best_plot = None, -1, None
for f in sorted(glob.glob("***REMOVED-PATH***/Documents/h32/detector/samples/raccoon_*.jpg")):
    r = model(f, device=DEVICE, verbose=False, conf=0.2)[0]
    dets = [(model.names[int(b.cls)], round(float(b.conf), 2), [int(x) for x in b.xyxy[0]]) for b in r.boxes]
    print(f.split('/')[-1], "->", dets if dets else "(nothing)")
    if len(dets) > best_n:
        best_n, best_f, best_plot = len(dets), f, r.plot()

if best_plot is not None:
    out = "***REMOVED-PATH***/Documents/h32/detector/samples/_md_annotated.jpg"
    cv2.imwrite(out, best_plot)
    print(f"\nannotated ({best_f.split('/')[-1]}) -> {out}")
