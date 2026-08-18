#!/usr/bin/env python3
"""
Log the gate aperture's raw numbers through the night, so `min_edge` can be measured.

Not part of the detector and not on its path: it pulls frames from the go2rtc snapshot
endpoint, which fans out the session already open, and writes one line a minute. It
records the UNGUARDED share and absolute edge energy — the point is to find out where the
dark puts them, which a guarded reading would hide by returning None.

    ./.venv/bin/python detector/trace_gate_night.py [--camera gate] [--secs 60] [--hours 14]
"""
import argparse, json, os, sys, time, urllib.request
import cv2, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.dirname(BASE))
import h32env
from gate import GateWatcher, is_daylight

a = argparse.ArgumentParser()
a.add_argument("--camera", default="gate"); a.add_argument("--secs", type=float, default=60)
a.add_argument("--hours", type=float, default=14)
a = a.parse_args()

cfg = h32env.detector_config(camera=a.camera); g = cfg["gate"]
w = GateWatcher(g["aperture"], cfg.get("zone_space"), read_band=g.get("read_band", (0, .45)))
out = os.path.join(BASE, f"gate_night.{a.camera}.jsonl")
end = time.time() + a.hours * 3600
print(f"tracing {a.camera} every {a.secs:.0f}s for {a.hours}h -> {out}", flush=True)
with open(out, "a") as f:
    while time.time() < end:
        try:
            buf = np.frombuffer(urllib.request.urlopen(cfg["frame_url"], timeout=8).read(), np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            share, energy = w.edges(im)
            sat = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "share": None if share is None else round(share, 4),
                                "energy": None if energy is None else round(energy, 2),
                                "sat": round(sat, 2), "daylight": is_daylight(im)}) + "\n")
            f.flush()
        except Exception as e:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "error": str(e)}) + "\n")
            f.flush()
        time.sleep(a.secs)
