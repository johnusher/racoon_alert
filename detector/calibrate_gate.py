#!/usr/bin/env python3
"""
Measure the gate's open and shut, instead of guessing at them.

    h32 gate                     what does the gate score right now?
    h32 gate --set closed        shut the gate, run this, wait
    h32 gate --set open          open the gate, run this, wait
    h32 gate --apply             write the threshold into web/cameras.json
    h32 gate --camera gate       when there is more than one gate

The shipped `closed_above` is a guess and says so (`"calibrated": false`). The CLOSED end
was measured on the real gate — 0.678 — but nobody has opened it in front of the camera,
so the OPEN end comes from stand-in textures: pavement, path, foliage and hedge, which
scored 0.426-0.469. The threshold sits halfway between. That is a reasonable guess and it
is still a guess, and a guess is not what you want deciding whether a 2-year-old is
standing at an open gate.

Two runs of about twenty seconds each fix it. Frames come from the go2rtc snapshot
endpoint, which fans out the session the detector is ALREADY holding — this opens no new
connection to the camera, which matters on a VIMTAG that has fallen off the network
before when asked for two.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))
import h32env                                                    # noqa: E402
from gate import GateWatcher, is_daylight                        # noqa: E402

REGISTRY = os.path.join(os.path.dirname(BASE), "web", "cameras.json")


def store_path(camera):
    return os.path.join(BASE, f"gate.{camera}.json")


def grab(url, timeout=5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        buf = np.frombuffer(r.read(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def sample(watcher, url, secs, label=""):
    """Score frames for `secs` seconds. Returns the list of scores."""
    out, t0, warned = [], time.time(), False
    while time.time() - t0 < secs:
        try:
            frame = grab(url)
        except Exception as e:
            print(f"  cannot read a frame: {e}", file=sys.stderr)
            time.sleep(1.0)
            continue
        if frame is None:
            time.sleep(0.5)
            continue
        if not is_daylight(frame) and not warned:
            warned = True
            print("  ⚠️  this looks like an infra-red frame. The gate rules are daytime "
                  "only, so calibrate in daylight or the numbers describe the wrong scene.")
        s = watcher.score(frame)
        if s is not None:
            out.append(s)
            print(f"\r  {label}{len(out):3d} frames   latest {s:.3f}   "
                  f"mean {np.mean(out):.3f}  min {min(out):.3f}  max {max(out):.3f}   ",
                  end="", flush=True)
        time.sleep(0.4)
    print()
    return out


def recommend(closed, opened):
    """Threshold midway between the two clusters, with a deadband that covers their spread.

    Midway between the MEANS is wrong when one class is far noisier than the other, so
    this splits the gap between the two nearest edges instead — the worst closed reading
    and the best open one — which is the boundary that actually has to hold.
    """
    lo_closed, hi_open = min(closed), max(opened)
    gap = lo_closed - hi_open
    thr = hi_open + gap / 2.0
    band = max(0.01, gap / 4.0)
    return thr, band, gap


def apply_to_registry(camera, thr, band, path=REGISTRY):
    """Surgically rewrite the three numbers, leaving the file's comments and layout alone."""
    s = open(path).read()
    block = re.search(r'"id"\s*:\s*"%s".*?"gate"\s*:\s*\{.*?\}' % re.escape(camera), s, re.S)
    if not block:
        return False, f"no gate block for camera {camera!r} in {path}"
    b = block.group(0)
    nb = re.sub(r'("closed_above"\s*:\s*)[-\d.]+', lambda m: f"{m.group(1)}{thr:.3f}", b)
    nb = re.sub(r'("deadband"\s*:\s*)[-\d.]+', lambda m: f"{m.group(1)}{band:.3f}", nb)
    nb = re.sub(r'("calibrated"\s*:\s*)(true|false)', lambda m: f"{m.group(1)}true", nb)
    if nb == b:
        return False, "found the gate block but none of the three keys were in it"
    s = s[:block.start()] + nb + s[block.end():]
    json.loads(s)                                    # never leave the registry unparseable
    open(path, "w").write(s)
    return True, None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure a gate's open and shut.")
    ap.add_argument("--camera", default=None, help="camera id (default: the first with a gate)")
    ap.add_argument("--set", dest="label", choices=("closed", "open"), help="record this state")
    ap.add_argument("--secs", type=float, default=20.0, help="how long to sample")
    ap.add_argument("--apply", action="store_true", help="write the threshold into cameras.json")
    ap.add_argument("--no-write", action="store_true", help="recommend but do not write")
    a = ap.parse_args(argv)

    camera = a.camera
    if camera is None:
        for c in h32env.cameras():
            if (c.detect.get("gate") or {}).get("aperture"):
                camera = c.id
                break
    if camera is None:
        print("no camera has a `detect.gate` block in web/cameras.json", file=sys.stderr)
        return 1

    cfg = h32env.detector_config(camera=camera)
    g = cfg.get("gate") or {}
    if not g.get("aperture"):
        print(f"camera {camera!r} has no gate aperture configured", file=sys.stderr)
        return 1
    w = GateWatcher(g["aperture"], cfg.get("zone_space"),
                    closed_above=g.get("closed_above", 0.57),
                    deadband=g.get("deadband", 0.04),
                    read_band=g.get("read_band", (0.0, 0.45)))
    url = cfg["frame_url"]
    store = {}
    if os.path.exists(store_path(camera)):
        store = json.load(open(store_path(camera)))

    if a.label:
        print(f"gate [{camera}]: sampling {a.secs:.0f}s of the gate {a.label.upper()} — "
              f"leave it {a.label} and stay out of shot")
        got = sample(w, url, a.secs, label=f"{a.label}: ")
        if not got:
            print("no readable frames — is go2rtc up (h32 status)?", file=sys.stderr)
            return 1
        store[a.label] = got
        store["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(store, open(store_path(camera), "w"), indent=1)
        print(f"  saved {len(got)} '{a.label}' samples to {os.path.basename(store_path(camera))}")

    have = [k for k in ("closed", "open") if store.get(k)]
    if not a.label and not a.apply:
        try:
            s = w.score(grab(url))
        except Exception as e:
            print(f"cannot read a frame: {e}", file=sys.stderr)
            return 1
        print(f"gate [{camera}]: score {s:.3f}   configured closed_above "
              f"{g.get('closed_above'):.3f}±{g.get('deadband'):.3f}   "
              f"=> reads {'CLOSED' if s and s >= g.get('closed_above', 1) else 'open/unsure'}")
        print(f"  calibrated: {g.get('calibrated', False)}   samples on file: {have or 'none'}")
        return 0

    if len(have) < 2:
        print(f"\nhave {have or 'nothing'} — need both. Run the other one:")
        for k in ("closed", "open"):
            if k not in have:
                print(f"    h32 gate --set {k}")
        return 0

    thr, band, gap = recommend(store["closed"], store["open"])
    print(f"\n  closed: mean {np.mean(store['closed']):.3f}  "
          f"range {min(store['closed']):.3f}-{max(store['closed']):.3f}  n={len(store['closed'])}")
    print(f"  open:   mean {np.mean(store['open']):.3f}  "
          f"range {min(store['open']):.3f}-{max(store['open']):.3f}  n={len(store['open'])}")
    print(f"  gap between them: {gap:.3f}")
    if gap <= 0:
        print("\n  ⚠️  THE TWO OVERLAP. This measure cannot tell this gate's states apart —\n"
              "      check the aperture polygon is really on the gate (the outline is drawn\n"
              "      on the monitor), and that you sampled the states you meant to.")
        return 2
    if gap < 0.08:
        print("  ⚠️  that is a narrow gap; expect the odd unsure reading.")
    print(f"\n  recommend  closed_above {thr:.3f}   deadband {band:.3f}")
    if a.no_write:
        return 0
    ok, err = apply_to_registry(camera, thr, band)
    if not ok:
        print(f"  could not write it: {err}", file=sys.stderr)
        return 1
    print(f"  written to web/cameras.json (calibrated: true) — restart with `h32 detect`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
