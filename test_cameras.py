#!/usr/bin/env python3
"""
Tests for the camera registry (h32env.cameras).

The registry is what makes 1, 2 and 3 cameras the same code path, so the cases that
matter most here are the ABSENT ones: a camera listed in cameras.json but with no URL in
local.env must vanish completely — no tile, no detector, no events — rather than becoming
a half-camera that everything downstream has to special-case.

Run:  ./.venv/bin/python test_cameras.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h32env

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def raises(fn):
    """Did fn() raise? Used where failing loudly is the required behaviour."""
    try:
        fn()
        return False
    except Exception:
        return True


REGISTRY = {
    "cameras": [
        {"id": "west", "name": "West · patio & pond", "short": "West",
         "kit": "Victure PC530", "monitor_port": 8090, "zone_space": [1920, 1080],
         "detect": {"conf": {"animal": 0.20}}, "zones": [], "rules": []},
        {"id": "south", "name": "South · lawn & car port", "short": "South",
         "kit": "Vimtag 2.5K", "monitor_port": 8091, "zone_space": [2560, 1440],
         "detect": {}, "zones": [], "rules": []},
        {"id": "gate", "name": "East · entrance gate", "short": "Gate",
         "kit": "Vimtag 2.5K", "monitor_port": 8092, "zone_space": [2560, 1440],
         "detect": {"conf": {"person": 0.25}}, "zones": [], "rules": []},
    ]
}

# west and south configured, gate still boxed — the exact 2-camera case as of 2026-08-17.
ENV = {
    "H32_CAM_WEST": "rtsp://admin:pw@192.168.1.216:554/realmonitor?channel=0&stream=0.sdp",
    "H32_CAM_WEST_SUB": "rtsp://admin:pw@192.168.1.216:554/realmonitor?channel=0&stream=1.sdp",
    "H32_CAM_SOUTH": "onvif://admin:pw@192.168.1.124:80?subtype=0",
    "H32_CAM_SOUTH_SUB": "onvif://admin:pw@192.168.1.124:80?subtype=1",
    "H32_CAM_GATE": "",
    "H32_CAM_GATE_SUB": "",
}

TMP = tempfile.mkdtemp(prefix="h32cams")
REG = os.path.join(TMP, "cameras.json")
open(REG, "w").write(json.dumps(REGISTRY))


def cams(env=None):
    return h32env.cameras(registry_path=REG, env=ENV if env is None else env)


def by_id(env=None):
    return {c.id: c for c in cams(env)}


print("\n1. the list, and what counts as a camera that exists")
check("all three are listed in file order",
      [c.id for c in cams()] == ["west", "south", "gate"])
check("a camera with a URL is configured", by_id()["west"].configured and by_id()["south"].configured)
check("a camera with a BLANK url is not configured", not by_id()["gate"].configured)
check("a camera missing from local.env entirely is not configured",
      not by_id({k: v for k, v in ENV.items() if not k.startswith("H32_CAM_GATE")})["gate"].configured)
# go2rtc leaves an unset ${VAR} literal in place rather than blanking it (measured), so a
# camera whose placeholder never got expanded must not read as configured either.
check("an unexpanded ${…} placeholder is not configured",
      not by_id(dict(ENV, H32_CAM_GATE="${H32_CAM_GATE}"))["gate"].configured)

print("\n2. one, two or three cameras are the same code path")
check("two configured", [c.id for c in h32env.configured_cameras(REG, ENV)] == ["west", "south"])
check("one configured",
      [c.id for c in h32env.configured_cameras(REG, {"H32_CAM_WEST": ENV["H32_CAM_WEST"]})] == ["west"])
check("three configured",
      [c.id for c in h32env.configured_cameras(
          REG, dict(ENV, H32_CAM_GATE="onvif://a:b@10.0.0.9:80?subtype=0"))]
      == ["west", "south", "gate"])
check("zero configured is not an error", h32env.configured_cameras(REG, {}) == [])

print("\n3. lookup")
check("by id", h32env.camera("south", REG, ENV).short == "South")
check("an unknown id is None, not an exception", h32env.camera("nope", REG, ENV) is None)

print("\n4. urls — the protocol difference is data, not code")
check("the Victure keeps its rtsp:// scheme", by_id()["west"].url.startswith("rtsp://"))
check("the Vimtag keeps its onvif:// scheme", by_id()["south"].url.startswith("onvif://"))
check("sub stream is its own url", by_id()["south"].url_sub.endswith("subtype=1"))
check("a camera with no sub stream falls back to main, rather than breaking",
      by_id(dict(ENV, H32_CAM_SOUTH_SUB=""))["south"].url_sub == by_id()["south"].url)
check("public() never leaks a url (they carry passwords)",
      not any("://" in str(v) for v in by_id()["south"].public().values()))

print("\n5. every camera gets its own paths — scenery belongs to a PLACE")
west, south = h32env.camera("west", REG, ENV), h32env.camera("south", REG, ENV)
check("stream names are the camera id", (south.stream, south.stream_sub) == ("south", "south_sub"))
check("ports differ", west.monitor_port == 8090 and south.monitor_port == 8091)
check("events dirs differ", west.events_dir != south.events_dir)
check("scenery files differ", west.scenery_path != south.scenery_path)
check("buffer dirs differ", west.buffer_dir != south.buffer_dir)
check("scenery is named for the camera", south.scenery_path.endswith("scenery.south.json"))
check("events live under the camera", south.events_dir.endswith(os.path.join("events", "south")))
# A port derived from position would move when a camera is added, silently pointing the
# monitor tile at the wrong detector.
env_no_west = {k: v for k, v in ENV.items() if not k.startswith("H32_CAM_WEST")}
live = h32env.configured_cameras(REG, env_no_west)
check("a camera's port does not move when another is removed",
      [c.id for c in live] == ["south"] and live[0].monitor_port == 8091)

print("\n6. zone_space, zones and rules")
check("zones and rules default to empty — every camera behaves as today until one is drawn",
      all(c.zones == [] and c.rules == [] for c in cams()))
check("zone_space is per camera", west.zone_space == [1920, 1080] and south.zone_space == [2560, 1440])
reg2 = json.loads(json.dumps(REGISTRY))
del reg2["cameras"][0]["zone_space"]
p2 = os.path.join(TMP, "r2.json")
open(p2, "w").write(json.dumps(reg2))
check("zone_space defaults when absent",
      h32env.camera("west", p2, ENV).zone_space == [1920, 1080])

print("\n7. config merge — a camera states only what differs")
base = os.path.join(TMP, "config.json")
open(base, "w").write(json.dumps({
    "imgsz": 1280,
    "conf": {"animal": 0.20, "person": 0.30},
    "scenery": {"enabled": True, "min_move": 0.02},
}))
cfg = h32env.detector_config(base, camera="gate", registry_path=REG, env=ENV)
check("an untouched default survives", cfg["imgsz"] == 1280)
check("the camera override wins", cfg["conf"]["person"] == 0.25)
check("a sibling key is not clobbered by the override", cfg["conf"]["animal"] == 0.20)
check("a nested default survives", cfg["scenery"]["min_move"] == 0.02)

cfg_s = h32env.detector_config(base, camera="south", registry_path=REG, env=ENV)
check("config carries the camera's own frame url", cfg_s["frame_url"].endswith("src=south"))
check("config carries the camera's own rtsp url", cfg_s["rtsp_main"].endswith("/south"))
check("config carries the camera's own port", cfg_s["monitor_port"] == 8091)
check("config carries the camera's own paths",
      cfg_s["scenery_path"].endswith("scenery.south.json"))
# The recorder hands rtsp_camera_direct to ffmpeg. onvif:// is a go2rtc-only scheme, so
# passing it through would give a recorder that starts fine and never writes a clip.
cfg_w = h32env.detector_config(base, camera="west", registry_path=REG, env=ENV)
check("an rtsp camera is recorded direct, skipping the go2rtc hop",
      cfg_w["rtsp_camera_direct"].startswith("rtsp://"))
check("an onvif camera is NOT handed to ffmpeg — it records via the restream",
      cfg_s["rtsp_camera_direct"] is None, str(cfg_s["rtsp_camera_direct"]))
check("an unknown camera raises rather than silently running as another",
      raises(lambda: h32env.detector_config(base, camera="ghost", registry_path=REG, env=ENV)))

print("\n8. the legacy single-camera API still works (talk.py, capture/)")
check("detector_config with no camera still works",
      h32env.detector_config(base)["imgsz"] == 1280)
check("rtsp() still builds a Victure URL", h32env.rtsp(0).startswith("rtsp://"))
check("CAMERA_IP is still exported", isinstance(h32env.CAMERA_IP, str))

print("\n9. the REAL registry in the repo")
real = h32env.cameras()
check("web/cameras.json parses and lists cameras", bool(real))
ids = [c.id for c in real]
ports = [c.monitor_port for c in real]
check("camera ids are unique", len(set(ids)) == len(ids), str(ids))
check("no two cameras share a monitor port", len(set(ports)) == len(ports), str(ports))
check("no camera id collides with a go2rtc sub-stream name",
      not (set(ids) & {f"{i}_sub" for i in ids}), str(ids))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print(f"all camera-registry checks passed ({len(ids)} cameras in web/cameras.json, "
      f"{len([c for c in real if c.configured])} configured)")
