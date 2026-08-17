#!/usr/bin/env python3
"""
Site-local settings for h32 — the camera list, credentials, LAN layout, alert e-mail.

None of it is in the repo. Values come from `local.env` at the repo root (gitignored;
copy `local.env.example` and fill it in). Real environment variables win over the file,
which is how the `h32` launcher hands the same values to go2rtc: it exports them, and
go2rtc expands the `${H32_*}` placeholders in `go2rtc.yaml`.

The file is read by path rather than inherited from the environment on purpose — the
capture tools run under `sudo`, which strips the environment.

There are two halves:

  * The CAMERA REGISTRY (`cameras()`), which pairs `web/cameras.json` — who exists, what
    they are called, where their detector listens — with the URLs in `local.env`. A camera
    with no URL is simply not configured, and drops out of everything: no tile, no
    detector, no events. That is what makes one, two and three cameras the same code path,
    rather than three cases to maintain.
  * The LEGACY SINGLE-CAMERA vars (`CAMERA_IP`, `rtsp()`, …), still used by talk.py and the
    capture/ reverse-engineering tools, which are specific to the Victure.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(ROOT, "web", "cameras.json")

# Defaults are deliberately generic — a fresh clone starts with nobody's real network.
DEFAULTS = {
    "H32_CAMERA_IP":    "192.168.1.100",
    "H32_CAMERA_USER":  "admin",
    "H32_CAMERA_PASS":  "",
    "H32_GATEWAY_IP":   "192.168.1.1",
    "H32_PHONE_IP":     "",
    "H32_IFACE":        "en0",
    "H32_EMAIL_TO":     "",
    "H32_CAMERA_DEVID": "",           # per-camera id for two-way talk (see local.env)
    "H32_CAMERA_CONST": "e4126900",   # platform constant for the talk protocol
}


def _load():
    """local.env, overlaid with any real H32_* environment variables.

    Every H32_* key in the file is kept, not just the known ones — the per-camera
    H32_CAM_<ID> URLs are named after whatever is in cameras.json, so they cannot be
    listed here in advance.
    """
    vals = dict(DEFAULTS)
    path = os.path.join(ROOT, "local.env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    vals.update({k: v for k, v in os.environ.items() if k.startswith("H32_") and v})
    return vals


_env = _load()

CAMERA_IP    = _env["H32_CAMERA_IP"]
CAMERA_USER  = _env["H32_CAMERA_USER"]
CAMERA_PASS  = _env["H32_CAMERA_PASS"]
GATEWAY_IP   = _env["H32_GATEWAY_IP"]
PHONE_IP     = _env["H32_PHONE_IP"]
IFACE        = _env["H32_IFACE"]
EMAIL_TO     = _env["H32_EMAIL_TO"]
CAMERA_DEVID = _env["H32_CAMERA_DEVID"]
CAMERA_CONST = _env["H32_CAMERA_CONST"]

GO2RTC_API  = "http://127.0.0.1:1984"
GO2RTC_RTSP = "rtsp://127.0.0.1:8554"


def rtsp(stream=0):
    """Direct Victure RTSP URL. stream 0 = 1080p main, 1 = 640x360 sub.

    Legacy: only the Victure has a guessable RTSP path. The Vimtags hand out a
    single-use token per connection, so their URL can only come from ONVIF — which is
    why the registry stores whole source URLs instead of an address and a path.
    """
    return (f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
            f"/realmonitor?channel=0&stream={stream}.sdp")


# ----------------------------------------------------------------- registry ---

class Camera:
    """One camera: its identity from cameras.json, its URLs from local.env.

    Every per-camera path is derived from the id, so adding a camera cannot collide
    with an existing one. Scenery in particular is learned furniture and belongs to a
    PLACE, so it is never shared between cameras; faces and animal references are about
    the household and deliberately are.
    """

    def __init__(self, spec, env):
        self.id = spec["id"]
        self.name = spec.get("name", self.id)
        self.short = spec.get("short", self.name)
        self.kit = spec.get("kit", "")
        self.monitor_port = spec.get("monitor_port", 8090)
        self.zone_space = spec.get("zone_space", [1920, 1080])
        self.detect = spec.get("detect", {}) or {}
        self.zones = spec.get("zones", []) or []
        self.rules = spec.get("rules", []) or []

        key = f"H32_CAM_{self.id.upper()}"
        self.url = (env.get(key) or "").strip()
        # A camera with only one stream still works; the tile just shows the main one.
        self.url_sub = (env.get(key + "_SUB") or "").strip() or self.url

    @property
    def configured(self):
        """Is there actually a camera behind this entry? Blank URL = still boxed."""
        return bool(self.url) and not self.url.startswith("${")

    @property
    def host(self):
        """The camera's address on the LAN, dug out of whatever scheme its URL uses.

        Used to watch the link to it. These cameras report no WiFi signal strength of
        their own — the 2.5K Vimtag answers GetDot11Status with nothing and advertises
        Dot11Configuration=false — so reachability measured from here is the only honest
        link metric available.
        """
        if not self.configured:
            return ""
        rest = self.url.split("://", 1)[-1]
        rest = rest.rsplit("@", 1)[-1]          # strip user:pass@
        hostport = rest.split("/", 1)[0].split("?", 1)[0]
        if hostport.startswith("["):            # IPv6 literal
            return hostport.split("]", 1)[0].lstrip("[")
        return hostport.split(":", 1)[0]

    # go2rtc stream names are the camera id, so every URL in the app is predictable.
    @property
    def stream(self):
        return self.id

    @property
    def stream_sub(self):
        return f"{self.id}_sub"

    @property
    def events_dir(self):
        return os.path.join(ROOT, "detector", "events", self.id)

    @property
    def scenery_path(self):
        return os.path.join(ROOT, "detector", f"scenery.{self.id}.json")

    @property
    def buffer_dir(self):
        return os.path.join(ROOT, "detector", "buffer", self.id)

    @property
    def frame_url(self):
        return f"{GO2RTC_API}/api/frame.jpeg?src={self.id}"

    @property
    def rtsp_url(self):
        return f"{GO2RTC_RTSP}/{self.id}"

    def public(self):
        """What the monitor page needs. No URLs — those carry passwords."""
        return {"id": self.id, "name": self.name, "short": self.short, "kit": self.kit,
                "monitor_port": self.monitor_port, "zone_space": self.zone_space,
                "zones": self.zones, "rules": self.rules, "configured": self.configured}

    def __repr__(self):
        state = "configured" if self.configured else "absent"
        return f"<Camera {self.id} {state} :{self.monitor_port}>"


def cameras(registry_path=None, env=None):
    """Every camera in the registry, in file order, configured or not."""
    path = registry_path or REGISTRY
    env = _env if env is None else env
    if not os.path.exists(path):
        return []
    spec = json.load(open(path))
    return [Camera(c, env) for c in spec.get("cameras", [])]


def configured_cameras(registry_path=None, env=None):
    """Only the cameras that actually exist on the network right now."""
    return [c for c in cameras(registry_path, env) if c.configured]


def _by_id(camera_id, registry_path=None, env=None):
    for c in cameras(registry_path, env):
        if c.id == camera_id:
            return c
    return None


def camera(camera_id, registry_path=None, env=None):
    """One camera by id, or None."""
    return _by_id(camera_id, registry_path, env)


def _merge(base, over):
    """Recursive dict merge — an override must not clobber its siblings.

    `{"conf": {"person": 0.25}}` over `{"conf": {"animal": 0.20, "person": 0.30}}`
    has to leave `animal` alone, or every camera would need to restate every default.
    """
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def detector_config(path=None, camera=None, registry_path=None, env=None):
    """detector/config.json with the site-local blanks filled in from local.env.

    With a camera id, the camera's own `detect` overrides are merged over the defaults
    and its streams and paths are filled in — so config.json keeps the documented
    rationale for every threshold in one place and a camera states only what differs.
    """
    path = path or os.path.join(ROOT, "detector", "config.json")
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg["rtsp_camera_direct"] = cfg.get("rtsp_camera_direct") or rtsp(0)
    email = cfg.setdefault("email", {})
    email["to"] = email.get("to") or EMAIL_TO

    if camera:
        cam = _by_id(camera, registry_path, env)
        if cam is None:
            raise KeyError(f"no camera {camera!r} in {registry_path or REGISTRY}")
        cfg = _merge(cfg, cam.detect)
        # The recorder hands this URL straight to ffmpeg, so it is only usable when the
        # camera speaks something ffmpeg can open. `onvif://` is a go2rtc-only scheme —
        # for those cameras the recorder has to go through the go2rtc restream instead,
        # which is what `rtsp_main` is. Getting this wrong yields a recorder that starts
        # cleanly and silently never writes a clip.
        direct = cam.url if cam.url.startswith(("rtsp://", "rtmp://", "http://", "https://")) else None
        cfg.update({
            "camera": cam.id,
            "camera_name": cam.name,
            "camera_host": cam.host,
            "frame_url": cam.frame_url,
            "rtsp_main": cam.rtsp_url,
            "rtsp_camera_direct": direct,
            "monitor_port": cam.monitor_port,
            "events_dir": cam.events_dir,
            "scenery_path": cam.scenery_path,
            "buffer_dir": cam.buffer_dir,
            "zone_space": cam.zone_space,
            "zones": cam.zones,
            "rules": cam.rules,
        })
    return cfg


if __name__ == "__main__":
    found = os.path.exists(os.path.join(ROOT, "local.env"))
    print(f"local.env: {'found' if found else 'MISSING — cp local.env.example local.env'}")
    print(f"  gateway  {GATEWAY_IP}   phone {PHONE_IP or '(unset)'}   iface {IFACE}")
    print(f"  email to {EMAIL_TO or '(unset)'}")
    print()
    all_cams = cameras()
    if not all_cams:
        print("  no cameras — web/cameras.json is missing or empty")
    for c in all_cams:
        if c.configured:
            scheme = c.url.split("://", 1)[0]
            print(f"  ✓ {c.id:<6} {c.name:<26} {scheme:<6} :{c.monitor_port}")
        else:
            print(f"  · {c.id:<6} {c.name:<26} {'—':<6} not configured (H32_CAM_{c.id.upper()})")
    live = [c for c in all_cams if c.configured]
    print(f"\n  {len(live)} of {len(all_cams)} configured")
