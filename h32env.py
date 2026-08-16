#!/usr/bin/env python3
"""
Site-local settings for h32 — camera address, credentials, LAN layout, alert e-mail.

None of it is in the repo. Values come from `local.env` at the repo root (gitignored;
copy `local.env.example` and fill it in). Real environment variables win over the file,
which is how the `h32` launcher hands the same values to go2rtc: it exports them, and
go2rtc expands the `${H32_*}` placeholders in `go2rtc.yaml`.

The file is read by path rather than inherited from the environment on purpose — the
capture tools run under `sudo`, which strips the environment.
"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))

# Defaults are deliberately generic — a fresh clone starts with nobody's real network.
DEFAULTS = {
    "H32_CAMERA_IP":   "192.168.1.100",
    "H32_CAMERA_USER": "admin",
    "H32_CAMERA_PASS": "",
    "H32_GATEWAY_IP":  "***REMOVED-IP***",
    "H32_PHONE_IP":    "",
    "H32_IFACE":       "en0",
    "H32_EMAIL_TO":    "",
}


def _load():
    vals = dict(DEFAULTS)
    path = os.path.join(ROOT, "local.env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    vals.update({k: v for k, v in os.environ.items() if k in DEFAULTS and v})
    return vals


_env = _load()

CAMERA_IP   = _env["H32_CAMERA_IP"]
CAMERA_USER = _env["H32_CAMERA_USER"]
CAMERA_PASS = _env["H32_CAMERA_PASS"]
GATEWAY_IP  = _env["H32_GATEWAY_IP"]
PHONE_IP    = _env["H32_PHONE_IP"]
IFACE       = _env["H32_IFACE"]
EMAIL_TO    = _env["H32_EMAIL_TO"]


def rtsp(stream=0):
    """Direct camera RTSP URL. stream 0 = 1080p main, 1 = 640x360 sub."""
    return (f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554"
            f"/realmonitor?channel=0&stream={stream}.sdp")


def detector_config(path=None):
    """detector/config.json with the site-local blanks filled in from local.env."""
    path = path or os.path.join(ROOT, "detector", "config.json")
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    cfg["rtsp_camera_direct"] = cfg.get("rtsp_camera_direct") or rtsp(0)
    email = cfg.setdefault("email", {})
    email["to"] = email.get("to") or EMAIL_TO
    return cfg


if __name__ == "__main__":
    print(f"local.env: {'found' if os.path.exists(os.path.join(ROOT, 'local.env')) else 'MISSING — cp local.env.example local.env'}")
    print(f"  camera   {CAMERA_USER}:{'*' * len(CAMERA_PASS)}@{CAMERA_IP}")
    print(f"  gateway  {GATEWAY_IP}   phone {PHONE_IP or '(unset)'}   iface {IFACE}")
    print(f"  email to {EMAIL_TO or '(unset)'}")
