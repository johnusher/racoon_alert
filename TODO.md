# h32 — roadmap

Short-lived notes live in commit messages; this file is for threads that span sessions.

## 1. Multi-camera — 2× VIMTAG 2.5K outdoor (bought 2026-08-16, not yet powered on)

Goal: all three cameras on one screen, the same detector/AI running across all of them.

**Everything here is gated on one unknown: do the VIMTAGs expose a local stream?**
The listing (`amazon.de/dp/B0FL7P95Y6`) advertises app/cloud/Alexa and says nothing about
ONVIF or RTSP. VIMTAG models have historically supported both, but budget makers disable
RTSP in some firmware, and this is a 2026 model. If they turn out cloud-only there is no
local video to run AI on, and the options narrow to returning them or an ONVIF-capable
replacement — so **probe before building anything.**

Probe, once they are on the WiFi (~2 minutes, read-only):

```
nmap -p 80,554,8000,8080,8554,2020,34567 <camera-ip>   # what is even listening
# ONVIF device service — the handshake that reveals the RTSP URL
curl -s -X POST http://<ip>/onvif/device_service -H 'Content-Type: application/soap+xml' \
     -d '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>
         <GetCapabilities xmlns="http://www.onvif.org/ver10/device/wsdl"/></s:Body></s:Envelope>'
./.venv/bin/python capture/… or a plain RTSP DESCRIBE against the usual paths
```

Then, only if a local stream exists:

- **go2rtc** is already multi-stream — adding cameras is `streams:` entries, not a rewrite.
- **`web/index.html` / `web/monitor.html`** are single-stream: need a camera picker and/or
  a grid view. The monitor's box overlay is already keyed to a source frame size, so it
  generalises, but `state.json` has to become per-camera.
- **`detector/detect.py` is the real work.** It is a single-camera script built on module-level
  globals (`S`, `LK`, one `CircularRecorder`, one `MonitorServer`, one `SceneryFilter`).
  Three cameras means either three processes (simplest, ~3× the RAM and one model per
  process) or one process with per-camera state objects and a shared model (leaner, needs
  the globals refactored into a class). Decide with measurements: MegaDetector at 1280px on
  MPS is the cost driver, and 3 cameras at 3 fps may not fit — per-camera `fps` may have to drop.
- **`detector/events/`** is a flat directory; filenames would need a camera prefix, and
  `scenery.json` must be per-camera (learned furniture is specific to where a camera points).

## 2. Identifying particular people

Spike pending: measure whether faces in this camera's actual footage are recognisable
before choosing any stack. OpenCV 5.0.0 (already in the venv) ships YuNet face detection
and SFace recognition, so no new dependencies — just two model files from OpenCV's zoo.

Known constraint: every real-person frame in the archive is night IR from a high angle
(body ~840px tall, so a face ~110px — workable on paper, hard in practice). There is no
daylight person footage at all, because every daytime "person" event was the bench.
Walking through the garden once in daylight would give ground truth we know the answer to.

⚠️ If this gets built: enrolled faces and embeddings are personal data about real people
and must be gitignored from the start — this repo is public.

## 3. Two-way audio — closed, not possible locally on the PC530

Probed 2026-08-16 and ruled out; do not re-litigate without new firmware:

- RTSP `DESCRIBE` returns the same SDP with and without `Require: www.onvif.org/ver20/backchannel`
  — video + PCMA mic only, no `sendonly` track.
- The camera ignores the `Require:` header entirely (RFC 2326 says it must answer `551`
  if it does not understand the option), so it does not implement the mechanism at all.
- `Public:` lists no `ANNOUNCE` and no `RECORD` — the two methods needed to push a stream
  *to* a device. `Server: H264DVR 1.0`.

Consistent with the earlier finding that PTZ and talk are cloud-brokered. The only route
left is the transparent MITM in `capture/`, which is parked on purpose.

**The new VIMTAGs advertise 2-way audio** — worth re-testing there, since a camera with a
working ONVIF backchannel would give this for free.
