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

## 2. Identifying particular people — feasible, one measurement missing

Spiked 2026-08-16 against the one real person in the archive (the 00:32 clip, night IR).
Stack needs **no new dependencies**: OpenCV 5.0.0 in the venv already ships YuNet face
detection and SFace recognition; only two model files from OpenCV's zoo.

**What the footage supports:**

| | result |
|---|---|
| face height when found | 107–117 px (SFace wants ≥50) |
| YuNet confidence | 0.87–0.90 |
| self-match across frames | **10/10 pairs**, cosine 0.517–0.833 (same-identity threshold 0.363) |
| face visible when a person is detected | **5 of 13 frames ≈ 38%** |

So recognition works *when a face is visible*, even in night IR — but a face is visible
well under half the time. **Identify per visit, not per frame:** collect every face across
the visit, match each, and vote. Unknown by default.

⚠️ **Gate face detection on a person box.** Run whole-frame and it hallucinates: 13 "faces"
in 39 frames of an *empty* garden, every one of them the plastic bucket — two dark marks
read as eyes, the rim as a hairline. Pareidolia, the same failure mode as the bench, and
the scenery filter already makes person boxes trustworthy enough to gate on.

⚠️ **The missing measurement — do not skip this.** The archive holds exactly one identified
human (the other night "person" events were a pair of shoes and the bench). So we measured
that he matches *himself*; we could not measure whether he fails to match *someone else*.
The false-accept rate is unknown, and night IR at a steep angle degrades discrimination in
precisely the way that inflates false accepts. **Before building: one daylight walk-through
by John and one by somebody else.** That yields the daylight sample the archive completely
lacks, a second identity to measure false accepts against, and ground truth for both.

Design note: enrol from camera frames, not a phone selfie — matching a daylight selfie
against night IR is a cross-domain problem and much harder than like-for-like.

⚠️ Enrolled faces and embeddings are personal data about real people and must be gitignored
from the start — this repo is public.

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
