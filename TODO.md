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

## 2. Identifying particular people — BUILT, one measurement still missing

Shipped 2026-08-16: `faces.py` + `enroll.py` + `test_faces.py`, wired into `detect.py`.
Events carry a `who=`; the monitor draws the face box and name. See the README for use.

**Still to do — the false-accept measurement.** Everything below was measured against the
one person in the archive, so we know he matches *himself*; we have never checked that
somebody else *fails* to match him. Until that is done, leave `known_suppresses_event`
false: turning it on lets a stranger who happens to match you cancel your own alert.

```
detector/enroll.py add john live --secs 40     # John, in daylight
detector/enroll.py test <clip of someone else> # every line must say UNKNOWN
```

Held-out check that has been done (enrolled on 21.0–23.6s of the 00:32 clip, tested on
23.6–26.5s, which enrolment never saw):

| | score vs enrolled |
|---|---|
| same man, held-out frames | **7/7 identified**, 0.608–0.968 |
| bench false-positive clip | 0.091 → unknown |
| raccoon clip | no faces found at all |

### What the footage supports

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

Consistent with the earlier finding that PTZ and talk are cloud-brokered.

**Requested feature, blocked on the above:** buttons on `h32 detect` (and the plain viewer)
to talk from the MacBook to the camera — either live mic, or one-click playback of a
canned `.wav` (a TTS "hello", a deterrent noise). The `.wav` variant is the easier of the
two, but both need the same thing the PC530 refuses: an inbound audio channel. Build them
the moment a camera actually accepts one — not before, since a button that cannot work is
worse than none.
`web/index.html` already has the pattern: the 🎛 drawer carries disabled PTZ/talk buttons
with a PENDING tag. go2rtc supports two-way audio natively for backchannel-capable
cameras (`media=video,audio,microphone` in the player), so on such a camera this is a
small job — a button that adds `microphone` to the stream's media and a `.wav`/TTS path
via `ffmpeg` into the same backchannel.

**Two untested leads, in order of promise:**

1. **The new VIMTAGs advertise 2-way audio.** If they expose an ONVIF backchannel this
   comes almost free. Test with the same `DESCRIBE` probe as above — a `sendonly` audio
   media in the SDP is the yes/no.
2. **Port 34567 on the PC530 has never been probed for audio.** Its RTSP server
   identifies as `Server: H264DVR 1.0` and 34567 is the classic XiongMai/Sofia "DVRIP"
   port — a documented protocol with open-source clients (e.g. `python-dvr`) that
   implement talk on some XiongMai devices. Our PTZ work only ever targeted 23456. This
   is a real lead but a much bigger job than the backchannel probe, and it may well hit
   the same cloud-brokered wall. Worth a timeboxed spike only if talk on *this* camera
   matters more than talk on the new ones.

The transparent MITM in `capture/` remains parked on purpose.
