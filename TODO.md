# h32 — roadmap

Short-lived notes live in commit messages; this file is for threads that span sessions.

## 1. Multi-camera — DONE for 2 cameras, 3rd awaiting the gate mount (2026-08-17)

Goal: every camera on one screen, the same detector/AI running across all of them.
**The gating unknown is answered: the VIMTAGs do expose a local stream.** Built and
running today with `west` (Victure) + `south` (VIMTAG); `gate` is a blank line in
local.env until the second VIMTAG is mounted.

### What the VIMTAG actually is (measured 2026-08-17, not taken from the listing)

| | |
|---|---|
| Local stream | ✅ full ONVIF 2.4 — Device + Media + **PTZ** services on `:80` |
| Main | 2560×1440 **HEVC** 20 fps + AAC |
| Sub | 640×360 HEVC 20 fps |
| RTSP path | `/live/<devid>_p0_<TOKEN>` — **rotates on every resolution AND is single-use** |
| Detector decode | 1.02 ms/frame (981 fps on one thread) — 2% of a core |
| Browser | plays natively; `web/video-rtc.js:605` already scores `hvc1.` above H.264 |

⚠️ **`onvif://` is mandatory, not a convenience.** A token resolved by go2rtc and handed
to ffmpeg seconds later is rejected by the camera. There is no URL that can be written
down, so anything that wants this camera's video must ask ONVIF for a fresh one at
connect time — which `onvif://` does per producer start. Do not "optimise" this into a
cached RTSP URL.

⚠️ **Transcoding the 2.5K HEVC through go2rtc fails** — `ffmpeg:south#video=h264` dies on
repeated `Could not find ref with POC` / `Error constructing the frame RPS`, because
go2rtc's transcode chain re-serves the stream and the child ffmpeg cannot rebuild the
reference set. It is not needed (the player does HEVC over MSE), and the 640×360 sub
*does* transcode cleanly if it is ever wanted. Don't rediscover this.

⚠️ **The app password is NOT the ONVIF password.** Setting the phone-app account to
`egg123` left ONVIF answering only to `admin`/`admin` (egg123 gets ONVIF 400). The
credential that streams the video is a separate "ONVIF / third-party access" user.
**Both VIMTAGs are still on the factory ONVIF password.**

### Codec: switch the VIMTAGs to H.264 in the app

They are on H.265. It works, but `web/video-rtc.js` scores WebRTC+H265 0x240 >
MSE+H265 0x230 > WebRTC+H264 0x220, and no desktop browser does H.265 over WebRTC — so
H.265 lands on **MSE**, costing ~1s of latency versus WebRTC's sub-second. H.264 also
makes all three cameras one code path and keeps the transcode fallback available. The
bitrate cost is irrelevant on a link measured at 0% loss.

### ⚠️ The south VIMTAG fell off the network under two concurrent streams (2026-08-17)

After hours of stable single-stream probing, `south` went to **100% ping loss with an
incomplete ARP entry** — off the LAN entirely, not merely stalled — and did not come back
on its own. At the time h32 was asking it for **two** concurrent RTSP sessions: `south`
(detector + recorder) and `south_sub` (the wall tile). go2rtc opens one connection per
stream NAME, so the sub tile is a genuine second session.

Mitigation shipped: `tile_stream: "main"` per camera in `web/cameras.json`, which puts the
tile, the detector and the recorder on ONE producer, so the camera sees one session.
**This is a hypothesis, not a proven cause** — it is consistent with the evidence and the
change is free, but a camera that drops out on ONE stream would disprove it. The link
watch below is what will settle it, because it now records every dropout.

### ⚠️ The real constraint is the 2.4 GHz cell, not the cameras (2026-08-17, evening)

Measured while the west camera's video was streaming perfectly:

| target | latency | loss |
|---|---|---|
| gateway `192.168.1.1` | **10 ms** | 0% |
| west camera `.216` | **1877–4320 ms** | 0% eventual |
| south VIMTAG `.124` | unreachable | 100% |
| gate VIMTAG `.128` (freshly paired) | unreachable | 100% |

The Mac's own link is healthy (−63 dBm, SNR 29 dB, 115 Mbit/s, MCS 13) and the gateway
answers in 10 ms, so **the AP and this Mac are fine** — it is the path to the cameras
that is collapsing. West's video kept flowing throughout, because TCP retries and go2rtc
buffers where ICMP simply dies; "the picture is fine" is therefore NOT evidence that the
link is fine, which is exactly why the link watch is worth having.

Airtime, not bandwidth, is the scarce resource. Scan of the band: **12 networks on 2.4 GHz,
4 of them sharing our channel 7, with neighbours at −56 dBm — stronger than our own AP at
−63 dBm.** Channel 11 was empty in the same scan. A client on a weak link transmits at a
low MCS and so occupies far more airtime per byte (the classic 802.11 performance
anomaly), which means **a marginal camera does not just suffer, it degrades the whole
cell for everything else.** Both VIMTAGs being unreachable while the Victure merely
crawls fits that exactly.

⚠️ **This supersedes the earlier "two concurrent RTSP sessions killed the south camera"
theory, and probably the "the unit is faulty" one too.** Neither is disproven, but a
congested, weak 2.4 GHz cell explains all of it — the south camera dying, its refusal to
come back, and a brand-new second camera being unreachable from the moment it paired —
without needing two separate hardware faults.

**Decisive test, not yet run:** power off BOTH VIMTAGs and re-measure west. If west's
round trip returns to ~10 ms, the VIMTAGs are crushing the cell and the fix is radio, not
software.

Fixes, by expected value:
1. **Move the AP to 2.4 GHz channel 11** (non-overlapping, empty in the scan; ours shares 7
   with three others, one of them stronger than us). Free, two minutes.
2. **Drop the VIMTAGs from 2.5K to 1080p in the app.** A marginal link plus 2.5K is the
   worst combination available; fewer bits is less airtime for everyone. Note the detector
   gains nothing from 2.5K anyway — it resizes every frame to 1280.
3. **Get the cameras closer to an AP**, or put one near the garden. For the permanent
   mount this is the real answer: cameras on a weak radio are a standing tax on the whole
   house network.
4. 5 GHz is empty here but penetrates walls badly — plausible for a camera near the AP,
   not for the far end of the garden.

### Link health — there is no WiFi signal strength to read

The VIMTAG advertises `Dot11Configuration=false` and answers `GetDot11Status` with
nothing, so **RSSI is not available from these cameras** and a signal-bars icon would be
invented. `detector/link.py` measures the honest substitute instead — ping RTT, rolling
packet loss and dropout count — and the monitor shows it per tile. It also separates the
two faults that look identical on screen: *off the network* (radio/power) versus
*reachable but no video* (stream/session). One lost ping is deliberately not an outage,
and a long outage counts as one dropout rather than one per packet.

### Shape of the thing that got built

- **`web/cameras.json`** — the registry: id, name, monitor port, tile stream, zone space.
  No secrets. Served by go2rtc for free (it already serves `web/`), read off disk by the
  detector. Camera URLs live in `local.env` as `H32_CAM_<ID>` / `_SUB`, **quoted** — an
  unquoted `&` in an RTSP query string is a shell parse error that stops the whole app.
- **A blank URL means the camera does not exist** — no tile, no detector, no events. That
  is what makes 1, 2 and 3 cameras the same code path rather than three layouts; all four
  cases (0/1/2/3) are verified by rendering the page headless.
- **One detector process per camera**, `detect.py --camera <id>`, deriving events dir,
  buffer, learned scenery and monitor port from the id. Measured: MegaDetector is 84 ms/
  frame on this M3 Pro's MPS, so 3 cameras × 3 fps = 9 of ~15 available fps.
- **`h32 probe <ip>`** answers the setup-day question in one command; `h32 detect`
  supervises one detector per configured camera and Ctrl-C stops them all.

### Still to do

- Mount `gate`, then add its two lines to local.env. Nothing else should be needed.
- **Zones + rules** — schema is in `cameras.json` (`zones`, `rules`, `zone_space`) and
  ships empty, so every camera behaves exactly as before until a polygon is drawn. The
  polygons cannot be drawn until the cameras are aimed. Target rules: anyone at the gate,
  and the 2-year-old at the gate. ⚠️ Zone coordinates are in each camera's OWN frame size
  — 2560×1440 for the VIMTAGs, 1920×1080 for the Victure.
- A "draw the zone on the focused tile" tool, once the cameras are aimed. Until then the
  polygons are hand-written, the same convention `roi`/`exclude_roi` already use.
- **Zero cloud comms** (John, 2026-08-17): block each camera's WAN access at the router so
  nothing reaches the vendor cloud. Needs a static DHCP lease per camera first, then a
  firewall rule, then re-verify ONVIF/RTSP still work locally (they should — h32 only ever
  talks to the LAN). Watch for the camera's clock drifting once NTP is blocked, since the
  OSD timestamp is burned into every recorded frame.
- Cross-camera reasoning (someone at the gate, then on the south lawn) is impossible with
  one process per camera. That was a deliberate trade for isolation; revisit only if a
  rule actually needs it.


### Hardware, and why `imgsz` cannot be traded for it (measured 2026-08-17)

The whole hardware bill is **one model**. Measured per frame on this Mac's CPU at 4 threads
(the core count a Pi has): MegaDetector at 1280 **442 ms**, SpeciesNet on a crop 152 ms,
CLAHE on 1080p 2.7 ms, JPEG encode 2.8 ms, and decoding this camera's stream **0.69 ms/frame**
(it is 1080p20 at only 0.6 Mbit/s — measured off `buffer/*.ts`). Everything except detection
is rounding error, so the sizing question is only ever "how many MegaDetector frames/second".

**No Pi runs that on its CPU.** yolov9-c is 102 GFLOPs at 640, so ~408 at our 1280; a Pi 5
does ~50–70 GFLOPS effective, i.e. **~6–8 s per frame** against the 333 ms that 3 fps needs —
about 5× the whole machine for *one* camera. An accelerator is mandatory, not an optimisation.

So the obvious move is to drop `imgsz` and buy less hardware. **It does not work**, and the
reason is not the one you would guess. All 68 recorded clips were replayed at 1280/960/640
(same CLAHE, same thresholds, same `min_hits`/`window`), then fed through `SceneryFilter`
exactly as `test_scenery.py` does, with one filter walking the whole archive in time order —
the closest model of a detector that has been watching. Reproduce with
**`detector/imgsz_sweep.py run`** then **`report`** (~30 min for the first, seconds after;
the cache is gitignored). Point it at a different model to re-run this whole comparison —
which is exactly what you want after quantising for a Hailo:

| imgsz | live clips that fire (of 41) | **furniture clips that fire (of 26)** |
|---|---|---|
| 1280 | 38 | **1** |
| 960  | 36 | **3** |
| 640  | 35 | **10** |

Missed animals are the *small* half of the cost. The large half is that **a lower `imgsz`
invents people**: 2 340 detections at 960 and **4 569 at 640** that 1280 never made, 75% of
them on spots already known to be furniture, and overwhelmingly classed `person`
(4 274 of the 640 ones). Downscaling turns the bucket, the trough and the plant pot back into
people — the exact failure the scenery filter exists to fight, arriving faster than it can learn.

- **Loss is confined to small and distant objects**, which is precisely the pond case. Recall
  of 1280's own detections, by box size: <150px **64.5% / 59.7%** (960/640), 150–300px
  85.0% / 71.9%, 300–600px 99.1% / 98.8%. Anything close to the camera is unaffected.
- **The 03:53:07 raccoon dies at 640.** Same animal, same box, confidence **0.75 → 0.36 → 0.14**
  — under the `animal: 0.20` floor. The 21:18 cat goes 28 frames → 20 → **0**.
- ⚠️ **A lower `conf` cannot buy it back, and this is the finding that closes the question.**
  1280 at 0.20 gives 100% recall with 2 216 furniture detections. 640 at 0.10 — the most
  permissive setting tested — reaches only **88.8%** recall with **5 342** furniture detections.
  There is no threshold at which a smaller `imgsz` matches 1280 on *either* axis; it is
  dominated, not traded off. Don't re-derive this.
- ⚠️ **Nor is lowering per-camera `fps` free**, for a reason specific to this system: an event
  needs `min_hits=2` within a `window` of 5 *frames* (`detect.py:516`), and the 03:53 raccoon
  appeared in 2 frames of a 37-second clip. Halving the sample rate halves the chances of the
  only two it gets.
- **What the wobble does**, since the scenery filter's whole separation rests on it (rock 0.064
  vs raccoon 0.062): on the main trough spot, median wobble rises 0.016 → 0.021 → 0.024, but
  the **max goes 0.124 → 0.090 → 0.353**. At 640 a static spot's box can swing 5.7× the
  distance the raccoon actually moved, and new fixed spots appear that 1280 never detects at all.
  Every learned threshold would need re-deriving against a noisier detector.

**Conclusion: 1280 stays, and the hardware has to meet it.** That fixes the requirement at
3 fps × 1280 per camera. Hailo's model zoo puts yolov9c at 640 at 68.2 FPS on a Hailo-8, so
~17 FPS at 1280 (÷4 for pixels) — **5 cameras of theoretical NPU capacity, 4 with headroom**;
a Hailo-8L is half that, so 2. Buy the 26 TOPS AI HAT+, not the 13. Two further practicalities:
the **Hailo compiler does not run on ARM or macOS** (an x86_64 Linux box or cloud VM is needed
for the .pt→.onnx→.har→.hef conversion), and **every published Hailo YOLO benchmark is at 640**
— 1280 is unproven there, so treat the ÷4 as an upper bound and verify before buying five
cameras' worth. The int8 quantisation that conversion implies lands squarely in the 0.06-sized
gap the scenery filter measures in, so budget for re-deriving `jitter_slack`/`min_move` on Pi
footage. ⚠️ Also: put the circular buffer in **tmpfs** — 120 s at 0.6 Mbit/s is only ~9 MB per
camera, and writing 2-second segments to an SD card continuously will destroy it.

## 0. Cat/human/raccoon + learning the 4 household people (in progress 2026-08-16)

John: *"cat, human and raccoon detector. also … record the different humans so we can
learn them"* — 4 people (male + female adult, 6yo, 2yo boy). Design = a shared **harvester**
feeding two learners.

- ✅ **Harvester** (`gallery.py`): detector saves face + animal crops (gitignored, deduped,
  bounded) with SFace embeddings on faces. Running now.
- ✅ **Species** (`speciesnet.py`, 2026-08-16): Google's SpeciesNet on the crop — names the
  species AND overrules MegaDetector's bad `person` calls. Replaced the CLIP matcher in
  `species.py`, which was **measured to key on lighting, not species** (empty night
  pavement scored `raccoon 0.919`, above the real cat; a night human scored 0.843). Its
  "100% leave-one-out" was measuring night-vs-day. Weights: `get-speciesnet.sh`.
  - ⚠️ **A "day model + night model" split is NOT the answer** and was considered and
    rejected on evidence: SpeciesNet is robust across both (night human 0.975, daylight
    human 0.999, empty night pavement `blank` 0.983). The old classifier was *accidentally*
    a day/night model, which is precisely what was wrong with it. Don't rebuild that.
  - ⚠️ **Eyeshine is not the give-away it looks like.** Tempting, and the biology is right
    (cats + raccoons have a tapetum lucidum, humans do not — so it is an *animal-vs-human*
    cue, not a cat-vs-raccoon one). But it is high-precision / low-recall: scanned all 380
    frames of the 21:18 cat clip for bright paired blobs in the box and found **zero** —
    the cat faced away throughout. Present ⇒ almost certainly an animal; absent ⇒ proves
    nothing. Worth adding only as a cheap extra confirmation, never as the test.
  - MegaDetector itself is unfixable here by tuning: on that cat it said person in 23 of 29
    sampled frames, and CLAHE-off / imgsz 1920 / test-time augmentation were all measured
    to be no better (1920 loses the animal entirely in 7/10 frames). The second opinion on
    the crop is the fix, not a detector knob.
- ⚠️ **Night IR washes faces to pure white up close** (seen 2026-08-16 20:51 — John's face/shirt blown out by the IR illuminator). That destroys the detail SFace needs, so night-up-close recognition is unreliable; **daylight faces are the good case**. The cluster-and-label tool must **filter over-exposed crops** (e.g. skip a face crop whose mean luma is near-saturated) so blown-out night faces don't pollute the enrolments.
- ⏭ **Cluster-and-label tool (NOT built yet):** once face crops accumulate in the gallery,
  a tool that clusters the SFace embeddings (sklearn agglomerative/DBSCAN on cosine),
  shows a contact sheet per cluster, and lets John name them → enrols Dad/Mum/6yo/2yo into
  `faces_store.npz`. This is how the kids get learned (one-shot enrol of a 2yo won't stick).
  ⚠️ **The 2yo from a high night-IR fisheye is at/past the edge of face recognition** — may
  only ever resolve to "a small child". **Person-box height** is a free coarse size prior
  (adult vs 6yo vs 2yo) for when no face is visible — worth adding to the harvest metadata.
- ⚠️ **SpeciesNet cannot name the hedgehog — measured 2026-08-17.** It reads the 03:42
  visitor (`20260817_034249_animal.mp4`) as `blank` 0.51–0.96 while
  `western european hedgehog` scores **0.00004–0.00012**. Tried and did not help: tight /
  +50% / +150% / +400% crop padding, raw / CLAHE / gamma 0.5 / histogram-equalised /
  4× upscaled crops, and the full frame. Its top guesses are all New World species
  (virginia opossum, central american agouti, white-lipped peccary) — the head does not
  cover a night-IR European hedgehog on this camera, and **no threshold reaches it**.
  So the harvested crops ARE needed to fix the classifier after all, for the species it
  misses outright — this reverses the note that used to sit here.
- ✅ **Local reference matching (BUILT 2026-08-17, live):** `animal_match.py`, wired into
  `verify_species` and on by default (`species.local_refs`). When SpeciesNet names
  nothing, the crop's feature is matched against `animal_refs.npz`; `HEDGEHOG` is in
  `email.trigger_on`. Threshold 0.60 is measured — same animal on a DIFFERENT night runs
  cosine mean 0.518, different things max out at 0.552, so 0.60 clears the whole observed
  negative range; per-frame that names ~50% of cross-visit crops with nothing wrong (0.55
  gets 75% but calls a PERSON a raccoon; 0.70 names nothing). **Expect about half of
  visits to be named, not all.** Verified end to end: the 03:42 clip tags HEDGEHOG, the
  trough clips stay a generic ANIMAL (0/38 and 0/10 crops named), and no person crop is
  named as an animal. The negatives do that work, not the threshold — furniture/empty/
  person are references too, and a crop nearest to one of those is never named.
  - ⚠️ **STILL only one hedgehog visit.** Every number above comes from cat/raccoon/person
    (two visits each); the hedgehog's own cross-visit behaviour is UNMEASURED. Re-measure
    after the next visit — `h32 harvest && h32 label`, then re-run the threshold sweep.
  - ⚠️ Cluster 11 of the first labelling (`20260817_035833`, box clipped at x2=1919) was
    left unlabelled: it could be an animal or a shadow. Decide it when there is a better view.
- ⏭ **Superseded note — the pre-matcher plan (harvest + label, BUILT 2026-08-17):**
  `speciesnet.py` now returns the 1280-d pooled feature off the same forward pass (free);
  `harvest_refs.py` mines it out of the saved clips; `label_animals.py` clusters and names
  them into `animal_refs.npz`. Measured on the first 37 crops: **12/13 hedgehog crops have
  another hedgehog as nearest neighbour**, and at `cos>=0.55` they form one pure cluster of
  12 — so retrieval works even though the classifier head does not. Still to do:
  - the matcher itself (threshold, margin, cross-frame voting like `faces.py`), then
    `HEDGEHOG` into `email.trigger_on`;
  - ⚠️ **only ONE hedgehog event exists so far**, so cross-event generalisation is
    UNTESTED — the whole risk is that it keys on the paving, not the animal;
  - the acceptance test that killed `species.py`: empty pavement, the bench, the trough
    and the plant pot must NOT match, and leave-one-out must be run night-vs-night only.
    The one bad neighbour already seen was an elongated partial crop matching the empty
    pavement of `20260817_040814` at cos 0.933 — exactly that failure mode. Label such
    crops `d` (drop) so they become negatives rather than references.
- ⏭ **Individual-cat ID** (our black cat vs the neighbours') stays an embedding job too,
  and now shares the same harvest/label pipeline.
- ⚠️ Everything harvested is real people's biometrics incl. children → gitignored, local only.

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
2. **34567 DVRIP is a red herring; the app speaks `cc dd ee ff`. Capture harness built,
   awaiting one talk session (spiked 2026-08-16).**

   Two probes this session. First, active DVRIP against 34567: an unauthenticated
   **KeepAlive answered `Ret:100`** but the **LOGIN (msgid 1000) is silently ignored**
   (every variant), so `OPTalkClaim` is unreachable. ⚠️ **Hammering 34567 rebooted the
   camera** (~65s, ping stayed up = app-watchdog restart, recovered clean) — if ever
   re-probed: one connection at a time, ≥2s apart, watch 554 between.

   Then reading `ptz_capture.pcap` collapsed the DVRIP angle: **the app<->camera link is
   the proprietary `cc dd ee ff` protocol** (device const `e4 12 69 00`), not DVRIP.
   34567's DVRIP is a vestigial stub the app never uses. In a whole PTZ session the phone
   sent the camera only **30 keepalive frames (type `0x01`)** — the pan went via cloud.

   So the real experiment is a **talk capture**: `capture/capture_talk.py` (built; its
   parser is validated against the PTZ pcap). It spoofs the phone against camera **and**
   gateway and watches for a **new `cc dd ee ff` message type phone->camera** while talk
   is held — a new high-volume type = LOCAL (buildable: a client sends the same frames to
   `:23456`); a flood to a cloud IP = cloud-brokered (parked, same as PTZ).

   ⚠️ **Needs John:** `sudo ./.venv/bin/python capture/capture_talk.py`, then in IPC360
   press & hold talk saying "hello" ~5s, ×2–3, Ctrl-C. Capture is passive (route+sniff),
   does NOT stress the camera like the active probing did. **Prior probability leans
   cloud-brokered** (talk is the same protocol family as the cloud-brokered PTZ) — but
   talk is a *stream*, not a one-shot command, so there is a real chance it is streamed
   locally. The capture settles it either way.

The transparent MITM in `capture/` remains parked on purpose.
