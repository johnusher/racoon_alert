# h32 — Victure PC530 local camera viewer

A self-contained **local** web app for viewing (and eventually controlling) a Victure
PC530 Wi-Fi camera — no vendor cloud involved. Runs on this MacBook now; designed to
also drive a Raspberry Pi + HDMI screen later.

Type **`h32`** in a terminal → the media server starts and the viewer opens in your browser.

```
h32            start the server (if needed) and open the viewer
h32 stop       stop the media server
h32 restart    restart it
h32 status     is it running?
h32 log        follow the go2rtc log
```

Viewer: <http://127.0.0.1:1984/>

## Setup (fresh clone)

The go2rtc binary is gitignored (not stored in the repo). After cloning:

```
./get-go2rtc.sh        # download the pinned go2rtc binary
python3 -m venv .venv && ./.venv/bin/pip install scapy   # only needed for PTZ capture
```

Then `h32` (add the alias if it isn't already: `alias h32="$PWD/h32"` in `~/.zshrc`).

## Controls

| Action | Mouse | Keyboard |
|---|---|---|
| Enable audio | "Tap for sound" / 🔇 | `m` |
| Digital zoom | ＋ / － / wheel | `+` `-` |
| Digital pan (when zoomed) | PAN arrows | ← ↑ ↓ → |
| Reset view | ⤢ | `0` |
| Snapshot (PNG) | 📷 | `s` |
| Fullscreen | ⛶ | `f` |
| Switch 1080p / 360p | 1080p / 360p | — |
| Camera controls drawer | 🎛 | — |

## What works vs. pending

| Feature | State |
|---|---|
| Live 1080p / 360p video | ✅ working (WebRTC, sub-second; MSE fallback) |
| Listen to camera mic | ✅ working (in the stream) |
| Digital zoom + pan, snapshot, fullscreen | ✅ working (browser-side) |
| **Mechanical PTZ** (pan/tilt the lens) | ⏳ pending — needs a one-time app packet capture |
| **Two-way talk** | ⏳ pending — proprietary audio uplink |

The 🎛 drawer shows the PTZ/talk buttons, disabled, until we capture the command set.

## Camera reference

- **IP:** `***REMOVED-IP***`  (RTSP `:554`, ONVIF `:8080`, proprietary `:23456`/`:34567`)
- **ONVIF/RTSP login:** `admin` / `<vendor default>`
- **Main stream:** `rtsp://***REMOVED-CREDS***@***REMOVED-IP***:554/realmonitor?channel=0&stream=0.sdp` (1080p H.264 + PCM-alaw audio)
- **Sub stream:** `…&stream=1.sdp` (640×360)
- ⚠️ **ONVIF must stay enabled in the IPC360 app** — that's what opens `:554`/`:8080`. If you turn it off, video stops working here.
- Security: stream is unencrypted on the LAN; default password. Fine at home — do **not** port-forward `:554` to the internet.

## How it works

```
camera (RTSP) ──► go2rtc ──► WebRTC/MSE ──► browser (web/index.html)
                    ▲
              go2rtc.yaml
```

- `go2rtc` — single Go binary; ingests the camera RTSP and serves low-latency WebRTC. Also serves the UI (`api.static_dir`).
- `go2rtc.yaml` — stream + server config.
- `web/` — the viewer: `index.html` (UI) + `video-rtc.js` / `video-stream.js` (official go2rtc player, vendored so it's served same-origin).
- `h32` — launcher script (registered as an `alias` in `~/.zshrc`).

## Reverse-engineering notes (so we don't re-derive)

- ONVIF on this camera exposes **video + mic audio only**. It advertises a PTZ service and audio *outputs*, but PTZ `GetNodes`/profiles are empty and audio-output ops fault — verified a `ContinuousMove` produces **zero** frame movement. So neither PTZ nor talk is reachable via ONVIF.
- The camera is the **IPC365 / "360Eyes"** platform (app: **IPC360**). Its PTZ + talk run over a **proprietary TCP protocol on port 23456**.
- PTZ protocol *structure* is known (from `MiguelDLM/360eyes_controller`): 68-byte packets, `pan`=int32 @ offset 40, `tilt` @ 44, `zoom` @ 48; `stop`=zero velocities; no auth/handshake. **But** the fixed header carries a device/session token — that project's bytes moved nothing on our camera (tested both ports, multiple magnitudes).
- ➡️ **Next step to enable PTZ/talk:** capture the IPC360 app moving our camera (ARP-spoof MITM on the LAN via the Mac), read the real 23456 bytes, and replicate them in a small control server that the UI's 🎛 buttons call. Talk is a follow-on (live audio uplink, harder).

## Raspberry Pi (later)

The same stream runs fullscreen on a Pi's HDMI with `mpv`/`ffmpeg` against the RTSP URL
(or point the Pi's browser at this Mac's go2rtc). To be built.

## Files

```
h32              launcher (start/stop/restart/status/log)
go2rtc           media-server binary (darwin arm64, v1.9.14)
go2rtc.yaml      config (camera streams + server)
go2rtc.log       runtime log
web/index.html   viewer UI
web/video-rtc.js, web/video-stream.js   go2rtc player (vendored)
```
