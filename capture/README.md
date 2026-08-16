# `capture/` — PTZ / cloud reverse-engineering tools

Tooling used to work out how this camera's pan-tilt-zoom actually works. The conclusion is
in the main [README](../README.md): on firmware V3.15.73 PTZ is **cloud-brokered**, local
replay does nothing, so PTZ and two-way talk are parked. These scripts are kept so the
finding doesn't have to be re-derived.

> **Use on your own camera, on your own network.** `capture_ptz.py` and `capture_cloud.py`
> work by ARP-spoofing so traffic between two devices routes through this Mac. That is fine
> for inspecting hardware you own on a network you control, and nothing else. They restore
> the ARP tables on exit.

Settings (camera IP, gateway, phone, interface) come from `local.env` at the repo root —
see [`local.env.example`](../local.env.example). Run everything from the repo root:

```
sudo ./.venv/bin/python capture/capture_ptz.py <PHONE_IP>   # phone <-> camera, ports 23456/34567
sudo ./.venv/bin/python capture/capture_cloud.py            # camera <-> vendor cloud
sudo ./.venv/bin/python capture/capture_talk.py [PHONE_IP]  # two-way TALK: local or cloud?
./.venv/bin/python capture/parse_ptz.py                     # analyse ptz_capture.pcap
./.venv/bin/python capture/parse_ptz2.py                    # …all ports, every cc-dd-ee-ff frame
./.venv/bin/python capture/parse_cloud.py                   # analyse cloud_capture.pcap
./.venv/bin/python capture/parse_talk.py [file.pcap]        # analyse talk_capture.pcap
```

`ptz_session.py`, `ptz_try.py`, `ptz_try2.py` replay candidate PTZ packets straight at the
camera and frame-diff the stream to see whether the lens actually moved. All ~15 variants
produced zero movement — that is the evidence for the cloud-brokered conclusion.

## Two-way talk (`capture_talk.py` / `parse_talk.py`)

Answers one question: when you press TALK in the app, does the audio go to the camera
**locally** (buildable) or to the **vendor cloud** (parked, same wall as PTZ)?

The app<->camera link is the proprietary **`cc dd ee ff`** protocol (device const
`e4 12 69 00`), NOT plaintext DVRIP — confirmed from `ptz_capture.pcap`, where during a
whole PTZ session the phone sent the camera only **30 keepalive frames (type `0x01`)** and
the actual pan went via cloud. So `capture_talk.py` spoofs the phone against **both** the
camera and the gateway (to see LAN and internet at once) and watches for a **new
`cc dd ee ff` message type** phone->camera while talk is held — a new high-volume type is
local audio; a flood to a cloud IP instead means cloud-brokered.

To run it: `sudo ./.venv/bin/python capture/capture_talk.py`, then in IPC360 open the
camera, **press & hold talk and say "hello" for ~5 s, release, repeat 2–3×**, Ctrl-C.
It prints a verdict and saves `talk_capture.pcap`; `parse_talk.py` re-analyses it offline.
Capture is passive (route + sniff) — unlike active port-probing it does not stress the
camera.

**The `.pcap` files these produce are gitignored, and should stay that way** — raw capture
of camera traffic can contain device identifiers and session tokens.
