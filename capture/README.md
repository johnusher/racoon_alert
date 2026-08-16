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
./.venv/bin/python capture/parse_ptz.py                     # analyse ptz_capture.pcap
./.venv/bin/python capture/parse_ptz2.py                    # …all ports, every cc-dd-ee-ff frame
./.venv/bin/python capture/parse_cloud.py                   # analyse cloud_capture.pcap
```

`ptz_session.py`, `ptz_try.py`, `ptz_try2.py` replay candidate PTZ packets straight at the
camera and frame-diff the stream to see whether the lens actually moved. All ~15 variants
produced zero movement — that is the evidence for the cloud-brokered conclusion.

**The `.pcap` files these produce are gitignored, and should stay that way** — raw capture
of camera traffic can contain device identifiers and session tokens.
