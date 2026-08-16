#!/usr/bin/env python3
"""
Analyse a talk_capture.pcap (or any capture) offline: where did the talk audio go, and —
if it went to the camera locally — what `cc dd ee ff` message type carries it?

    ./.venv/bin/python capture/parse_talk.py [path.pcap]     # default capture/talk_capture.pcap

This firmware's app<->camera link is the proprietary `cc dd ee ff` protocol (device const
e4 12 69 00), not DVRIP. Frame header: magic(4) + msgtype(u32 LE) + device-const(4) +
length(u32 LE) + payload. During PTZ the phone sends the camera only 20-byte keepalives
(type 0x01), so any NEW high-volume type phone->camera is the talk audio.

Answers, in order:
  1. Byte volume per (endpoint, port), phone as sender — the audio path stands out.
  2. cc-dd-ee-ff message types the phone sent the camera (new type = local audio).
  3. Local vs cloud verdict.
"""
import os, sys, struct
from collections import defaultdict, Counter

from scapy.all import rdpcap, IP, TCP, UDP, Raw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h32env

CAMERA_IP, PHONE_IP, GATEWAY_IP = h32env.CAMERA_IP, h32env.PHONE_IP, h32env.GATEWAY_IP
KEEPALIVE_TYPE, VIDEO_TYPE = 0x01, 0x9c45
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "talk_capture.pcap")

if not os.path.exists(PATH):
    sys.exit(f"no such capture: {PATH}\n(run capture/capture_talk.py first)")

pk = rdpcap(PATH)
print(f"{os.path.basename(PATH)}: {len(pk)} packets\n")


def label(ip):
    if ip == CAMERA_IP:
        return "CAMERA(local)"
    if PHONE_IP and ip.rsplit(".", 1)[0] == PHONE_IP.rsplit(".", 1)[0] and ip != GATEWAY_IP:
        return f"LAN {ip}"
    return f"CLOUD {ip}"


def ccddeeff_frames(payload):
    i, out = 0, []
    while i + 16 <= len(payload):
        if payload[i:i + 4] != b"\xcc\xdd\xee\xff":
            i += 1
            continue
        mtype = struct.unpack_from("<I", payload, i + 4)[0]
        length = struct.unpack_from("<I", payload, i + 12)[0]
        out.append((mtype, length))
        i += max(16, length if 0 < length < 200000 else 16)
    return out


# 1. volume, phone as sender
vol = defaultdict(int)
for p in pk:
    if IP not in p:
        continue
    lay = p[TCP] if TCP in p else (p[UDP] if UDP in p else None)
    if lay is None or Raw not in p:
        continue
    if p[IP].src == PHONE_IP:
        proto = "TCP" if TCP in p else "UDP"
        vol[(label(p[IP].dst), proto, lay.dport)] += len(bytes(p[Raw].load))

print("=" * 70)
print("PHONE -> ? : payload bytes (the audio path is the big one)")
print("=" * 70)
rows = sorted(vol.items(), key=lambda kv: -kv[1])
total = sum(v for _, v in rows) or 1
for (lab, proto, port), n in rows[:16]:
    print(f"  {lab:22} {proto} :{port:<6} {n:>9} B  {'#' * int(44 * n / total)}")

# 2. cc-dd-ee-ff message types phone -> camera
phone_frames = b"".join(bytes(p[Raw].load) for p in pk
                        if IP in p and TCP in p and Raw in p
                        and p[IP].src == PHONE_IP and p[TCP].dport == 23456)
types = Counter()
bytes_by_type = Counter()
for mt, ln in ccddeeff_frames(phone_frames):
    types[mt] += 1
    bytes_by_type[mt] += ln
print("\n" + "=" * 70)
print("cc-dd-ee-ff message types the PHONE sent the CAMERA")
print("=" * 70)
if not types:
    print("  (none — the phone sent the camera no cc-dd-ee-ff frames)")
for mt, n in types.most_common():
    tag = " keepalive" if mt == KEEPALIVE_TYPE else "  <<< NEW — candidate TALK AUDIO"
    print(f"  type 0x{mt:08x} ({mt:6}): {n:5} frames, {bytes_by_type[mt]:8} B{tag}")

# 3. verdict
new_local = [t for t in types if t != KEEPALIVE_TYPE]
top = rows[0][0][0] if rows else None
print("\n" + "=" * 70)
print("VERDICT:", end=" ")
if top and top.startswith("CAMERA") and new_local:
    t = new_local[0]
    print(f"TALK IS LOCAL — phone sent the camera a new cc-dd-ee-ff type 0x{t:08x}.")
    print(f"  Next: dump that type's payloads to work out the audio codec/framing, then a")
    print(f"  mic / play-a-.wav client sends the same frames straight to {CAMERA_IP}:23456.")
elif top and top.startswith("CLOUD"):
    print("TALK IS CLOUD-BROKERED — the audio flood went to the vendor cloud, not the")
    print("  camera. Same wall as PTZ; local injection will not work. Stays parked.")
elif top and top.startswith("CAMERA"):
    print("phone->camera was keepalives only — no local audio. Almost certainly")
    print("  cloud-brokered; check the CLOUD rows above.")
else:
    print("inconclusive — no clear audio path. Re-capture holding talk longer.")
