#!/usr/bin/env python3
"""Parse the MITM pcap: pull control-port payloads phone<->camera, decode PTZ."""
import os, sys, struct, collections
from scapy.all import rdpcap, IP, TCP, Raw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import h32env                                   # camera/LAN settings from local.env

PCAP = os.path.join(HERE, "ptz_capture.pcap")
CAM, PHONE = h32env.CAMERA_IP, h32env.PHONE_IP
PORTS = {23456, 34567}

def describe(d):
    if len(d) >= 52 and d[:4] == b"\xcc\xdd\xee\xff":
        pan, tilt, zoom = struct.unpack_from("<iii", d, 40)
        bits = []
        if pan:  bits.append(f"pan={pan:+d}({'R' if pan>0 else 'L'})")
        if tilt: bits.append(f"tilt={tilt:+d}({'U' if tilt>0 else 'D'})")
        if zoom: bits.append(f"zoom={zoom:+d}({'IN' if zoom>0 else 'OUT'})")
        return "PTZ " + (" ".join(bits) if bits else "STOP")
    return None

pkts = rdpcap(PCAP)
# tally which ports carried phone->camera payloads
port_bytes = collections.Counter()
ptz_hits = []
payloads = []   # (dir, port, bytes)
for pk in pkts:
    if IP not in pk or TCP not in pk or Raw not in pk: continue
    sp, dp = pk[TCP].sport, pk[TCP].dport
    if sp not in PORTS and dp not in PORTS: continue
    d = bytes(pk[Raw].load)
    src, dst = pk[IP].src, pk[IP].dst
    direction = f"{'PHONE->CAM' if src==PHONE else 'CAM->PHONE' if src==CAM else src+'->'+dst}"
    port = dp if dp in PORTS else sp
    port_bytes[(direction, port)] += len(d)
    payloads.append((direction, port, d))
    desc = describe(d)
    if desc and src == PHONE:
        ptz_hits.append((port, desc, d))

print("=== control-port traffic (bytes) ===")
for (dr, po), n in sorted(port_bytes.items()):
    print(f"  {dr:12} :{po}  {n} bytes")

print(f"\n=== PTZ commands decoded (phone->camera): {len(ptz_hits)} ===")
seen = set()
for port, desc, d in ptz_hits:
    print(f"  :{port}  {desc}")
    key = d[:40]
    if key not in seen:
        seen.add(key)
        print(f"        header40: {d[:40].hex()}")
        print(f"        full{len(d):>3}: {d.hex()}")

# If no cc-dd-ee-ff PTZ frames, dump distinct short phone->cam payloads to eyeball the real format
if not ptz_hits:
    print("\n[!] No cc-dd-ee-ff PTZ frames. Distinct phone->camera control payloads (<=128B):")
    uniq = collections.OrderedDict()
    for dr, po, d in payloads:
        if dr == "PHONE->CAM" and len(d) <= 128:
            uniq.setdefault((po, d), 0)
            uniq[(po, d)] += 1
    for (po, d), c in list(uniq.items())[:60]:
        print(f"  :{po} x{c} len={len(d):>3}  {d.hex()}")
