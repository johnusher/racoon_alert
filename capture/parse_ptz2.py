#!/usr/bin/env python3
"""Comprehensive parse: ALL ports phone<->camera, hunt every cc-dd-ee-ff frame."""
import os, sys, struct, collections
from scapy.all import rdpcap, IP, TCP, Raw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import h32env                                   # camera/LAN settings from local.env

PCAP = os.path.join(HERE, "ptz_capture.pcap")
CAM, PHONE = h32env.CAMERA_IP, h32env.PHONE_IP
MAGIC = b"\xcc\xdd\xee\xff"

pkts = rdpcap(PCAP)

conns = collections.Counter()          # (sport,dport,dir) -> bytes
p2c_payloads = collections.Counter()   # distinct phone->cam payloads
magic_frames = []                      # every cc-dd-ee-ff occurrence

for pk in pkts:
    if IP not in pk or TCP not in pk: continue
    src, dst = pk[IP].src, pk[IP].dst
    if {src,dst} != {CAM,PHONE}: continue
    sp, dp = pk[TCP].sport, pk[TCP].dport
    dirn = "P->C" if src==PHONE else "C->P"
    dport_service = dp if src==PHONE else sp
    if Raw in pk:
        d = bytes(pk[Raw].load)
        conns[(dirn, dport_service)] += len(d)
        if dirn=="P->C" and len(d) <= 160:
            p2c_payloads[(dport_service, d)] += 1
        # hunt magic anywhere in payload
        idx = d.find(MAGIC)
        while idx != -1:
            frag = d[idx:idx+72]
            magic_frames.append((dirn, dport_service, len(d), frag))
            idx = d.find(MAGIC, idx+4)

print("=== phone<->camera traffic by service port (bytes) ===")
for (dirn, po), n in sorted(conns.items(), key=lambda x:-x[1]):
    print(f"  {dirn}  port {po:>5}  {n:>9} bytes")

print(f"\n=== distinct PHONE->CAMERA payloads (<=160B): {len(p2c_payloads)} ===")
for (po, d), c in sorted(p2c_payloads.items(), key=lambda x:-x[1]):
    print(f"  port {po:>5}  x{c:<3} len={len(d):>3}  {d.hex()}")

print(f"\n=== every cc-dd-ee-ff frame seen (any direction/port): {len(magic_frames)} ===")
uniq = collections.OrderedDict()
for dirn, po, tot, frag in magic_frames:
    uniq.setdefault((dirn, po, frag), 0)
    uniq[(dirn, po, frag)] += 1
for (dirn, po, frag), c in uniq.items():
    fields = ""
    if len(frag) >= 16:
        t, cmd, ln = struct.unpack_from("<III", frag, 4)
        fields = f" type=0x{t:08x} cmd=0x{cmd:08x} len={ln}"
    ptz = ""
    if len(frag) >= 52:
        pan, tilt, zoom = struct.unpack_from("<iii", frag, 40)
        if pan or tilt or zoom:
            ptz = f"  >>> pan={pan:+d} tilt={tilt:+d} zoom={zoom:+d}"
    print(f"  {dirn} :{po} x{c:<3} {frag[:28].hex()}…{fields}{ptz}")
