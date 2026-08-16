#!/usr/bin/env python3
"""Analyze the camera<->cloud capture: plaintext PTZ vs TLS."""
import os, sys, struct, collections
from scapy.all import rdpcap, IP, TCP, Raw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import h32env                                   # camera/LAN settings from local.env

PCAP=os.path.join(HERE, "cloud_capture.pcap")
CAM=h32env.CAMERA_IP; LAN=CAM.rsplit(".",1)[0]+"."; MAGIC=b"\xcc\xdd\xee\xff"
pkts=rdpcap(PCAP)

ext_ips=collections.Counter(); frames=[]; tls=collections.Counter(); other=[]
for pk in pkts:
    if IP not in pk or TCP not in pk: continue
    src,dst=pk[IP].src,pk[IP].dst
    if CAM not in (src,dst): continue
    peer = dst if src==CAM else src
    if peer.startswith(LAN): continue
    ext_ips[peer]+=1
    if Raw not in pk: continue
    d=bytes(pk[Raw].load)
    arrow="CLOUD->CAM" if dst==CAM else "CAM->CLOUD"
    port = pk[TCP].dport if dst==CAM else pk[TCP].sport
    if d[:4]==MAGIC:
        frames.append((arrow,peer,port,d))
    elif d and d[0] in (0x14,0x15,0x16,0x17):
        tls[(arrow,peer,port)]+=1
    else:
        other.append((arrow,peer,port,d))

print("=== external peers (cloud) ===")
for ip,n in ext_ips.most_common(): print(f"  {ip}  {n} pkts")
print(f"\n=== plaintext cc-dd-ee-ff cloud frames: {len(frames)} ===")
for arrow,peer,port,d in frames:
    t,cmd,ln=(struct.unpack_from('<III',d,4)+(0,0,0))[:3] if len(d)>=16 else (0,0,0)
    print(f"  {arrow} {peer}:{port} len={len(d)} type=0x{t:08x} cmd=0x{cmd:08x}")
    print(f"      {d.hex()}")
print(f"\n=== TLS-looking record streams: {sum(tls.values())} pkts ===")
for (arrow,peer,port),n in tls.items(): print(f"  {arrow} {peer}:{port}  x{n}")
print(f"\n=== other (non-magic, non-TLS) payloads: {len(other)} ===")
for arrow,peer,port,d in other[:12]:
    print(f"  {arrow} {peer}:{port} len={len(d)}  {d[:40].hex()}")
