#!/usr/bin/env python3
"""
Capture the CAMERA's traffic to the internet (cloud) by MITM (ARP-spoof camera<->gateway),
to see whether cloud-delivered PTZ commands arrive in plaintext (cc-dd-ee-ff) or TLS.

Run as root from the venv, then pan in the IPC360 app:
    sudo ***REMOVED-PATH***/Documents/h32/.venv/bin/python \
         ***REMOVED-PATH***/Documents/h32/capture/capture_cloud.py
Auto-stops after 150s. Live-prints any cc-dd-ee-ff frames to/from the cloud.
"""
import sys, os, time, struct, threading, signal, collections
from datetime import datetime

if os.geteuid() != 0:
    sys.exit("Must run as root (sudo). See header for the command.")

from scapy.all import Ether, ARP, srp, sendp, sniff, wrpcap, IP, TCP, Raw, conf, get_if_hwaddr

CAMERA_IP = "***REMOVED-IP***"
GATEWAY_IP = "***REMOVED-IP***"
IFACE = "en0"
LAN_PREFIX = "192.168.1."
PCAP = "***REMOVED-PATH***/Documents/h32/capture/cloud_capture.pcap"
DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 150
MAGIC = b"\xcc\xdd\xee\xff"
conf.iface = IFACE; conf.verb = 0
MY_MAC = get_if_hwaddr(IFACE)

def mac_of(ip):
    ans, _ = srp(Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst=ip), timeout=3, iface=IFACE, retry=2)
    for _, r in ans: return r.hwsrc
    return None

print(f"[*] resolving MACs (camera {CAMERA_IP}, gateway {GATEWAY_IP}) …")
cam_mac = mac_of(CAMERA_IP); gw_mac = mac_of(GATEWAY_IP)
if not cam_mac or not gw_mac:
    sys.exit(f"[!] could not resolve MACs (camera={cam_mac}, gateway={gw_mac}). Is the camera back online?")
print(f"[*] camera={cam_mac}  gateway={gw_mac}  me={MY_MAC}")

old_fwd = os.popen("sysctl -n net.inet.ip.forwarding").read().strip()
os.system("sysctl -w net.inet.ip.forwarding=1 >/dev/null")

stop = threading.Event()
def spoof():
    to_cam = Ether(dst=cam_mac)/ARP(op=2, psrc=GATEWAY_IP, hwsrc=MY_MAC, pdst=CAMERA_IP, hwdst=cam_mac)
    to_gw  = Ether(dst=gw_mac)/ARP(op=2, psrc=CAMERA_IP,  hwsrc=MY_MAC, pdst=GATEWAY_IP, hwdst=gw_mac)
    while not stop.is_set():
        sendp(to_cam, iface=IFACE); sendp(to_gw, iface=IFACE); stop.wait(2)

def restore():
    print("\n[*] restoring ARP …")
    for _ in range(5):
        sendp(Ether(dst=cam_mac)/ARP(op=2, psrc=GATEWAY_IP, hwsrc=gw_mac,  pdst=CAMERA_IP, hwdst=cam_mac), iface=IFACE)
        sendp(Ether(dst=gw_mac)/ARP(op=2, psrc=CAMERA_IP,  hwsrc=cam_mac, pdst=GATEWAY_IP, hwdst=gw_mac), iface=IFACE)
        time.sleep(0.2)
    os.system(f"sysctl -w net.inet.ip.forwarding={old_fwd} >/dev/null")

pkts = []; ext_frames = collections.Counter(); ext_ips = collections.Counter(); tls_bytes = 0
def on_pkt(pk):
    global tls_bytes
    if IP not in pk or TCP not in pk: return
    src, dst = pk[IP].src, pk[IP].dst
    if CAMERA_IP not in (src, dst): return
    other = dst if src == CAMERA_IP else src
    if other.startswith(LAN_PREFIX): return   # skip LAN peers; we want cloud
    pkts.append(pk)
    ext_ips[other] += 1
    if Raw not in pk: return
    d = bytes(pk[Raw].load)
    ts = datetime.now().strftime("%H:%M:%S")
    if d[:4] == MAGIC:
        t, cmd, ln = (struct.unpack_from("<III", d, 4) + (0,0,0))[:3] if len(d) >= 16 else (0,0,0)
        arrow = "CLOUD->CAM" if dst == CAMERA_IP else "CAM->CLOUD"
        key = (arrow, other, d[:16])
        ext_frames[key] += 1
        if ext_frames[key] <= 3 or len(d) <= 96:   # show new/short frames live
            print(f"  [{ts}] {arrow} {other} len={len(d)}  {d[:32].hex()}  (type=0x{t:08x} cmd=0x{cmd:08x})")
    elif d[0] in (0x16, 0x17) and dst == CAMERA_IP:
        tls_bytes += len(d)   # TLS record types (handshake/appdata)

def cleanup(*_):
    stop.set(); restore()
    if pkts:
        wrpcap(PCAP, pkts)
        try: os.chmod(PCAP, 0o644)
        except Exception: pass
    print(f"\n[*] external peers (cloud) seen: {dict(ext_ips)}")
    print(f"[*] plaintext cc-dd-ee-ff cloud frames: {sum(ext_frames.values())} ; TLS-looking bytes to cam: {tls_bytes}")
    print(f"[*] saved {len(pkts)} pkts -> {PCAP}")
    os._exit(0)

signal.signal(signal.SIGINT, cleanup); signal.signal(signal.SIGTERM, cleanup)
threading.Thread(target=spoof, daemon=True).start()
print(f"\n[*] MITM active on camera<->internet. PAN in the IPC360 app now (right/left/up/down).")
print(f"[*] Auto-stops in {DURATION}s. Watching for plaintext cloud commands…\n")
try:
    sniff(iface=IFACE, filter=f"tcp and host {CAMERA_IP}", prn=on_pkt, store=False, timeout=DURATION)
except Exception as e:
    print("[!] sniff error:", e)
cleanup()
