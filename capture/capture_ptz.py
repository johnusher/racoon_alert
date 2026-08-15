#!/usr/bin/env python3
"""
Capture the IPC360 app's PTZ commands to our camera by MITM (ARP-spoof) on the LAN.

Routes traffic between the PHONE and the CAMERA through this Mac, sniffs the
proprietary control ports (23456 / 34567), and live-decodes any PTZ packets so we
can read THIS camera's real command bytes (incl. the device/session header token).

Run as root, from the project venv:
    sudo ***REMOVED-PATH***/Documents/h32/.venv/bin/python \
         ***REMOVED-PATH***/Documents/h32/capture/capture_ptz.py <PHONE_IP>

Then, in the IPC360 app, pan the camera: RIGHT, LEFT, UP, DOWN, ZOOM+ , ZOOM- ,
pausing ~1s between each. Watch this terminal — decoded PTZ lines appear live.
Press Ctrl-C to stop; it restores ARP and saves capture/ptz_capture.pcap.
"""
import sys, os, time, struct, threading, signal
from datetime import datetime

if os.geteuid() != 0:
    sys.exit("Must run as root (sudo). See header for the exact command.")

from scapy.all import (Ether, ARP, srp, sendp, sniff, wrpcap, conf, get_if_hwaddr)

CAMERA_IP = "***REMOVED-IP***"
PORTS = {23456, 34567}
IFACE = "en0"
PCAP = "***REMOVED-PATH***/Documents/h32/capture/ptz_capture.pcap"

if len(sys.argv) < 2:
    sys.exit("usage: capture_ptz.py <PHONE_IP>  (the phone running IPC360)")
PHONE_IP = sys.argv[1]
conf.iface = IFACE
conf.verb = 0
MY_MAC = get_if_hwaddr(IFACE)

def mac_of(ip):
    ans, _ = srp(Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst=ip), timeout=3, iface=IFACE, retry=2)
    for _, r in ans:
        return r.hwsrc
    return None

print(f"[*] interface {IFACE} ({MY_MAC})")
print(f"[*] resolving MACs — camera {CAMERA_IP}, phone {PHONE_IP} …")
cam_mac = mac_of(CAMERA_IP)
phone_mac = mac_of(PHONE_IP)
if not cam_mac or not phone_mac:
    sys.exit(f"[!] could not resolve MACs (camera={cam_mac}, phone={phone_mac}). "
             "Make sure the phone is awake and on this Wi-Fi.")
print(f"[*] camera {CAMERA_IP} = {cam_mac}")
print(f"[*] phone  {PHONE_IP} = {phone_mac}")

# enable IP forwarding so both sides keep working while we're in the middle
old_fwd = os.popen("sysctl -n net.inet.ip.forwarding").read().strip()
os.system("sysctl -w net.inet.ip.forwarding=1 >/dev/null")

stop = threading.Event()
def spoof_loop():
    # tell phone: CAMERA_IP is at MY_MAC ; tell camera: PHONE_IP is at MY_MAC
    p_to_phone = Ether(dst=phone_mac)/ARP(op=2, psrc=CAMERA_IP, hwsrc=MY_MAC, pdst=PHONE_IP, hwdst=phone_mac)
    p_to_cam   = Ether(dst=cam_mac)/ARP(op=2, psrc=PHONE_IP,  hwsrc=MY_MAC, pdst=CAMERA_IP, hwdst=cam_mac)
    while not stop.is_set():
        sendp(p_to_phone, iface=IFACE)
        sendp(p_to_cam,   iface=IFACE)
        stop.wait(2)

def restore():
    print("\n[*] restoring ARP tables …")
    for _ in range(5):
        sendp(Ether(dst=phone_mac)/ARP(op=2, psrc=CAMERA_IP, hwsrc=cam_mac,   pdst=PHONE_IP, hwdst=phone_mac), iface=IFACE)
        sendp(Ether(dst=cam_mac)/ARP(op=2, psrc=PHONE_IP,   hwsrc=phone_mac, pdst=CAMERA_IP, hwdst=cam_mac),   iface=IFACE)
        time.sleep(0.2)
    os.system(f"sysctl -w net.inet.ip.forwarding={old_fwd} >/dev/null")
    print(f"[*] forwarding restored to {old_fwd}")

DIRS = {(0,0,0):"STOP"}
def describe(payload):
    if len(payload) >= 52 and payload[:4] == b"\xcc\xdd\xee\xff":
        pan, tilt, zoom = struct.unpack_from("<iii", payload, 40)
        bits = []
        if pan:  bits.append(f"pan={pan:+d}({'RIGHT' if pan>0 else 'LEFT'})")
        if tilt: bits.append(f"tilt={tilt:+d}({'UP' if tilt>0 else 'DOWN'})")
        if zoom: bits.append(f"zoom={zoom:+d}({'IN' if zoom>0 else 'OUT'})")
        return "PTZ " + (" ".join(bits) if bits else "STOP (all-zero)")
    return None

pkts = []
seen_headers = set()
def on_pkt(pkt):
    from scapy.all import TCP, IP, Raw
    if IP not in pkt or TCP not in pkt or Raw not in pkt:
        return
    if pkt[TCP].sport not in PORTS and pkt[TCP].dport not in PORTS:
        return
    pkts.append(pkt)
    src, dst = pkt[IP].src, pkt[IP].dst
    data = bytes(pkt[Raw].load)
    arrow = f"{src}->{dst}:{pkt[TCP].dport}"
    d = describe(data)
    ts = datetime.now().strftime("%H:%M:%S")
    if d:
        hdr = data[:40].hex()
        print(f"  [{ts}] {arrow}  {d}")
        if hdr not in seen_headers:
            seen_headers.add(hdr)
            print(f"           header(40B): {hdr}")
            print(f"           full  ({len(data)}B): {data.hex()}")
    else:
        # show non-PTZ control payloads compactly (handshake/login/keepalive)
        print(f"  [{ts}] {arrow}  {len(data)}B  {data[:24].hex()}{'…' if len(data)>24 else ''}")

def cleanup(*_):
    stop.set(); restore()
    if pkts:
        wrpcap(PCAP, pkts)
        try: os.chmod(PCAP, 0o644)
        except Exception: pass
        print(f"[*] saved {len(pkts)} packets -> {PCAP}")
    else:
        print("[!] no control-port packets captured — see notes below.")
    os._exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 120
t = threading.Thread(target=spoof_loop, daemon=True); t.start()
print(f"\n[*] MITM active. Now PAN the camera in the IPC360 app (right/left/up/down/zoom).")
print(f"[*] Watching ports {sorted(PORTS)} between phone and camera.")
print(f"[*] Auto-stops in {DURATION}s (or press Ctrl-C).\n")
bpf = f"tcp and host {CAMERA_IP} and host {PHONE_IP}"
try:
    sniff(iface=IFACE, filter=bpf, prn=on_pkt, store=False, timeout=DURATION)
except Exception as e:
    print("[!] sniff error:", e)
cleanup()
