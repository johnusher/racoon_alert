#!/usr/bin/env python3
"""
Capture an IPC360 two-way-talk session by MITM (ARP-spoof) on the LAN.

The one question this answers: when you press TALK in the app and speak, where does
the audio actually go?

  • phone -> camera:23456   → talk is LOCAL. It arrives as a NEW `cc dd ee ff` message
                              type the phone has not sent before (during PTZ the phone
                              sends the camera only 20-byte keepalives, type 0x01), so a
                              new high-volume type is the tell. That type is what a local
                              mic / play-a-.wav client would reproduce.
  • phone -> the vendor cloud → talk is CLOUD-BROKERED, the same wall that parked PTZ.
                              Injecting locally will not work.

This firmware's app<->camera protocol is the proprietary **`cc dd ee ff`** framing (device
const e4 12 69 00), NOT plaintext DVRIP — verified from the existing PTZ capture. So this
decodes `cc dd ee ff` frames, not DVRIP JSON. Header: magic(4) + msgtype(u32) +
device-const(4) + length(u32) + payload.

To see both links at once it spoofs the PHONE against BOTH the camera (so LAN traffic
routes through this Mac) AND the gateway (so the phone's internet traffic does too).
Audio is a continuous ~8 KB/s flood, so whichever link lights up is the answer.

Point it at YOUR OWN camera and YOUR OWN phone on YOUR OWN network — see capture/README.md.
Run as root, from the repo root:

    sudo ./.venv/bin/python capture/capture_talk.py [PHONE_IP] [SECONDS]

Then in the IPC360 app: open the camera, press and HOLD the talk/mic button and speak
("hello, hello") for ~5s, release, repeat two or three times. Watch this terminal.
Ctrl-C (or the timeout) restores ARP and saves capture/talk_capture.pcap.
"""
import sys, os, time, struct, threading, signal
from collections import defaultdict
from datetime import datetime

if os.geteuid() != 0:
    sys.exit("Must run as root (sudo). See the header for the exact command.")

from scapy.all import (Ether, ARP, srp, sendp, sniff, wrpcap, conf,
                       get_if_hwaddr, TCP, UDP, IP, Raw)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import h32env

CAMERA_IP = h32env.CAMERA_IP
GATEWAY_IP = h32env.GATEWAY_IP
IFACE = h32env.IFACE
PHONE_IP = sys.argv[1] if len(sys.argv) > 1 else h32env.PHONE_IP
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 90
CONTROL_PORTS = {23456, 34567}
PCAP = os.path.join(HERE, "talk_capture.pcap")

if not PHONE_IP:
    sys.exit("usage: capture_talk.py <PHONE_IP> [seconds]  (or set H32_PHONE_IP in local.env)")

conf.iface = IFACE
conf.verb = 0
MY_MAC = get_if_hwaddr(IFACE)


def mac_of(ip):
    ans, _ = srp(Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip), timeout=3, iface=IFACE, retry=2)
    for _, r in ans:
        return r.hwsrc
    return None


print(f"[*] interface {IFACE} ({MY_MAC})")
print(f"[*] resolving MACs — camera {CAMERA_IP}, phone {PHONE_IP}, gateway {GATEWAY_IP} …")
cam_mac = mac_of(CAMERA_IP)
phone_mac = mac_of(PHONE_IP)
gw_mac = mac_of(GATEWAY_IP)
if not all((cam_mac, phone_mac, gw_mac)):
    sys.exit(f"[!] could not resolve MACs (camera={cam_mac}, phone={phone_mac}, "
             f"gateway={gw_mac}). Make sure the phone is awake and on this Wi-Fi.")
print(f"[*] camera  {CAMERA_IP} = {cam_mac}")
print(f"[*] phone   {PHONE_IP} = {phone_mac}")
print(f"[*] gateway {GATEWAY_IP} = {gw_mac}")

old_fwd = os.popen("sysctl -n net.inet.ip.forwarding").read().strip()
os.system("sysctl -w net.inet.ip.forwarding=1 >/dev/null")

stop = threading.Event()


def spoof_loop():
    # phone thinks camera AND gateway are us; camera+gateway think the phone is us.
    frames = [
        Ether(dst=phone_mac) / ARP(op=2, psrc=CAMERA_IP, hwsrc=MY_MAC, pdst=PHONE_IP, hwdst=phone_mac),
        Ether(dst=phone_mac) / ARP(op=2, psrc=GATEWAY_IP, hwsrc=MY_MAC, pdst=PHONE_IP, hwdst=phone_mac),
        Ether(dst=cam_mac) / ARP(op=2, psrc=PHONE_IP, hwsrc=MY_MAC, pdst=CAMERA_IP, hwdst=cam_mac),
        Ether(dst=gw_mac) / ARP(op=2, psrc=PHONE_IP, hwsrc=MY_MAC, pdst=GATEWAY_IP, hwdst=gw_mac),
    ]
    while not stop.is_set():
        for f in frames:
            sendp(f, iface=IFACE)
        stop.wait(2)


def restore():
    print("\n[*] restoring ARP tables …")
    truth = [
        Ether(dst=phone_mac) / ARP(op=2, psrc=CAMERA_IP, hwsrc=cam_mac, pdst=PHONE_IP, hwdst=phone_mac),
        Ether(dst=phone_mac) / ARP(op=2, psrc=GATEWAY_IP, hwsrc=gw_mac, pdst=PHONE_IP, hwdst=phone_mac),
        Ether(dst=cam_mac) / ARP(op=2, psrc=PHONE_IP, hwsrc=phone_mac, pdst=CAMERA_IP, hwdst=cam_mac),
        Ether(dst=gw_mac) / ARP(op=2, psrc=PHONE_IP, hwsrc=phone_mac, pdst=GATEWAY_IP, hwdst=gw_mac),
    ]
    for _ in range(5):
        for f in truth:
            sendp(f, iface=IFACE)
        time.sleep(0.2)
    os.system(f"sysctl -w net.inet.ip.forwarding={old_fwd} >/dev/null")
    print(f"[*] forwarding restored to {old_fwd}")


KEEPALIVE_TYPE = 0x01           # the 20-byte heartbeat the phone sends during PTZ
VIDEO_TYPE = 0x9c45             # camera->phone media stream (from the PTZ capture)


def ccddeeff_types(payload):
    """Yield each cc-dd-ee-ff message type in a TCP payload: (msgtype, frame_len)."""
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


# --- live tallies -----------------------------------------------------------
bytes_to = defaultdict(int)     # (endpoint_label, port) -> bytes  (phone as source)
phone_types = defaultdict(int)  # cc-dd-ee-ff msgtype phone->camera -> count
pkts = []
new_types_seen = set()
LOCK = threading.Lock()


def label(ip):
    if ip == CAMERA_IP:
        return "CAMERA(local)"
    if ip.rsplit(".", 1)[0] == PHONE_IP.rsplit(".", 1)[0] and ip != GATEWAY_IP:
        return f"LAN {ip}"
    return f"CLOUD {ip}"


def on_pkt(pkt):
    if IP not in pkt:
        return
    lay = pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)
    if lay is None:
        return
    pkts.append(pkt)
    src, dst = pkt[IP].src, pkt[IP].dst
    plen = len(bytes(pkt[Raw].load)) if Raw in pkt else 0
    ts = datetime.now().strftime("%H:%M:%S")

    if src == PHONE_IP and plen:
        with LOCK:
            bytes_to[(label(dst), lay.dport)] += plen

    # decode the phone's cc-dd-ee-ff frames to the camera: a NEW type during talk is audio
    if src == PHONE_IP and dst == CAMERA_IP and Raw in pkt:
        for mtype, flen in ccddeeff_types(bytes(pkt[Raw].load)):
            with LOCK:
                phone_types[mtype] += 1
            if mtype not in (KEEPALIVE_TYPE,) and mtype not in new_types_seen:
                new_types_seen.add(mtype)
                print(f"  [{ts}] phone->CAMERA  NEW cc-dd-ee-ff type 0x{mtype:08x} "
                      f"({mtype}) len={flen}  <<< not a keepalive — candidate talk audio")


def summary():
    print("\n" + "=" * 66)
    print("WHERE THE PHONE'S BYTES WENT (phone as sender)")
    print("=" * 66)
    with LOCK:
        rows = sorted(bytes_to.items(), key=lambda kv: -kv[1])
    if not rows:
        print("  nothing — did the talk button actually get pressed?")
        return
    total = sum(v for _, v in rows)
    for (lab, port), n in rows[:14]:
        bar = "#" * int(40 * n / total)
        print(f"  {lab:20} :{port:<6} {n:>9} B  {bar}")
    with LOCK:
        ptypes = dict(phone_types)
    print("\n  cc-dd-ee-ff types phone SENT the camera:")
    for mt, n in sorted(ptypes.items(), key=lambda kv: -kv[1]):
        note = " (keepalive)" if mt == KEEPALIVE_TYPE else "  <<< candidate talk audio"
        print(f"    type 0x{mt:08x} ({mt}): {n}{note if mt != KEEPALIVE_TYPE else ' (keepalive)'}")

    top_lab = rows[0][0][0]
    new_local = [t for t in ptypes if t != KEEPALIVE_TYPE]
    print("\n  VERDICT:", end=" ")
    if top_lab.startswith("CAMERA") and new_local:
        print("audio went to the CAMERA locally as a new cc-dd-ee-ff type → talk is LOCAL.")
        print("  Parse the pcap (parse_talk.py) to pin the audio message format.")
    elif top_lab.startswith("CLOUD"):
        print("the bulk went to the CLOUD → talk is cloud-brokered, same wall as PTZ.")
        print("  Local injection will not work; parked for the same reason.")
    elif top_lab.startswith("CAMERA"):
        print("bytes went to the camera but only as keepalives — likely cloud-brokered.")
        print("  Check the CLOUD rows above; re-run holding talk longer if unsure.")
    else:
        print(f"inconclusive — most bytes went to {top_lab}. Re-run and hold talk longer.")


def cleanup(*_):
    stop.set()
    restore()
    if pkts:
        wrpcap(PCAP, pkts)
        try:
            os.chmod(PCAP, 0o644)
        except Exception:
            pass
        print(f"[*] saved {len(pkts)} packets -> {PCAP}")
    summary()
    os._exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

threading.Thread(target=spoof_loop, daemon=True).start()
print(f"\n[*] MITM active. In IPC360: open the camera, then PRESS & HOLD talk and speak")
print(f"    ('hello, hello') for ~5s, release, and repeat 2–3 times.")
print(f"[*] Auto-stops in {DURATION}s (or Ctrl-C).\n")
try:
    sniff(iface=IFACE, filter=f"host {PHONE_IP}", prn=on_pkt, store=False, timeout=DURATION)
except Exception as e:
    print("[!] sniff error:", e)
cleanup()
