#!/usr/bin/env python3
"""Test local PTZ variants using OUR captured device constant e4 12 69 00. Frame-diff verified."""
import os, socket, struct, subprocess, time, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h32env                                   # camera settings from local.env

IP = h32env.CAMERA_IP; PORT = 23456
RTSP = h32env.rtsp(1)
W, H = 80, 45

# MiguelDLM 'right' (68 bytes) — offset 8 is device magic (his e3), 40/44/48 = pan/tilt/zoom
BASE = bytearray([0xcc,0xdd,0xee,0xff,0x77,0x4f,0x00,0x00,0xe3,0x12,0x69,0x00,
        0x48,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xaf,0x93,0xc6,0x3b,
        0x09,0xf7,0x4b,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00])
HELLO = bytes([0xcc,0xdd,0xee,0xff,0x01,0x00,0x00,0x00,0xe4,0x12,0x69,0x00,0x14,0x00,0x00,0x00,0,0,0,0])

def build(pan=0, tilt=0, zoom=0, devmagic=b"\xe4\x12\x69\x00", token=None):
    b = bytearray(BASE)
    b[8:12] = devmagic
    if token is not None: b[20:28] = token
    struct.pack_into("<iii", b, 40, pan, tilt, zoom)
    return bytes(b)

def grab():
    p = subprocess.run(["perl","-e","alarm shift; exec @ARGV","15","ffmpeg","-nostdin","-loglevel","quiet",
        "-rtsp_transport","tcp","-i",RTSP,"-frames:v","1","-vf",f"scale={W}:{H},format=gray","-f","rawvideo","-"],
        capture_output=True); return p.stdout
def mad(a,b):
    n=min(len(a),len(b)); return None if n==0 else sum(abs(a[i]-b[i]) for i in range(n))/n

def run(label, pan_cmd, stop_cmd, hello=False):
    A = grab()
    try:
        s=socket.socket(); s.settimeout(4); s.connect((IP,PORT))
        if hello: s.send(HELLO); time.sleep(0.3)
        s.send(pan_cmd); time.sleep(1.6); s.send(stop_cmd); s.close()
    except Exception as e:
        print(f"  {label}: send err {e}"); return
    time.sleep(0.9); B=grab(); d=mad(A,B)
    # return
    try:
        s=socket.socket(); s.settimeout(4); s.connect((IP,PORT))
        if hello: s.send(HELLO); time.sleep(0.3)
        rev=bytearray(pan_cmd); struct.pack_into("<i",rev,40,-struct.unpack_from("<i",pan_cmd,40)[0]); s.send(bytes(rev)); time.sleep(1.6); s.send(stop_cmd); s.close()
    except Exception: pass
    time.sleep(1.2)
    verdict = "*** MOVED! ***" if d and d>10 else "no change"
    print(f"  {label}: MAD={d:.2f}  {verdict}")

HIS = bytes([0xaf,0x93,0xc6,0x3b,0x09,0xf7,0x4b,0x01])
print("Testing local PTZ variants (pan right ~1.6s each, frame-diff)…\n")
run("A e4-fix + his token",       build(5, token=HIS),        build(0, token=HIS))
run("B e4-fix + zero token",      build(5, token=bytes(8)),   build(0, token=bytes(8)))
run("C e4-fix + his token +hello",build(5, token=HIS),        build(0, token=HIS), hello=True)
run("D e4-fix big vel(20)",       build(20, token=HIS),       build(0, token=HIS))
run("E e3 original (control)",    build(5, devmagic=b'\xe3\x12\x69\x00', token=HIS), build(0, devmagic=b'\xe3\x12\x69\x00', token=HIS))
print("\n(>10 = real movement; ~1 = noise)")
