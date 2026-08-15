#!/usr/bin/env python3
"""Local PTZ with OUR captured device id (d8a4c03b) + OUR e4 constant. Frame-diff verified."""
import socket, struct, subprocess, time

IP="***REMOVED-IP***"; PORT=23456
RTSP="rtsp://***REMOVED-CREDS***@***REMOVED-IP***:554/realmonitor?channel=0&stream=1.sdp"
W,H=80,45
HELLO=bytes([0xcc,0xdd,0xee,0xff,0x01,0x00,0x00,0x00,0xe4,0x12,0x69,0x00,0x14,0x00,0x00,0x00,0,0,0,0])
# MiguelDLM 68-byte template; offset8=devmagic, offset20-27=token, offset40/44/48=pan/tilt/zoom
BASE=bytearray([0xcc,0xdd,0xee,0xff,0x77,0x4f,0x00,0x00,0xe4,0x12,0x69,0x00,
        0x48,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xaf,0x93,0xc6,0x3b,
        0x09,0xf7,0x4b,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

def cmd(token, pan=0,tilt=0,zoom=0):
    b=bytearray(BASE); b[20:28]=token; struct.pack_into("<iii",b,40,pan,tilt,zoom); return bytes(b)
def grab():
    p=subprocess.run(["perl","-e","alarm shift; exec @ARGV","15","ffmpeg","-nostdin","-loglevel","quiet",
        "-rtsp_transport","tcp","-i",RTSP,"-frames:v","1","-vf",f"scale={W}:{H},format=gray","-f","rawvideo","-"],
        capture_output=True); return p.stdout
def mad(a,b):
    n=min(len(a),len(b)); return None if n==0 else sum(abs(a[i]-b[i]) for i in range(n))/n

def test(label, token, pan=5, hello=False):
    A=grab()
    try:
        s=socket.socket(); s.settimeout(4); s.connect((IP,PORT))
        if hello: s.send(HELLO); time.sleep(0.4)
        s.send(cmd(token,pan=pan)); time.sleep(1.6); s.send(cmd(token,0,0,0)); s.close()
    except Exception as e:
        print(f"  {label}: err {e}"); return
    time.sleep(0.9); B=grab(); d=mad(A,B)
    v="*** MOVED! ***" if d and d>10 else "no change"
    print(f"  {label}: MAD={d:.2f}  {v}")
    # return to origin
    try:
        s=socket.socket(); s.settimeout(4); s.connect((IP,PORT))
        if hello: s.send(HELLO); time.sleep(0.4)
        s.send(cmd(token,pan=-pan)); time.sleep(1.6); s.send(cmd(token,0,0,0)); s.close()
    except Exception: pass
    time.sleep(1.1)

DEV=bytes([0xd8,0xa4,0xc0,0x3b])
print("Local PTZ with captured device id d8a4c03b …\n")
test("1 token=d8a4c03b x2",      DEV+DEV, pan=5)
test("2 token=d8a4c03b+fd277202",DEV+bytes([0xfd,0x27,0x72,0x02]), pan=5)
test("3 token=d8a4c03b x2 +hello",DEV+DEV, pan=5, hello=True)
test("4 token=d8a4c03b x2 vel20", DEV+DEV, pan=20)
test("5 token=d8a4c03b+zero",     DEV+bytes(4), pan=5)
print("\n(>10 = movement)")
