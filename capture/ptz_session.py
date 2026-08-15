#!/usr/bin/env python3
"""Open a live session: send hello, read the camera-assigned token, send PTZ with it."""
import socket, struct, subprocess, time

IP="***REMOVED-IP***"; PORT=23456
RTSP="rtsp://***REMOVED-CREDS***@***REMOVED-IP***:554/realmonitor?channel=0&stream=1.sdp"
W,H=80,45
HELLO=bytes([0xcc,0xdd,0xee,0xff,0x01,0x00,0x00,0x00,0xe4,0x12,0x69,0x00,0x14,0x00,0x00,0x00,0,0,0,0])
BASE=bytearray([0xcc,0xdd,0xee,0xff,0x77,0x4f,0x00,0x00,0xe4,0x12,0x69,0x00,
        0x48,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xaf,0x93,0xc6,0x3b,
        0x09,0xf7,0x4b,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

def grab():
    p=subprocess.run(["perl","-e","alarm shift; exec @ARGV","15","ffmpeg","-nostdin","-loglevel","quiet",
        "-rtsp_transport","tcp","-i",RTSP,"-frames:v","1","-vf",f"scale={W}:{H},format=gray","-f","rawvideo","-"],
        capture_output=True); return p.stdout
def mad(a,b):
    n=min(len(a),len(b)); return None if n==0 else sum(abs(a[i]-b[i]) for i in range(n))/n

def open_session():
    s=socket.socket(); s.settimeout(4); s.connect((IP,PORT))
    s.send(HELLO)
    buf=b""
    try:
        for _ in range(6):
            buf+=s.recv(4096)
            if len(buf)>=64: break
    except Exception: pass
    tok=None
    i=buf.find(b"\xcc\xdd\xee\xff")
    if i!=-1 and len(buf)>=i+28:
        tok=buf[i+20:i+28]
    print(f"  camera reply {len(buf)}B; first frame hdr: {buf[i:i+28].hex() if i!=-1 else 'none'}")
    print(f"  -> session token @20: {tok.hex() if tok else 'NOT FOUND'}")
    return s, tok

def ptz(sock, token, pan=0, tilt=0, zoom=0):
    b=bytearray(BASE); b[20:28]=token; struct.pack_into("<iii",b,40,pan,tilt,zoom); sock.send(bytes(b))

print("Opening session, reading camera token, then PTZ pan-right…\n")
A=grab()
s,tok=open_session()
if tok:
    ptz(s,tok,pan=5); time.sleep(1.8); ptz(s,tok,0,0,0)
    time.sleep(0.9); B=grab(); d=mad(A,B)
    print(f"\n  MAD after pan: {d:.2f}  {'*** MOVED! ***' if d and d>10 else 'no change'}")
    # return
    ptz(s,tok,pan=-5); time.sleep(1.8); ptz(s,tok,0,0,0); s.close()
else:
    print("  no token — cannot form PTZ")
    try: s.close()
    except Exception: pass
