#!/usr/bin/env python3
"""
Tests for the talk codec + framing — the parts that must be byte-exact or the camera
plays garbage. Validated against the REAL captured app frames where possible.

Run:  ./.venv/bin/python test_talk.py
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


print("1. A-law codec round-trips the signal")
# a speech-like sweep
t = np.arange(8000) / 8000.0
pcm = (np.sin(2 * np.pi * 300 * t) * 12000 * np.sin(2 * np.pi * 3 * t)).astype(np.int16)
back = talk.alaw_to_pcm(talk.pcm_to_alaw(pcm)).astype(np.float32)
corr = np.corrcoef(pcm.astype(np.float32), back)[0, 1]
rel = np.abs(back - pcm).mean() / (np.abs(pcm).mean() + 1)
check("PCM->A-law->PCM correlation > 0.999", corr > 0.999, f"corr={corr:.5f}")
check("mean relative error < 5%", rel < 0.05, f"{rel*100:.1f}%")

print("\n2. A-law encoder is idempotent (decode then re-encode is stable)")
# start from valid A-law bytes, decode, re-encode: standard A-law is a fixed point
alaw0 = bytes(range(256))
alaw1 = talk.pcm_to_alaw(talk.alaw_to_pcm(alaw0))
same = sum(a == b for a, b in zip(alaw0, alaw1))
check("re-encode matches >= 254/256 codewords", same >= 254, f"{same}/256")

print("\n3. codec validated against the captured app audio (if the pcap is present)")
pcap = "capture/talk_capture.pcap"
if os.path.exists(pcap):
    from scapy.all import rdpcap, IP, TCP, Raw
    import h32env
    pk = rdpcap(pcap)
    buf = b"".join(bytes(p[Raw].load) for p in pk if IP in p and TCP in p and Raw in p
                   and p[IP].src == h32env.PHONE_IP and p[TCP].dport == 23456)
    # pull the first audio frame's 320 A-law bytes
    i, alaw = 0, None
    while i + 16 <= len(buf):
        if buf[i:i+4] != b"\xcc\xdd\xee\xff":
            i += 1; continue
        mt = struct.unpack_from("<I", buf, i+4)[0]
        ln = struct.unpack_from("<I", buf, i+12)[0]
        if not (16 <= ln < 200000) or i+ln > len(buf):
            i += 4; continue
        if mt == talk.TYPE_AUDIO:
            alaw = buf[i+16+16:i+ln]        # strip outer(16)+inner(16) headers
            break
        i += ln
    if alaw:
        # decode the app's A-law with our table, re-encode, compare
        re = talk.pcm_to_alaw(talk.alaw_to_pcm(alaw))
        agree = sum(a == b for a, b in zip(alaw, re)) / len(alaw)
        check("our codec reproduces the app's A-law (>= 98%)", agree >= 0.98,
              f"{agree*100:.1f}% of {len(alaw)} bytes")
    else:
        print("  SKIP  no audio frame found in the capture")
else:
    print("  SKIP  no capture/talk_capture.pcap on disk")

print("\n4. framing is byte-exact")
ct = talk.CameraTalk(devid="d8a4c03b", const="e4126900")
ka = ct._keepalive()
check("keepalive is the exact 20 bytes the app sends",
      ka == bytes.fromhex("ccddeeff01000000e4126900140000000000 0000".replace(" ", "")),
      ka.hex())
ct.seq = 0
frame = ct._audio_frame(b"\x55" * talk.FRAME_SAMPLES)
check("audio frame length = 16 outer + 16 inner + 320 audio", len(frame) == 352, str(len(frame)))
check("audio outer header: magic+type 0x9c57+const+len",
      frame[:16] == bytes.fromhex("ccddeeff") + struct.pack("<I", 0x9c57)
      + bytes.fromhex("e4126900") + struct.pack("<I", 352))
check("audio inner header: zero+codec 0x29+devid+seq",
      frame[16:32] == b"\x00\x00\x00\x00" + struct.pack("<I", 0x29)
      + bytes.fromhex("d8a4c03b") + struct.pack("<I", 0))
check("sequence counter advances by 2", ct.seq == 2, f"seq={ct.seq}")

print("\n5. any wav resamples to 8 kHz mono int16")
stereo = np.zeros((16000, 2), np.int16)
mono8 = talk._resample_mono8k(stereo.reshape(-1), 16000, 2)
check("stereo 16 kHz -> mono 8 kHz halves the sample count", len(mono8) == 8000, str(len(mono8)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all talk codec/framing checks passed")
