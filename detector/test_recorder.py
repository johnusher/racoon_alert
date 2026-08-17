#!/usr/bin/env python3
"""
Tests for the recorder's stray-ffmpeg matching.

Worth pinning because this function decides what gets SIGKILL'd. Too loose and it kills
an unrelated ffmpeg on the machine; too tight and orphaned recorders keep running — and
an orphan is not merely wasted CPU, it holds an RTSP connection to the camera open and
writes seg_*.ts into the same buffer with the same naming as the live recorder, so
save_event() splices clips out of two unsynchronised streams. That happened for real on
2026-08-17: an orphan from a force-killed detector at 07:18 was still interleaving
segments with the live recorder at 07:46.

Run:  ../.venv/bin/python detector/test_recorder.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recorder import stray_recorder_pids

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


BUF = "/Users/jhon/Documents/h32/detector/buffer"
# Real `ps -axo pid=,command=` lines, trimmed. 53219 is the actual orphan that was found.
PS = f"""
  501 /sbin/launchd
25289 ./go2rtc -config ./go2rtc.yaml
53219 ffmpeg -nostdin -loglevel warning -rtsp_transport tcp -i rtsp://127.0.0.1:8554/camera -c:v copy -c:a aac -f segment -strftime 1 {BUF}/seg_%Y%m%d_%H%M%S.ts
56539 /opt/homebrew/.../Python /Users/jhon/Documents/h32/detector/detect.py
56561 ffmpeg -nostdin -loglevel warning -rtsp_transport tcp -i rtsp://127.0.0.1:8554/camera -c:v copy -c:a aac -f segment -strftime 1 {BUF}/seg_%Y%m%d_%H%M%S.ts
70001 ffmpeg -i /Users/someone/holiday.mov -c:v libx264 /Users/someone/holiday.mp4
70002 ffmpeg -i rtsp://other -f segment /Users/jhon/OTHER-PROJECT/buffer/seg_%Y%m%d.ts
70003 /usr/bin/vlc {BUF}/seg_20260817_074651.ts
"""

print("1. it finds the recorder ffmpegs writing to our buffer")
got = stray_recorder_pids(PS, BUF)
check("both of our ffmpegs matched", set(got) == {53219, 56561}, str(got))

print("\n2. it does not touch anything else on the machine")
check("an unrelated ffmpeg transcode is spared", 70001 not in got)
check("an ffmpeg writing to a DIFFERENT buffer is spared", 70002 not in got,
      "matching on 'ffmpeg' alone would have killed it")
check("a non-ffmpeg process merely reading our buffer is spared", 70003 not in got,
      "vlc playing a segment is not a stray recorder")
check("go2rtc is spared", 25289 not in got)
check("the detector itself is spared", 56539 not in got)

print("\n3. the caller never kills itself")
check("our own pid is excluded", 56561 not in stray_recorder_pids(PS, BUF, me=56561),
      str(stray_recorder_pids(PS, BUF, me=56561)))
check("…and the others are still found", stray_recorder_pids(PS, BUF, me=56561) == [53219])

print("\n4. junk input cannot make it kill something")
for bad in ("", "\n\n", "not a ps listing at all", "abc ffmpeg " + BUF,
            "  \n  12x ffmpeg " + BUF):
    got = stray_recorder_pids(bad, BUF)
    check(f"{bad[:28]!r} yields nothing", got == [], str(got))

print("\n5. an empty buffer path must not match everything")
# A bug here would kill every ffmpeg on the machine, so make the failure mode explicit.
got = stray_recorder_pids(PS, "/no/such/buffer/anywhere")
check("a buffer dir nothing uses matches nothing", got == [], str(got))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all recorder stray-matching checks passed")
