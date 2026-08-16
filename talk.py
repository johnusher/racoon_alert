#!/usr/bin/env python3
"""
Two-way talk for the h32 camera — send audio to the camera's speaker.

    ./.venv/bin/python talk.py --say "hello"        # macOS `say` -> the camera
    ./.venv/bin/python talk.py path/to/clip.wav     # any wav -> the camera
    ./.venv/bin/python talk.py --mic [--secs N]     # live from the Mac mic
    ./.venv/bin/python talk.py --tone               # a 1s test beep

How this works (reverse-engineered 2026-08-16, see capture/ and TODO.md): unlike PTZ,
two-way talk is NOT cloud-brokered — the IPC360 app streams it straight to the camera on
TCP :23456 using the proprietary `cc dd ee ff` protocol. Each 40 ms of G.711 A-law audio
(8 kHz mono, 320 bytes) rides in one message of type 0x9c57:

    outer: cc dd ee ff | <type 0x9c57> | <const e4126900> | <total len u32>
    inner: 00000000 | <codec tag 0x29> | <device id> | <seq u32> | <320 bytes A-law>

interleaved with 20-byte keepalives (type 0x01). We just reproduce that byte-for-byte.
The device id + const are per-camera and live in local.env (gitignored) — this is a
public repo.
"""
import argparse, os, socket, struct, subprocess, sys, time, wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import h32env

PORT = 23456
TYPE_KEEPALIVE = 0x01
TYPE_AUDIO = 0x9c57
CODEC_TAG = 0x29                 # G.711 A-law, from the captured frames
FRAME_SAMPLES = 320              # 40 ms of 8 kHz audio per message
FRAME_SECS = FRAME_SAMPLES / 8000.0


# ---- G.711 A-law codec (Python 3.13+ dropped the `audioop` module) ----------

def _alaw_decode_table():
    """A-law byte -> PCM int16 (ITU G.711). This is the reference that correctly decoded
    the camera's captured audio, so the encoder is built as its exact inverse."""
    tbl = np.zeros(256, np.int16)
    for a in range(256):
        aa = a ^ 0x55
        seg = (aa & 0x70) >> 4
        val = ((aa & 0x0f) << 4) + 8
        if seg >= 1:
            val += 0x100
        if seg > 1:
            val <<= (seg - 1)
        tbl[a] = val if (a & 0x80) else -val
    return tbl


def _alaw_encode_table(dec):
    """PCM(int16) -> A-law byte: for each sample, the codeword whose decoded value is
    nearest. Inverting the decode table guarantees encode/decode agree exactly (and that
    re-encoding a decoded codeword returns it), so what we send matches what the app sends."""
    pcm = np.arange(65536)
    pcm = np.where(pcm >= 32768, pcm - 65536, pcm).astype(np.int32)
    order = np.argsort(dec.astype(np.int32))          # codewords sorted by decoded value
    dvals = dec.astype(np.int32)[order]
    idx = np.clip(np.searchsorted(dvals, pcm), 1, 255)
    left, right = dvals[idx - 1], dvals[idx]
    pick_left = (pcm - left) <= (right - pcm)
    return np.where(pick_left, order[idx - 1], order[idx]).astype(np.uint8)


ALAW_DEC = _alaw_decode_table()
ALAW_ENC = _alaw_encode_table(ALAW_DEC)


def pcm_to_alaw(pcm):
    return ALAW_ENC[np.asarray(pcm, np.int16).astype(np.uint16)].tobytes()


def alaw_to_pcm(data):
    return ALAW_DEC[np.frombuffer(data, np.uint8)]


# ---- audio sources -> 8 kHz mono int16 --------------------------------------

def _resample_mono8k(pcm, rate, channels):
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != 8000:
        n = int(round(len(pcm) * 8000 / rate))
        pcm = np.interp(np.linspace(0, len(pcm), n, endpoint=False),
                        np.arange(len(pcm)), pcm.astype(np.float32))
    return np.clip(pcm, -32768, 32767).astype(np.int16)


def load_wav(path):
    with wave.open(path, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: need 16-bit PCM wav (got {width*8}-bit)")
    return _resample_mono8k(np.frombuffer(raw, np.int16), rate, ch)


def say_to_pcm(text, voice=None):
    """macOS `say` -> wav -> 8 kHz mono. No third-party TTS needed.
    voice: a `say -v` name, e.g. "Anna" (German female). None = system default."""
    tmp = os.path.join(BASE, ".talk_say.wav")
    cmd = ["say", "-o", tmp, "--data-format=LEI16@22050"]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    try:
        subprocess.run(cmd, check=True)
        return load_wav(tmp)
    except FileNotFoundError:
        raise SystemExit("`say` not found — this is macOS-only (or pass a .wav).")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def tone_pcm(secs=1.0, hz=880):
    t = np.arange(int(8000 * secs)) / 8000.0
    return (np.sin(2 * np.pi * hz * t) * 8000).astype(np.int16)


# ---- the camera talk client -------------------------------------------------

class CameraTalk:
    """Streams 8 kHz mono PCM to the camera speaker as cc-dd-ee-ff A-law frames."""

    def __init__(self, ip=None, devid=None, const=None, port=PORT):
        self.ip = ip or h32env.CAMERA_IP
        self.port = port
        devid = devid or h32env.CAMERA_DEVID
        const = const or h32env.CAMERA_CONST
        if not devid:
            raise SystemExit(
                "no camera device id — talk needs H32_CAMERA_DEVID in local.env.\n"
                "  Capture it once: sudo ./.venv/bin/python capture/capture_talk.py")
        self.devid = bytes.fromhex(devid)
        self.const = bytes.fromhex(const)
        self.seq = 0
        self.sock = None

    def _frame(self, mtype, inner):
        return (b"\xcc\xdd\xee\xff" + struct.pack("<I", mtype) + self.const
                + struct.pack("<I", 16 + len(inner)) + inner)

    def _keepalive(self):
        return self._frame(TYPE_KEEPALIVE, b"\x00\x00\x00\x00")

    def _audio_frame(self, alaw320):
        inner = (b"\x00\x00\x00\x00" + struct.pack("<I", CODEC_TAG) + self.devid
                 + struct.pack("<I", self.seq) + alaw320)
        self.seq += 2                       # the app steps the counter by 2
        return self._frame(TYPE_AUDIO, inner)

    def __enter__(self):
        self.sock = socket.create_connection((self.ip, self.port), timeout=5)
        self.sock.sendall(self._keepalive())
        self.sock.sendall(self._keepalive())
        return self

    def __exit__(self, *a):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def play(self, pcm8k, realtime=True, on_progress=None, lead_silence=0.4):
        """Send 8 kHz mono int16 PCM. Paced at 40 ms/frame so the camera plays it live.

        lead_silence: seconds of silence sent first, to let the camera switch its
        half-duplex speaker on before the actual audio — without it, a short greeting
        loses its first fraction of a second."""
        pcm = np.asarray(pcm8k, np.int16)
        if lead_silence > 0:
            pcm = np.concatenate([np.zeros(int(lead_silence * 8000), np.int16), pcm])
        pad = (-len(pcm)) % FRAME_SAMPLES
        if pad:
            pcm = np.concatenate([pcm, np.zeros(pad, np.int16)])
        n = len(pcm) // FRAME_SAMPLES
        start = time.monotonic()
        for i in range(n):
            chunk = pcm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            self.sock.sendall(self._audio_frame(pcm_to_alaw(chunk)))
            if i % 12 == 0:
                self.sock.sendall(self._keepalive())
            if on_progress and i % 25 == 0:
                on_progress(i / n)
            if realtime:
                target = start + (i + 1) * FRAME_SECS
                slack = target - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
        return n * FRAME_SECS


def main():
    ap = argparse.ArgumentParser(description="Send audio to the h32 camera speaker.")
    ap.add_argument("wav", nargs="?", help="path to a .wav to play")
    ap.add_argument("--say", metavar="TEXT", help="speak TEXT via macOS `say`")
    ap.add_argument("--voice", metavar="NAME", help="`say` voice, e.g. Anna (German female)")
    ap.add_argument("--tone", action="store_true", help="send a 1s test beep")
    ap.add_argument("--mic", action="store_true", help="stream the Mac microphone live")
    ap.add_argument("--secs", type=float, default=10.0, help="--mic duration (default 10)")
    args = ap.parse_args()

    if args.mic:
        return stream_mic(args.secs)
    if args.say:
        pcm = say_to_pcm(args.say, args.voice)
    elif args.tone:
        pcm = tone_pcm()
    elif args.wav:
        pcm = load_wav(args.wav)
    else:
        ap.error("give a .wav, --say TEXT, --tone, or --mic")

    with CameraTalk() as talk:
        print(f"→ camera {talk.ip}:{talk.port}  ({len(pcm)/8000:.1f}s audio)")
        talk.play(pcm, on_progress=lambda f: print(f"\r  {f*100:3.0f}%", end="", flush=True))
    print("\r  done — the camera should have played it.")


def stream_mic(secs):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("--mic needs `sounddevice`: ../.venv/bin/pip install sounddevice")
    with CameraTalk() as talk:
        print(f"→ camera {talk.ip}  · speaking live for {secs:.0f}s (Ctrl-C to stop)…")
        buf = sd.rec(int(secs * 8000), samplerate=8000, channels=1, dtype="int16")
        sd.wait()
        talk.play(buf.reshape(-1))
    print("done.")


if __name__ == "__main__":
    main()
