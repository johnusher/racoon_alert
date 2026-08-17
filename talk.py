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
import argparse, os, shutil, socket, struct, subprocess, sys, tempfile, time, wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import h32env

PORT = 23456
TYPE_KEEPALIVE = 0x01
TYPE_AUDIO = 0x9c57
TYPE_TALK_CTRL = 0x4f35          # open/close the speaker (the app sends this to the cloud;
                                 # the camera also honours it sent straight to :23456)
CODEC_TAG = 0x29                 # G.711 A-law, from the captured frames
FRAME_SAMPLES = 320              # 40 ms of 8 kHz audio per message
FRAME_SECS = FRAME_SAMPLES / 8000.0
REOPEN_EVERY = 25                # re-send the speaker-open ~1s, so anything that closes it
                                 # (the app's mic release, a camera timeout) recovers fast


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
    # A PER-PROCESS temp file, not a fixed one. There is a detector per camera now, and
    # they start within a second of each other: with a shared path one process's `finally`
    # deletes the wav another is still reading, and the read then fails with
    # FileNotFoundError — which used to be reported as "`say` not found", blaming the one
    # thing that was fine.
    fd, tmp = tempfile.mkstemp(prefix=".talk_say_", suffix=".wav", dir=BASE)
    os.close(fd)
    cmd = ["say", "-o", tmp, "--data-format=LEI16@22050"]
    if voice:
        cmd += ["-v", voice]
    cmd.append(text)
    try:
        if shutil.which("say") is None:
            raise SystemExit("`say` not found — this is macOS-only (or pass a .wav).")
        subprocess.run(cmd, check=True)
        return load_wav(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def tone_pcm(secs=1.0, hz=880):
    t = np.arange(int(8000 * secs)) / 8000.0
    return (np.sin(2 * np.pi * hz * t) * 8000).astype(np.int16)


# SomaFM channels (free internet radio, 128k mp3) — a continuous stream is the easiest
# possible test of "does the camera speaker play anything at all?".
SOMA = {
    "groovesalad": "https://ice1.somafm.com/groovesalad-128-mp3",
    "dronezone":   "https://ice1.somafm.com/dronezone-128-mp3",
    "indiepop":    "https://ice1.somafm.com/indiepop-128-mp3",
    "u80s":        "https://ice1.somafm.com/u80s-128-mp3",
    "lush":        "https://ice1.somafm.com/lush-128-mp3",
    "beatblender": "https://ice1.somafm.com/beatblender-128-mp3",
    "spacestation":"https://ice1.somafm.com/spacestation-128-mp3",
}


def stream_url(url, secs=None):
    """Decode any audio URL (SomaFM, etc.) with ffmpeg and stream it to the camera
    speaker until Ctrl-C or `secs` elapses. ffmpeg does the mp3/aac decode + resample."""
    import shutil
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (needed to decode the stream).")
    cmd = ["ffmpeg", "-loglevel", "error", "-reconnect", "1", "-reconnect_streamed", "1",
           "-i", url, "-f", "s16le", "-ar", "8000", "-ac", "1", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    deadline = (time.monotonic() + secs) if secs else None
    stop = (lambda: deadline is not None and time.monotonic() >= deadline)
    try:
        with CameraTalk() as talk:
            print(f"→ camera {talk.ip}:{talk.port}  · streaming {url}")
            print("  (go and listen — Ctrl-C to stop)" if not secs else f"  for {secs:.0f}s…")
            talk.stream_pcm_stdout(proc.stdout, should_stop=stop)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("\nstream stopped.")


# ---- the camera talk client -------------------------------------------------

class CameraTalk:
    """Streams 8 kHz mono PCM to the camera speaker as cc-dd-ee-ff A-law frames."""

    def __init__(self, ip=None, devid=None, const=None, port=PORT, session=None):
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
        self.session = bytes.fromhex(session) if session else b"\x00\x00\x00\x00"
        self.seq = 0
        self.sock = None

    def _frame(self, mtype, inner):
        return (b"\xcc\xdd\xee\xff" + struct.pack("<I", mtype) + self.const
                + struct.pack("<I", 16 + len(inner)) + inner)

    def _keepalive(self):
        return self._frame(TYPE_KEEPALIVE, b"\x00\x00\x00\x00")

    def _talk_ctrl(self, start):
        """Open (start=True) / close the camera speaker. Inner:
        [zero][device id][session][01][flag][zero][zero]. The session is a cloud handle
        the camera does not seem to validate on a local connection, so zero works."""
        flag = b"\x01\x00\x00\x00" if start else b"\x00\x00\x00\x00"
        inner = (b"\x00\x00\x00\x00" + self.devid + self.session
                 + b"\x01\x00\x00\x00" + flag + b"\x00" * 8)
        return self._frame(TYPE_TALK_CTRL, inner)

    def _audio_frame(self, alaw320):
        inner = (b"\x00\x00\x00\x00" + struct.pack("<I", CODEC_TAG) + self.devid
                 + struct.pack("<I", self.seq) + alaw320)
        self.seq += 2                       # the app steps the counter by 2
        return self._frame(TYPE_AUDIO, inner)

    def __enter__(self):
        self.sock = socket.create_connection((self.ip, self.port), timeout=5)
        self.sock.sendall(self._keepalive())
        self.sock.sendall(self._keepalive())
        self.sock.sendall(self._talk_ctrl(True))     # open the speaker
        time.sleep(0.2)
        return self

    def __exit__(self, *a):
        if self.sock:
            try:
                self.sock.sendall(self._talk_ctrl(False))   # close the speaker
            except OSError:
                pass
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _send_frames(self, frames, realtime=True, on_progress=None, total=None):
        """Core sender: `frames` is any iterable of exactly-320-sample int16 chunks.
        Paces at 40 ms/frame so the camera plays it live, with periodic keepalives."""
        start = time.monotonic()
        i = 0
        for chunk in frames:
            self.sock.sendall(self._audio_frame(pcm_to_alaw(chunk)))
            if i % 12 == 0:
                self.sock.sendall(self._keepalive())
            if i and i % REOPEN_EVERY == 0:
                self.sock.sendall(self._talk_ctrl(True))   # keep the speaker open
            if on_progress and total and i % 25 == 0:
                on_progress(i / total)
            if realtime:
                slack = start + (i + 1) * FRAME_SECS - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
            i += 1
        return i * FRAME_SECS

    def play(self, pcm8k, realtime=True, on_progress=None, lead_silence=0.4):
        """Send finite 8 kHz mono int16 PCM.

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
        frames = (pcm[k * FRAME_SAMPLES:(k + 1) * FRAME_SAMPLES] for k in range(n))
        return self._send_frames(frames, realtime, on_progress, n)

    def stream_pcm_stdout(self, stdout, realtime=True, should_stop=None):
        """Stream unbounded 8 kHz mono s16le PCM from a file object (e.g. ffmpeg stdout)
        to the camera speaker until EOF, error, or should_stop() returns True."""
        def frames():
            buf = b""
            need = FRAME_SAMPLES * 2
            while not (should_stop and should_stop()):
                data = stdout.read(4096)
                if not data:
                    break
                buf += data
                while len(buf) >= need:
                    yield np.frombuffer(buf[:need], np.int16)
                    buf = buf[need:]
        return self._send_frames(frames(), realtime)


def main():
    ap = argparse.ArgumentParser(description="Send audio to the h32 camera speaker.")
    ap.add_argument("wav", nargs="?", help="path to a .wav to play")
    ap.add_argument("--say", metavar="TEXT", help="speak TEXT via macOS `say`")
    ap.add_argument("--voice", metavar="NAME", help="`say` voice, e.g. Anna (German female)")
    ap.add_argument("--tone", action="store_true", help="send a 1s test beep")
    ap.add_argument("--mic", action="store_true", help="stream the Mac microphone live")
    ap.add_argument("--stream", metavar="URL", help="stream an audio URL (mp3/aac) live")
    ap.add_argument("--soma", metavar="CH", nargs="?", const="groovesalad",
                    help=f"stream a SomaFM channel ({', '.join(SOMA)})")
    ap.add_argument("--secs", type=float, default=None, help="duration limit (seconds)")
    args = ap.parse_args()

    if args.soma:
        url = SOMA.get(args.soma) or SOMA["groovesalad"]
        return stream_url(url, args.secs)
    if args.stream:
        return stream_url(args.stream, args.secs)
    if args.mic:
        return stream_mic(args.secs or 10.0)
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
