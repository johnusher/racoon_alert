#!/usr/bin/env python3
"""
Who is that? — face identification for the h32 detector.

Built on what the 2026-08-16 spike measured on this camera's own night-IR footage:

  • Recognition works when a face is visible. The one real person in the archive
    matched himself 10/10 times across frames, cosine 0.517–0.833, against SFace's
    0.363 same-identity threshold. Faces were 107–117 px, YuNet confident at 0.87–0.90.

  • A face is only visible in ~38% of the frames a person is detected in — he is often
    turned away or too far. So identity is decided PER VISIT by voting across every
    face seen, never from a single frame.

  • Whole-frame face detection hallucinates: 13 "faces" in 39 frames of an EMPTY garden,
    every one of them a plastic bucket whose two dark marks read as eyes. So faces are
    only ever looked for INSIDE a person box that has already passed the scenery filter.

  • The false-accept rate is UNMEASURED — the archive contains exactly one identified
    human, so we know he matches himself but not that he fails to match anyone else.
    Everything here therefore fails towards "unknown": a match must clear a threshold
    above SFace's own, must beat the runner-up by a margin, and must be seen more than
    once. Guessing is worse than admitting ignorance.

Two pieces, deliberately separable:
  FaceMatcher     — pure numpy. Enrolled embeddings, matching, voting. No OpenCV, no
                    models, no personal data needed to test it (see test_faces.py).
  FaceIdentifier  — wraps OpenCV's YuNet detector and SFace recogniser around a matcher.

Enrolled faces live in detector/faces_store.npz, which is GITIGNORED: this repo is
public and those embeddings are personal data about real people.
"""
import json, os, time
from collections import deque

import numpy as np


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


class FaceMatcher:
    """Enrolled identities, and the rules for deciding whether a face is one of them.

    Kept free of OpenCV on purpose: this is the part that makes decisions, so it is the
    part that has to be testable without models, downloads, or photographs of anybody.
    """

    def __init__(self, threshold=0.40, margin=0.10, vote_window_secs=45.0, min_votes=2):
        self.threshold = threshold        # above SFace's own 0.363 — we want headroom
        self.margin = margin              # best must beat the runner-up by this much
        self.vote_window_secs = vote_window_secs
        self.min_votes = min_votes
        self.people = {}                  # name -> (n, dim) float32, unit rows
        self.sightings = deque(maxlen=512)   # (t, name|None, score) for this visit

    # ---- enrolment ----

    def enroll(self, name, embeddings):
        rows = np.stack([_unit(e) for e in embeddings]) if len(embeddings) else None
        if rows is None:
            return
        if name in self.people:
            rows = np.vstack([self.people[name], rows])
        self.people[name] = rows.astype(np.float32)

    def forget(self, name):
        self.people.pop(name, None)

    @staticmethod
    def consistent(embeddings, cut=0.40, rounds=2):
        """Keep only the shots that agree with each other → (keep_idx, drop_idx).

        Enrolment shots are all supposed to be one person, so a shot that does not look
        like the consensus is not a hard pose — it is a mis-detection. Without this,
        enrolling from the 00:32 clip picks up a bright blob alongside eleven real faces,
        and that blob then goes on to match other bright blobs.
        """
        rows = np.stack([_unit(e) for e in embeddings])
        keep = np.arange(len(rows))
        for _ in range(rounds):
            if len(keep) < 3:
                break
            centre = _unit(rows[keep].mean(axis=0))
            sims = rows @ centre
            good = np.array([i for i in keep if sims[i] >= cut])
            if len(good) < 2 or len(good) == len(keep):
                keep = good if len(good) >= 2 else keep
                break
            keep = good
        drop = [i for i in range(len(rows)) if i not in set(keep.tolist())]
        return keep.tolist(), drop

    def names(self):
        return sorted(self.people)

    def count(self, name):
        return int(self.people[name].shape[0]) if name in self.people else 0

    # ---- matching one face ----

    def match(self, embedding):
        """→ (name|None, best_score, margin_over_runner_up).

        Unknown unless the best score clears `threshold` AND beats the best score of
        every *other* person by `margin`. The margin rule is what stops an ambiguous
        face being assigned to whichever enrolled person happens to edge ahead.
        """
        if not self.people:
            return None, 0.0, 0.0
        q = _unit(embedding)
        # rows are unit vectors, so the dot product IS the cosine similarity; take the
        # best of each person's enrolled shots.
        scores = {name: float(np.max(rows @ q)) for name, rows in self.people.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_name, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = best - runner if len(ranked) > 1 else best
        if best < self.threshold:
            return None, best, gap
        if len(ranked) > 1 and gap < self.margin:
            return None, best, gap          # too close to call — say so
        return best_name, best, gap

    # ---- voting across a visit ----

    def observe_match(self, name, score, now=None):
        self.sightings.append((time.time() if now is None else now, name, score))

    def verdict(self, now=None):
        """→ (name|None, votes_for_it, human-readable detail).

        A name wins only with at least `min_votes` sightings inside the window AND a
        strict majority over any rival. Unknown sightings never elect anybody.
        """
        now = time.time() if now is None else now
        fresh = [(t, n, s) for t, n, s in self.sightings
                 if now - t <= self.vote_window_secs]
        if not fresh:
            return None, 0, "no faces seen"
        tally, best_score = {}, {}
        for _, n, s in fresh:
            if n is None:
                continue
            tally[n] = tally.get(n, 0) + 1
            best_score[n] = max(best_score.get(n, 0.0), s)
        unknown = sum(1 for _, n, _ in fresh if n is None)
        if not tally:
            return None, 0, f"{unknown} face(s), none recognised"
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], -best_score[kv[0]]))
        name, votes = ranked[0]
        rival = ranked[1][1] if len(ranked) > 1 else 0
        detail = " ".join(f"{n}x{c}" for n, c in ranked)
        if unknown:
            detail += f" unknown x{unknown}"
        if votes < self.min_votes or votes <= rival:
            return None, votes, f"undecided ({detail})"
        return name, votes, f"{detail} best={best_score[name]:.2f}"

    def reset(self):
        """Start a fresh visit."""
        self.sightings.clear()


class FaceIdentifier:
    """FaceMatcher + OpenCV's YuNet (detect) and SFace (embed), with the person-box gate."""

    def __init__(self, models_dir, store_path, min_face_px=45, det_score=0.7,
                 upscale_to=640, **matcher_kw):
        import cv2
        self.cv2 = cv2
        self.store_path = store_path
        self.min_face_px = min_face_px
        self.upscale_to = upscale_to
        yunet = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        sface = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
        self.available = os.path.exists(yunet) and os.path.exists(sface)
        self.matcher = FaceMatcher(**matcher_kw)
        self.harvest = []                     # [(aligned_crop, embedding, name)] from last observe()
        if not self.available:
            return
        self.det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), det_score, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(sface, "")
        self.load()

    # ---- the store (gitignored: real people's biometrics) ----

    def load(self):
        if not os.path.exists(self.store_path):
            return
        try:
            data = np.load(self.store_path, allow_pickle=False)
            for key in data.files:
                if key.startswith("emb:"):
                    self.matcher.enroll(key[4:], list(data[key]))
        except (OSError, ValueError):
            pass

    def save(self):
        arrays = {f"emb:{n}": rows for n, rows in self.matcher.people.items()}
        tmp = f"{self.store_path}.tmp"
        with open(tmp, "wb") as fh:      # via a handle: savez appends .npz to a str path
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, self.store_path)

    # ---- detection + embedding ----

    def faces_in_crop(self, frame, box):
        """Faces inside ONE person box, upscaled so small faces are findable.
        Returns [(face_box_in_frame_coords, embedding, det_score, face_h_px)]."""
        cv2 = self.cv2
        x1, y1, x2, y2 = box
        pad = int(0.12 * max(1, x2 - x1))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2 = min(frame.shape[1], x2 + pad)
        cy2 = min(frame.shape[0], y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[1] < 16 or crop.shape[0] < 16:
            return []
        scale = max(1.0, min(6.0, self.upscale_to / crop.shape[1]))
        big = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                         interpolation=cv2.INTER_CUBIC)
        self.det.setInputSize((big.shape[1], big.shape[0]))
        _, rows = self.det.detect(big)
        out = []
        for r in (rows if rows is not None else []):
            fh = float(r[3]) / scale
            if fh < self.min_face_px:
                continue
            aligned = self.rec.alignCrop(big, r)
            emb = self.rec.feature(aligned)
            fb = [int(cx1 + r[0] / scale), int(cy1 + r[1] / scale),
                  int(cx1 + (r[0] + r[2]) / scale), int(cy1 + (r[3] + r[1]) / scale)]
            out.append((fb, emb.copy(), float(r[-1]), fh, aligned.copy()))
        return out

    def observe(self, frame, person_boxes, now=None):
        """Look for faces inside already-trusted person boxes and record what we saw.
        Returns [(face_box, name|None, score)] for the monitor. The aligned crop +
        embedding of each face are stashed in self.harvest for the gallery to collect."""
        self.harvest = []                     # [(aligned_crop, embedding, name)]
        if not self.available:
            return []
        now = time.time() if now is None else now
        hits = []
        for box in person_boxes:
            for fb, emb, _det, _fh, aligned in self.faces_in_crop(frame, box):
                name, score, _gap = self.matcher.match(emb)
                self.matcher.observe_match(name, score, now)
                hits.append((fb, name, score))
                self.harvest.append((aligned, emb, name))
        return hits

    def verdict(self, now=None):
        return self.matcher.verdict(now)

    def reset(self):
        self.matcher.reset()

    def describe(self):
        if not self.available:
            return "no models — run detector/get-face-models.sh"
        if not self.matcher.people:
            return "no one enrolled — see detector/enroll.py"
        return ", ".join(f"{n}({self.matcher.count(n)})" for n in self.matcher.names())
