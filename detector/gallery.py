#!/usr/bin/env python3
"""
Crop harvester — quietly collects examples the detector can learn from later.

Every confirmed detection can drop a crop here: a face crop for each person, an animal
crop for each cat/raccoon/fox. Those crops become the raw material for two things:
  • learning the household people — cluster the face embeddings, label the clusters
    (cluster_faces.py), enrol them (this is how the 2- and 6-year-old get learned, since
    a single deliberate enrolment of a small child rarely sticks);
  • training a real species classifier for this camera's night-IR look, once enough
    animal crops are labelled.

The whole point is a DIVERSE set, so the harvester fights three things: floods (a rate
limit per kind), near-duplicates (skip a crop whose embedding is ~identical to a recent
one), and unbounded growth (keep only the newest max_per_kind).

Everything here is REAL PEOPLE'S BIOMETRICS, including children — the gallery dir is
gitignored and stays local. This is a public repo.

Layout:  <dir>/<kind>s/<ts>_<seq>.jpg  +  <dir>/<kind>s/manifest.jsonl
Each manifest line: {"file","t", ...meta, "emb":[...] (if given)}.
"""
import json, os, time
from collections import deque

import numpy as np


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


class Gallery:
    def __init__(self, path, min_gap_secs=20.0, dedup_cos=0.94, max_per_kind=800,
                 recent_window=40):
        self.path = path
        self.min_gap_secs = min_gap_secs
        self.dedup_cos = dedup_cos            # skip a crop this similar to a recent one
        self.max_per_kind = max_per_kind
        self.recent_window = recent_window
        self._last_add = {}                   # kind -> last save time
        self._recent = {}                     # kind -> deque of recent embeddings
        os.makedirs(path, exist_ok=True)

    # ---- paths / io ----

    def _dir(self, kind):
        d = os.path.join(self.path, f"{kind}s")
        os.makedirs(d, exist_ok=True)
        return d

    def _manifest_path(self, kind):
        return os.path.join(self._dir(kind), "manifest.jsonl")

    def manifest(self, kind):
        p = self._manifest_path(kind)
        if not os.path.exists(p):
            return []
        out = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        return out

    def count(self, kind):
        return len(self.manifest(kind))

    def embeddings(self, kind):
        """→ (embeddings (n,dim) float32 unit rows, metas) for the entries that have one."""
        rows, metas = [], []
        for m in self.manifest(kind):
            if "emb" in m and m["emb"]:
                rows.append(_unit(m["emb"]))
                metas.append(m)
        if not rows:
            return np.zeros((0, 0), np.float32), []
        return np.stack(rows).astype(np.float32), metas

    # ---- adding ----

    def _too_similar(self, kind, emb):
        if emb is None or self.dedup_cos > 1.0:
            return False
        recent = self._recent.setdefault(kind, deque(maxlen=self.recent_window))
        u = _unit(emb)
        for r in recent:
            if float(np.dot(u, r)) >= self.dedup_cos:
                return True
        return False

    def ready(self, kind, now=None):
        """Would add() get past the rate limit right now?

        Lets a caller skip work it would only throw away. The animal harvest runs at the
        detection rate but saves at most one crop per min_gap_secs, while the embedding
        that makes a crop worth keeping costs a ~140ms SpeciesNet pass — so detect.py
        asks this first and pays that at the SAVE rate rather than at 3 fps.

        Only the rate limit; the near-duplicate check needs the embedding to answer.
        """
        now = time.time() if now is None else now
        return now - self._last_add.get(kind, -1e9) >= self.min_gap_secs

    def add(self, kind, crop_bgr, meta=None, embedding=None, now=None):
        """Save one crop if it passes the rate limit and the dedup check. Returns the
        saved filename, or None if skipped. `crop_bgr` is a BGR uint8 image."""
        now = time.time() if now is None else now
        if now - self._last_add.get(kind, -1e9) < self.min_gap_secs:
            return None
        if self._too_similar(kind, embedding):
            return None

        import cv2
        d = self._dir(kind)
        seq = self.count(kind)
        fname = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{seq:05d}.jpg"
        try:
            cv2.imwrite(os.path.join(d, fname), crop_bgr)
        except Exception:
            return None

        entry = {"file": fname, "t": round(now, 2)}
        if meta:
            entry.update(meta)
        if embedding is not None:
            entry["emb"] = [round(float(x), 5) for x in _unit(embedding).tolist()]
        with open(self._manifest_path(kind), "a") as fh:
            fh.write(json.dumps(entry) + "\n")

        self._last_add[kind] = now
        if embedding is not None:
            self._recent.setdefault(kind, deque(maxlen=self.recent_window)).append(_unit(embedding))
        self._prune(kind)
        return fname

    def _prune(self, kind):
        """Keep only the newest max_per_kind entries; delete the rest, jpg + manifest line."""
        entries = self.manifest(kind)
        if len(entries) <= self.max_per_kind:
            return
        keep = entries[-self.max_per_kind:]
        drop = entries[:-self.max_per_kind]
        d = self._dir(kind)
        for m in drop:
            try:
                os.remove(os.path.join(d, m["file"]))
            except OSError:
                pass
        tmp = self._manifest_path(kind) + ".tmp"
        with open(tmp, "w") as fh:
            for m in keep:
                fh.write(json.dumps(m) + "\n")
        os.replace(tmp, self._manifest_path(kind))

    def describe(self):
        kinds = []
        for kind in ("face", "animal"):
            n = self.count(kind)
            if n:
                kinds.append(f"{n} {kind}s")
        return ", ".join(kinds) if kinds else "empty"
