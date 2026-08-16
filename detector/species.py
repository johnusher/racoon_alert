#!/usr/bin/env python3
"""
Species classifier — splits MegaDetector's generic "animal" into cat / raccoon / fox / …

⚠️ SUPERSEDED 2026-08-16 by speciesnet.py, and no longer wired into detect.py. The
"known limit" warned about at the bottom of this docstring turned out to be the whole
story: measured on the real archive this classifier keyed on LIGHTING, not species —
an empty patch of night pavement scored raccoon 0.919, higher than the actual cat did,
and a real human at night scored raccoon 0.843. Its "100% leave-one-out" was measuring
night-vs-day. Kept for the SpeciesMatcher unit (still tested) and the reference-set CLI;
read speciesnet.py before reviving any of it.

MegaDetector only says animal/person/vehicle. To tell John's black cat from a raccoon we
classify the animal CROP. CLIP *zero-shot* was measured useless here — on this camera's
night-IR crops the text-matching softmax is flat (~1/N, no signal). But CLIP's image
EMBEDDINGS separate the species cleanly (raccoon vs cat: 100% leave-one-out on the first
labelled clips, inter-class cosine 0.76 vs intra-class 0.92). So this is nearest-reference
on CLIP image embeddings, trained on labelled example crops — not zero-shot.

  SpeciesMatcher   — pure numpy: labelled reference embeddings, nearest-reference vote,
                     unknown when nothing is close enough or two species tie. Testable
                     without CLIP (see test_species.py).
  SpeciesClassifier— wraps a CLIP image encoder around the matcher.

References live in detector/species_refs.npz (gitignored — it is training data, and grows).
Bootstrap + grow it with:  detector/species.py train <label> <clip-or-image>...

⚠️ Known limit while data is thin: the first raccoons are night-IR and the first cat is
daylight, so the classifier partly keys on lighting. It improves as the harvester collects
crops across conditions and they get labelled.
"""
import os, sys, glob, json, time

import numpy as np


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


class SpeciesMatcher:
    """Labelled reference embeddings + the rule for naming a new crop. Pure numpy."""

    def __init__(self, min_sim=0.55, margin=0.04):
        self.min_sim = min_sim      # below this the animal is "other" (fox/dog/unseen)
        self.margin = margin        # top species must beat the runner-up by this
        self.refs = {}              # label -> (n, dim) unit rows

    def add(self, label, embeddings):
        rows = np.stack([_unit(e) for e in embeddings]) if len(embeddings) else None
        if rows is None:
            return
        if label in self.refs:
            rows = np.vstack([self.refs[label], rows])
        self.refs[label] = rows.astype(np.float32)

    def labels(self):
        return sorted(self.refs)

    def count(self, label):
        return int(self.refs[label].shape[0]) if label in self.refs else 0

    def classify(self, embedding):
        """→ (label|None, best_similarity, margin). None = 'other/unknown'."""
        if not self.refs:
            return None, 0.0, 0.0
        q = _unit(embedding)
        best = {lab: float(np.max(rows @ q)) for lab, rows in self.refs.items()}
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        lab, sim = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = sim - runner if len(ranked) > 1 else sim
        if sim < self.min_sim:
            return None, sim, gap
        if len(ranked) > 1 and gap < self.margin:
            return None, sim, gap
        return lab, sim, gap

    # ---- persistence ----

    def save(self, path):
        arrays = {f"ref:{lab}": rows for lab, rows in self.refs.items()}
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)

    def load(self, path):
        if not os.path.exists(path):
            return
        try:
            data = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return
        for k in data.files:
            if k.startswith("ref:"):
                self.refs[k[4:]] = data[k].astype(np.float32)


class SpeciesClassifier:
    """CLIP image embeddings + a SpeciesMatcher. Lazy CLIP load (only when needed)."""

    def __init__(self, refs_path, min_sim=0.55, margin=0.04,
                 clip_model="ViT-B-32", clip_pretrained="laion2b_s34b_b79k"):
        self.refs_path = refs_path
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.matcher = SpeciesMatcher(min_sim, margin)
        self.matcher.load(refs_path)
        self._clip = None

    @property
    def available(self):
        try:
            import open_clip  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_clip(self):
        if self._clip is not None:
            return
        import torch, open_clip
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model, _, pre = open_clip.create_model_and_transforms(
            self.clip_model, pretrained=self.clip_pretrained)
        self._clip = (model.to(dev).eval(), pre, dev, torch)

    def embed(self, crop_bgr):
        """CLIP image embedding of a BGR crop (unit vector)."""
        import cv2
        from PIL import Image
        self._load_clip()
        model, pre, dev, torch = self._clip
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = pre(Image.fromarray(rgb)).unsqueeze(0).to(dev)
        with torch.no_grad():
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy()

    def classify(self, crop_bgr):
        """→ (label|None, similarity). None = 'other/unknown' animal."""
        if not self.matcher.refs:
            return None, 0.0
        lab, sim, _gap = self.matcher.classify(self.embed(crop_bgr))
        return lab, sim

    def train(self, label, crops):
        self.matcher.add(label, [self.embed(c) for c in crops])
        self.matcher.save(self.refs_path)

    def describe(self):
        if not self.matcher.refs:
            return "untrained — detector/species.py train <label> <clips>"
        return ", ".join(f"{lab}({self.matcher.count(lab)})" for lab in self.matcher.labels())


# ---- CLI: bootstrap / grow the reference set from labelled clips or stills ----

def _animal_crops(path, max_n=40, every_s=0.3):
    """Yield MegaDetector animal crops from a clip or still, CLAHE-enhanced like detect.py."""
    import cv2, torch
    from ultralytics import YOLO
    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(base)); sys.path.insert(0, base)
    import h32env
    cfg = h32env.detector_config(os.path.join(base, "config.json"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    md = YOLO(os.path.join(base, cfg["model"]))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enh(f):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def boxes(frame):
        r = md(enh(frame), imgsz=cfg["imgsz"], conf=0.25, device=dev, verbose=False)[0]
        for b in r.boxes:
            if md.names[int(b.cls)] == "animal":
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                c = frame[max(0, y1):y2, max(0, x1):x2]
                if c.size and c.shape[1] >= 20:
                    yield c

    out = []
    if path.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(path)
        if img is not None:
            out.extend(list(boxes(img))[:1])
        return out
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    step = max(1, int(fps * every_s)); i = 0
    while len(out) < max_n:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            for c in boxes(f):
                out.append(c); break
        i += 1
    cap.release()
    return out


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    refs = os.path.join(base, "species_refs.npz")
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        print(SpeciesClassifier(refs).describe()); return
    if len(sys.argv) >= 4 and sys.argv[1] == "train":
        label, sources = sys.argv[2], sys.argv[3:]
        sc = SpeciesClassifier(refs)
        if not sc.available:
            sys.exit("open_clip not installed — ../.venv/bin/pip install open_clip_torch")
        crops = []
        for s in sources:
            got = _animal_crops(s)
            print(f"  {s}: {len(got)} animal crops")
            crops += got
        if not crops:
            sys.exit("  no animal crops found — nothing trained")
        sc.train(label, crops)
        print(f"trained '{label}' from {len(crops)} crops -> now: {sc.describe()}")
        return
    print(__doc__)
    print("usage: species.py train <label> <clip|image>...   |   species.py list")


if __name__ == "__main__":
    main()
