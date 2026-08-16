#!/usr/bin/env python3
"""
SpeciesNet — the second opinion on what MegaDetector just found.

MegaDetector only says animal/person/vehicle, and on this camera it says them *wrongly*
in the dark. On 2026-08-16 21:18 a black cat sitting in the bushes at the right edge of
the frame scored `person 0.74`, sent a PERSON e-mail and made the camera say "Hallo." to
a cat. Measured over that clip MegaDetector called it a person in 23 sampled frames and
an animal in 3 — the confusion is the detector's own, not a preprocessing artefact
(CLAHE off, imgsz 1920 and test-time augmentation were all tried; none helped).

So the class MegaDetector puts on a box cannot be the last word. This module runs
Google's SpeciesNet classifier (EfficientNetV2-M, 65M camera-trap images, ~2000 classes,
Apache-2.0) over the CROP and answers the question that actually matters:

    is this a human — yes, no, or not sure?

...and, when the crop is good enough, names the species outright.

--- why this replaced the CLIP reference matcher (see species.py) --------------------

species.py classified the crop by nearest neighbour over a handful of labelled CLIP
embeddings. Its docstring warned it "partly keys on lighting". Measured on the real
archive, it keyed on *nothing else*:

    an empty patch of night pavement  -> raccoon 0.919     <- higher than the actual cat
    an empty night bush               -> raccoon 0.891
    a real human at night             -> raccoon 0.843
    the 21:18 cat, all 29 crops       -> raccoon 0.89–0.94
    daylight patches                  -> cat, or unknown

The 12 references were 7 night-IR raccoons and 5 daylight cats, so CLIP separated
illumination and the species labels came along for the ride. Its "100% leave-one-out"
was measuring night-vs-day. SpeciesNet has no such problem on the same crops:

    empty night pavement -> blank 0.983      real human at night -> human 0.975
    empty night bush     -> blank 0.947      the 03:53 raccoon   -> northern raccoon 0.978

--- the numbers the thresholds come from ---------------------------------------------

P(human) over the archive, on crops MegaDetector had labelled `person`:

    the 21:18 cat        n=29    max 0.0855   mean 0.0088   min 0.0017
    real people          n=9     min 0.4909   mean 0.8337   max 0.9990

A 5.7x gap, so any veto in 0.10–0.45 separates them perfectly; `human_veto` sits in the
middle at 0.25. The rule is deliberately one-sided — it only ever says "this is NOT a
person". Anything in between leaves the person event alone, so an unsure classifier
costs a false alarm, never a missed one.

`species_min` is 0.50: the good raccoon crop clears it at 0.978, while the poor crops
from 20260816_035358 land at 0.28–0.34 and stay a generic ANIMAL rather than a
confident wrong answer.

--- loading ---------------------------------------------------------------------------

The `speciesnet` PyPI package needs Python <3.14 and pulls the old yolov5 that this
project already refuses (see requirements.txt), so — exactly as with MegaDetector's
weights through ultralytics — we load the released weights directly. They are a
torch.fx GraphModule converted from ONNX, so `onnx2torch` must be importable for the
unpickle to resolve, but is never called by us.

Weights (~225MB, gitignored):  detector/get-speciesnet.sh
"""
import os

import numpy as np

IMG_SIZE = 480
# Labels that are real classes but not a species we would ever want to announce.
NOT_A_SPECIES = frozenset({"blank", "vehicle", "human", "animal"})

# SpeciesNet speaks in common names ("northern raccoon"); events, filenames and the
# email trigger_on list want one short word. The last word is right far more often than
# not — but not always, and "raccoon dog" is a very different visitor from a dog, so the
# ones we actually expect in this garden are spelled out.
SHORT_NAMES = {
    "northern raccoon": "raccoon", "crab-eating raccoon": "raccoon",
    "raccoon dog": "raccoondog",
    "domestic cat": "cat", "domestic dog": "dog",
    "red fox": "fox", "european hedgehog": "hedgehog",
    "european badger": "badger", "european rabbit": "rabbit",
    "eurasian red squirrel": "squirrel", "beech marten": "marten",
    "european pine marten": "marten",
}


def short_name(label):
    """'northern raccoon' -> 'raccoon'. Safe for a filename and for trigger_on."""
    if not label:
        return None
    key = label.strip().lower()
    if key in SHORT_NAMES:
        return SHORT_NAMES[key]
    word = key.replace("-", " ").split()[-1] if key.split() else key
    return "".join(ch for ch in word if ch.isalnum()) or None


class Verdict:
    """What SpeciesNet thinks a single crop is.

    is_human / not_human are deliberately NOT complements: between the two thresholds
    the classifier has no opinion, and the caller must keep whatever it already believed.
    """

    __slots__ = ("human_p", "blank_p", "top_label", "top_p", "species", "is_human", "not_human")

    def __init__(self, human_p, blank_p, top_label, top_p, species, is_human, not_human):
        self.human_p, self.blank_p = human_p, blank_p
        self.top_label, self.top_p = top_label, top_p
        self.species = species
        self.is_human, self.not_human = is_human, not_human

    @property
    def tag(self):
        """The one-word event tag for this species — CAT, RACCOON — or None."""
        s = short_name(self.species)
        return s.upper() if s else None

    @property
    def identified(self):
        """Did SpeciesNet positively recognise something worth alerting about?

        This is the promote side of the veto, and the answer to a failure the movement
        gate cannot fix on its own: a person who walks out and STANDS STILL never
        displaces their box, so scenery.py holds them back as unproven for ever (see
        the 22:48 friend in test_scenery.py). Furniture cannot sneak in this way — the
        bench, the rock, the plant pot and the orange bucket all read `blank` 0.92-1.00,
        which is neither a human nor a named species.
        """
        return bool(self.is_human or self.species)

    def __repr__(self):
        return (f"<Verdict {self.top_label}={self.top_p:.3f} human={self.human_p:.3f}"
                f"{' NOT-HUMAN' if self.not_human else ''}"
                f"{' HUMAN' if self.is_human else ''}>")

    def describe(self):
        """One line for events.log — the whole basis of the decision, so a wrong call
        is diagnosable after the fact instead of being a silent re-tag."""
        top = f"speciesnet {self.top_label}={self.top_p:.2f}"
        if self.top_label != "human":                    # else it reads "human=1.00 human=1.00"
            top += f" human={self.human_p:.2f}"
        return top + (f" -> {self.species}" if self.species else "")


class SpeciesRules:
    """The decision layer, kept free of torch so it can be tested on plain vectors."""

    def __init__(self, labels, human_veto=0.25, human_min=0.45, species_min=0.50):
        self.labels = list(labels)
        self.human_veto = human_veto      # below this P(human): definitely not a person
        self.human_min = human_min        # at or above this: definitely a person
        self.species_min = species_min    # name a species only this confidently
        self._i_human = self.labels.index("human") if "human" in self.labels else -1
        self._i_blank = self.labels.index("blank") if "blank" in self.labels else -1

    def verdict(self, probs):
        probs = np.asarray(probs, np.float32).ravel()
        human_p = float(probs[self._i_human]) if self._i_human >= 0 else 0.0
        blank_p = float(probs[self._i_blank]) if self._i_blank >= 0 else 0.0
        top = int(np.argmax(probs))
        top_label, top_p = self.labels[top], float(probs[top])
        species = top_label if (top_p >= self.species_min
                                and top_label not in NOT_A_SPECIES) else None
        # With no 'human' class we cannot judge humanity at all — say nothing rather
        # than veto everything, because a veto suppresses a person alert.
        known = self._i_human >= 0
        return Verdict(human_p, blank_p, top_label, top_p, species,
                       is_human=known and human_p >= self.human_min,
                       not_human=known and human_p < self.human_veto)


def load_labels(path):
    """SpeciesNet's label file: UUID;class;order;family;genus;species;common_name."""
    out = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(";")
            if len(parts) >= 7:
                out.append(parts[6] or ";".join(p for p in parts[1:6] if p))
    return out


class SpeciesNetClassifier:
    """SpeciesNet over a BGR crop. Torch is loaded lazily, on first use."""

    def __init__(self, model_path, labels_path, human_veto=0.25, human_min=0.45,
                 species_min=0.50):
        self.model_path, self.labels_path = model_path, labels_path
        self.labels = load_labels(labels_path) if os.path.exists(labels_path) else []
        self.rules = SpeciesRules(self.labels, human_veto, human_min, species_min) \
            if self.labels else None
        self._model = None

    @property
    def available(self):
        if not (os.path.exists(self.model_path) and self.labels and self.rules):
            return False
        try:
            import torch, onnx2torch  # noqa: F401   (onnx2torch: needed to unpickle)
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is not None:
            return
        import torch
        import onnx2torch  # noqa: F401   the pickle references its op classes
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = torch.load(self.model_path, map_location=dev, weights_only=False)
        self._model = (model.eval(), dev, torch)

    def probs(self, crop_bgr):
        """Softmax over all classes for one BGR crop.

        Preprocessing follows SpeciesNet's published recipe exactly, including the
        float->resize->uint8 round trip and antialias=False: it is not the obvious
        pipeline, and getting it wrong quietly costs accuracy rather than erroring.
        """
        import cv2
        import torchvision.transforms.functional as TF
        from PIL import Image
        self._load()
        model, dev, torch = self._model
        t = TF.pil_to_tensor(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
        t = TF.convert_image_dtype(t, torch.float32)
        t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=False)
        t = TF.convert_image_dtype(t, torch.uint8)
        arr = t.permute(1, 2, 0).numpy().astype("float32") / 255.0   # NHWC, [0,1]
        x = torch.from_numpy(arr).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return torch.softmax(out, dim=1)[0].cpu().numpy()

    def classify(self, crop_bgr):
        """→ Verdict, or None if the crop is empty."""
        if crop_bgr is None or not getattr(crop_bgr, "size", 0):
            return None
        return self.rules.verdict(self.probs(crop_bgr))

    def describe(self):
        if not os.path.exists(self.model_path):
            return "not installed — detector/get-speciesnet.sh"
        if not self.available:
            return "weights present, torch/onnx2torch missing — pip install -r requirements.txt"
        return (f"{len(self.labels)} classes, veto P(human)<{self.rules.human_veto}, "
                f"name species at ≥{self.rules.species_min}")


# ---- CLI: ask it about a clip or a still ----------------------------------------------

def main():
    import sys
    import cv2
    base = os.path.dirname(os.path.abspath(__file__))
    sn = SpeciesNetClassifier(os.path.join(base, "models", "speciesnet_crop_4.0.1a.pt"),
                              os.path.join(base, "models", "speciesnet_labels.txt"))
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"status: {sn.describe()}")
        print("usage: speciesnet.py <image-or-clip>...   (runs MegaDetector, then classifies each crop)")
        return
    if not sn.available:
        sys.exit(f"speciesnet unavailable: {sn.describe()}")

    import torch
    from ultralytics import YOLO
    sys.path.insert(0, os.path.dirname(base)); sys.path.insert(0, base)
    import h32env
    cfg = h32env.detector_config(os.path.join(base, "config.json"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    md = YOLO(os.path.join(base, cfg["model"]))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def enh(f):
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB); lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def report(frame, where):
        r = md(enh(frame), imgsz=cfg["imgsz"], conf=0.15, device=dev, verbose=False)[0]
        for b in r.boxes:
            cls = md.names[int(b.cls)]
            if cls == "vehicle":
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            v = sn.classify(crop)
            if v is None:
                continue
            verdict = "NOT a person" if (cls == "person" and v.not_human) else ""
            print(f"  {where:>9}  megadetector {cls}:{float(b.conf):.2f}  ->  "
                  f"{v.describe()}  {verdict}")

    for path in sys.argv[1:]:
        print(f"\n{path}")
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = cv2.imread(path)
            if img is not None:
                report(img, "still")
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % max(1, int(fps)) == 0:                  # ~1 fps
                report(f, f"{i/fps:5.1f}s")
            i += 1
        cap.release()


if __name__ == "__main__":
    main()
