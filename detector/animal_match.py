#!/usr/bin/env python3
"""
Name an animal SpeciesNet cannot, by matching this garden's own labelled crops.

WHY

SpeciesNet reads the hedgehog that visited at 03:42 on 2026-08-17 as `blank` 0.51-0.96
while `western european hedgehog` scores 0.0001, and no padding, contrast or upscale
moves it (harvest_refs.py has the full measurement). Its 1280-d pooled feature is still
a usable descriptor though, so a hedgehog gets named by looking like the hedgehog crops
labelled in label_animals.py — not by the classifier head.

THE NUMBERS THIS IS BUILT ON

Measured over the 36 labelled crops, cosine between pairs:

    same animal, SAME visit             mean 0.634   p90 0.814   max 0.920
    same animal, DIFFERENT visit        mean 0.518   p90 0.746   max 0.933
    different thing, different visit    mean 0.285   p90 0.414   MAX 0.552

The distributions barely overlap, which is what makes this possible at all. Per-crop,
hiding the crop's whole visit (what the live matcher faces — tonight's animal shares no
frame with any reference):

    threshold 0.55   75% named correctly, but calls a PERSON a raccoon
    threshold 0.60   50% named correctly, nothing wrong          <- default
    threshold 0.65   17% named correctly, nothing wrong
    threshold 0.70    0% named correctly

0.60 sits above the entire observed different-thing range (max 0.552) and is the last
threshold with any real recall. Expect roughly half of visits to be named, not all.

NEGATIVES ARE REFERENCES TOO

The stone trough, the watering can, empty pavement and the household's own heads seen
from above are all labelled and all loaded. If the nearest reference is one of those,
the answer is NO NAME — not the runner-up. That veto, not the threshold, is what stops
the trough being announced as wildlife, and it is the reason labelling the boring
clusters mattered as much as labelling the hedgehog.

⚠️ The hedgehog has been seen ONCE. Everything above is measured on cat, raccoon and
person, which have two visits each; cross-visit behaviour for the hedgehog itself is
UNMEASURED until it comes back. See TODO.md.
"""
import os

import numpy as np

# Labels that exist to say "do not announce this", never to be reported as an animal.
NOT_AN_ANIMAL = frozenset({"furniture", "empty", "person", "_drop"})

THRESHOLD = 0.60      # measured above — 0.55 starts calling a person a raccoon
MARGIN = 0.05         # best label must beat the runner-up by this much


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def save_refs(path, rows):
    """rows: [(file, label, embedding)] → npz. Shared with label_animals.py."""
    np.savez(path,
             files=np.array([r[0] for r in rows], dtype=object),
             labels=np.array([r[1] for r in rows], dtype=object),
             embs=(np.stack([_unit(r[2]) for r in rows]).astype(np.float32)
                   if rows else np.zeros((0, 0), np.float32)))


class AnimalMatcher:
    """Nearest labelled crop, with a threshold, a margin, and a negative veto.

    Deliberately the same shape as faces.py's matcher: everything fails towards "no
    name". A missed hedgehog costs nothing but a generic ANIMAL event; a wrong one puts
    the wrong word in an e-mail subject.
    """

    def __init__(self, path, threshold=THRESHOLD, margin=MARGIN):
        self.path = path
        self.threshold = threshold
        self.margin = margin
        self.by_label = {}
        self._load()

    def _load(self):
        if not (self.path and os.path.exists(self.path)):
            return
        try:
            with np.load(self.path, allow_pickle=True) as z:
                labels, embs = z["labels"].tolist(), z["embs"]
        except (OSError, ValueError, KeyError):
            return
        if not len(labels) or not embs.size:
            return
        for lab, row in zip(labels, embs):
            self.by_label.setdefault(str(lab), []).append(_unit(row))
        self.by_label = {k: np.stack(v) for k, v in self.by_label.items()}

    @property
    def labels(self):
        return list(self.by_label)

    @property
    def available(self):
        """Usable only if something here could actually be named.

        A set of nothing but negatives loads fine and can never return a name, so it is
        not 'available' — that would be a matcher that only ever costs time.
        """
        return bool(set(self.by_label) - NOT_AN_ANIMAL)

    def match(self, embedding):
        """→ (label|None, best_score, margin_over_runner_up).

        None when: nothing to match against, the best score misses `threshold`, two
        labels are within `margin` of each other, or the winner is a negative.
        """
        if embedding is None or not self.by_label:
            return None, 0.0, 0.0
        q = _unit(embedding)
        if q.shape[0] != next(iter(self.by_label.values())).shape[1]:
            return None, 0.0, 0.0            # refs from a different model — say nothing
        # Rows are unit vectors, so the dot product IS the cosine; take each label's best.
        scores = {lab: float(np.max(rows @ q)) for lab, rows in self.by_label.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_label, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else -1.0
        gap = best - runner if len(ranked) > 1 else best
        if best < self.threshold:
            return None, best, gap
        if len(ranked) > 1 and gap < self.margin:
            return None, best, gap           # too close to call — say so
        if best_label in NOT_AN_ANIMAL:
            return None, best, gap           # the veto: known furniture/pavement/person
        return best_label, best, gap

    def describe(self):
        if not self.by_label:
            return "no references — `h32 harvest` then `h32 label`"
        counts = ", ".join(f"{lab} {len(rows)}"
                           for lab, rows in sorted(self.by_label.items(),
                                                   key=lambda kv: -len(kv[1])))
        return (f"{sum(len(r) for r in self.by_label.values())} labelled crop(s) "
                f"[{counts}], name at cos>={self.threshold}")


def main():
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    m = AnimalMatcher(os.path.join(base, "animal_refs.npz"))
    print(__doc__)
    print(f"status: {m.describe()}")
    if not m.available:
        sys.exit(1)


if __name__ == "__main__":
    main()
