#!/usr/bin/env python3
"""
Tests for the animal-crop clustering — the step that turns a pile of harvested crops
into a handful of things to name.

Pure numpy on synthetic vectors, so this runs without the 225MB SpeciesNet weights and
without any crops on disk.

Run:  ../.venv/bin/python detector/test_label_animals.py
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_animals import cluster, unit, MIN_COS

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def blob(centre, n, spread, rng, dim=16):
    """n unit vectors scattered around one direction."""
    c = np.zeros(dim, np.float32)
    c[centre] = 1.0
    out = [unit(c + rng.normal(0, spread, dim).astype(np.float32)) for _ in range(n)]
    return np.stack(out)


rng = np.random.default_rng(7)

print("1. well-separated groups come back as separate clusters")
X = np.concatenate([blob(0, 6, 0.10, rng), blob(5, 5, 0.10, rng), blob(11, 4, 0.10, rng)])
lab = cluster(X, min_cos=0.80)
check("three blobs -> three clusters", len(set(lab.tolist())) == 3,
      f"{len(set(lab.tolist()))} clusters")
check("every crop is assigned", len(lab) == len(X) and lab.min() >= 0)
# Membership, not label numbering: the blobs must not be split or mixed.
groups = [set(np.flatnonzero(lab == k).tolist()) for k in sorted(set(lab.tolist()))]
check("the groups match the blobs",
      sorted(sorted(g) for g in groups) == [list(range(0, 6)), list(range(6, 11)),
                                            list(range(11, 15))],
      str(sorted(sorted(g) for g in groups)))

print("\n2. clusters are ordered biggest first, so the common visitor is named first")
check("cluster 0 is the largest", (lab == 0).sum() == 6, f"{(lab == 0).sum()}")
check("…and the smallest is last", (lab == lab.max()).sum() == 4, f"{(lab == lab.max()).sum()}")

print("\n3. the threshold controls how fussy it is")
check("a loose threshold merges everything", len(set(cluster(X, min_cos=-1.0).tolist())) == 1)
check("a strict threshold splits everything",
      len(set(cluster(X, min_cos=0.999).tolist())) == len(X))

print("\n4. identical crops collapse; unrelated ones do not")
same = np.stack([unit(np.array([1, 0, 0, 0], np.float32))] * 4)
check("four identical vectors are one cluster", len(set(cluster(same, 0.9).tolist())) == 1)
orth = np.eye(4, dtype=np.float32)
check("four orthogonal vectors stay four clusters",
      len(set(cluster(orth, 0.5).tolist())) == 4)

print("\n5. degenerate input does not explode")
check("no crops -> no labels", cluster(np.zeros((0, 8), np.float32), 0.8).shape == (0,))
check("one crop -> one cluster", cluster(np.ones((1, 8), np.float32), 0.8).tolist() == [0])

print("\n6. the default threshold is the measured one, not a face-recognition intuition")
# On the 37 crops harvested 2026-08-17, cos>=0.80 shattered ONE hedgehog into 8 clusters
# (biggest 3/13) while 0.55 grouped 12/13 into a single pure cluster. These are 80-142px
# night-IR crops; within-animal cosine runs 0.24-0.93. If this ever gets raised back
# towards a face-like 0.8, re-measure on real crops first — see cluster()'s docstring.
check("default groups loosely enough for night-IR crops", MIN_COS <= 0.60, str(MIN_COS))
check("…but not so loosely it merges everything", MIN_COS >= 0.45, str(MIN_COS))
# A spread as wide as the real hedgehog's must still land in one cluster at the default.
wide = np.stack([unit(np.array([1.0, 0.05, 0.0, 0.0], np.float32)),
                 unit(np.array([1.0, 0.60, 0.0, 0.0], np.float32)),
                 unit(np.array([1.0, 1.05, 0.0, 0.0], np.float32))])
check("a wide-but-related spread stays one cluster at the default",
      len(set(cluster(wide, MIN_COS).tolist())) == 1,
      f"{len(set(cluster(wide, MIN_COS).tolist()))} clusters")

print("\n7. unit() is the normalisation the cosine assumes")
v = unit(np.array([3.0, 4.0], np.float32))
check("a vector is normalised", abs(float(np.linalg.norm(v)) - 1.0) < 1e-6, str(v))
check("a zero vector stays finite rather than dividing by zero",
      bool(np.all(np.isfinite(unit(np.zeros(4, np.float32))))))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all animal-clustering checks passed")
