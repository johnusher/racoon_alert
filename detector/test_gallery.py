#!/usr/bin/env python3
"""
Tests for the crop harvester's bookkeeping — the rate-limiting, embedding dedup and
size-bounding that decide WHICH crops get kept. The point of harvesting is a DIVERSE
set to learn from, so near-duplicates and floods are the enemy.

Model-free: synthetic unit vectors stand in for embeddings, tiny arrays for crops.

Run:  ../.venv/bin/python detector/test_gallery.py
"""
import os, sys, tempfile, glob, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gallery import Gallery

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def vec(*p, dim=128):
    v = np.zeros(dim, np.float32)
    for i, x in enumerate(p):
        v[i] = x
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def crop():
    return np.zeros((40, 40, 3), np.uint8)


with tempfile.TemporaryDirectory() as d:
    print("1. rate-limit: no more than one crop of a kind per min_gap")
    g = Gallery(d, min_gap_secs=10, dedup_cos=0.99, max_per_kind=100)
    g.add("face", crop(), {"who": "?"}, embedding=vec(1, 0), now=100.0)
    g.add("face", crop(), {"who": "?"}, embedding=vec(0, 1), now=103.0)   # too soon
    check("second crop within the gap is skipped", g.count("face") == 1,
          f"{g.count('face')}")
    g.add("face", crop(), {"who": "?"}, embedding=vec(0, 1), now=115.0)   # gap passed
    check("a crop after the gap is kept", g.count("face") == 2)

    print("\n2. embedding dedup: near-identical crops are skipped even after the gap")
    g2 = Gallery(d + "/b", min_gap_secs=0, dedup_cos=0.95, max_per_kind=100)
    g2.add("face", crop(), {}, embedding=vec(1, 0, 0), now=1.0)
    g2.add("face", crop(), {}, embedding=vec(1, 0.05, 0), now=2.0)        # ~same person/pose
    check("a near-duplicate embedding is skipped", g2.count("face") == 1,
          f"{g2.count('face')}")
    g2.add("face", crop(), {}, embedding=vec(0, 1, 0), now=3.0)          # clearly different
    check("a genuinely different embedding is kept", g2.count("face") == 2)

    print("\n3. kinds are independent")
    g3 = Gallery(d + "/c", min_gap_secs=10, max_per_kind=100)
    g3.add("face", crop(), {}, embedding=vec(1, 0), now=1.0)
    g3.add("animal", crop(), {}, now=1.0)
    check("a face and an animal at the same instant both save", g3.count("face") == 1 and g3.count("animal") == 1)

    print("\n4. size bound: keeps the newest, prunes the oldest")
    g4 = Gallery(d + "/e", min_gap_secs=0, dedup_cos=1.1, max_per_kind=3)   # dedup off
    for i in range(6):
        g4.add("animal", crop(), {"i": i}, now=float(i))
    check("never exceeds max_per_kind on disk", g4.count("animal") == 3, f"{g4.count('animal')}")
    metas = [m["i"] for m in g4.manifest("animal")]
    check("the survivors are the newest", sorted(metas) == [3, 4, 5], str(sorted(metas)))

    print("\n5. crop files and manifest line up, and reload sees them")
    g5 = Gallery(d + "/f", min_gap_secs=0, dedup_cos=1.1, max_per_kind=100)
    for i in range(4):
        g5.add("animal", crop(), {"label": f"a{i}"}, now=float(i))
    jpgs = glob.glob(os.path.join(d + "/f", "animals", "*.jpg"))
    check("one jpg per kept crop", len(jpgs) == 4, str(len(jpgs)))
    g5b = Gallery(d + "/f", min_gap_secs=0, max_per_kind=100)
    check("a fresh Gallery reloads the existing count", g5b.count("animal") == 4,
          f"{g5b.count('animal')}")

    print("\n6. embeddings round-trip for the clustering step")
    g6 = Gallery(d + "/g", min_gap_secs=0, dedup_cos=1.1, max_per_kind=100)
    g6.add("face", crop(), {"t": 1}, embedding=vec(0.3, 0.7), now=1.0)
    embs, metas = g6.embeddings("face")
    check("embeddings come back as an (n, dim) array", embs.shape == (1, 128), str(embs.shape))
    check("and stay unit-length", abs(np.linalg.norm(embs[0]) - 1.0) < 1e-3)

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all gallery checks passed")
