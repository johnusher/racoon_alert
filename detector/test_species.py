#!/usr/bin/env python3
"""
Tests for the species MATCHER — nearest-reference vote, unknown when nothing is close or
two species tie. Model-free (synthetic unit vectors), so no CLIP or images needed.

Run:  ../.venv/bin/python detector/test_species.py
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from species import SpeciesMatcher

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def vec(*p, dim=64):
    v = np.zeros(dim, np.float32)
    for i, x in enumerate(p):
        v[i] = x
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


RACCOON, CAT, FOX = vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)

print("1. a trained species is recognised")
m = SpeciesMatcher(min_sim=0.55, margin=0.04)
check("empty matcher says unknown", m.classify(RACCOON)[0] is None)
m.add("raccoon", [RACCOON, vec(0.97, 0.1, 0)])
m.add("cat", [CAT, vec(0.1, 0.97, 0)])
check("a raccoon-like crop -> raccoon", m.classify(vec(0.98, 0.05, 0))[0] == "raccoon")
check("a cat-like crop -> cat", m.classify(vec(0.05, 0.98, 0))[0] == "cat")

print("\n2. an unseen animal is 'other', not forced into a class")
check("a fox (orthogonal) is unknown", m.classify(FOX)[0] is None,
      f"sim={m.classify(FOX)[1]:.2f}")

print("\n3. a crop between two species is unknown, not a coin-flip")
between = vec(0.7, 0.7, 0)
check("raccoon/cat midpoint -> unknown", m.classify(between)[0] is None,
      f"{m.classify(between)}")

print("\n4. more reference shots sharpen a class")
m.add("raccoon", [vec(0.9, 0.2, 0.1), vec(0.95, 0.0, 0.2)])
check("raccoon count grew", m.count("raccoon") == 4, str(m.count("raccoon")))
check("a noisy raccoon still lands", m.classify(vec(0.9, 0.15, 0.15))[0] == "raccoon")

print("\n5. references persist across a reload")
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "refs.npz")
    m.save(p)
    m2 = SpeciesMatcher(min_sim=0.55, margin=0.04)
    m2.load(p)
    check("reloaded matcher keeps the labels", set(m2.labels()) == {"raccoon", "cat"})
    check("and still classifies", m2.classify(vec(0.98, 0.05, 0))[0] == "raccoon")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all species-matcher checks passed")
