#!/usr/bin/env python3
"""
Tests for the local animal matcher — the thing that names a hedgehog SpeciesNet cannot.

The rules are tested on synthetic vectors (fast, no weights, no crops). The measured
section at the end runs only if detector/animal_refs.npz exists, since that file is this
garden's own labelled crops and is gitignored.

Run:  ../.venv/bin/python detector/test_animal_match.py
"""
import os, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animal_match import AnimalMatcher, NOT_AN_ANIMAL, save_refs

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def vec(*xs):
    v = np.zeros(8, np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    n = np.linalg.norm(v)
    return v / n if n else v


HOG, CAT, TROUGH = vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "refs.npz")
    save_refs(path, [("h1.jpg", "hedgehog", HOG), ("h2.jpg", "hedgehog", vec(1, .12, 0)),
                     ("c1.jpg", "cat", CAT),
                     ("f1.jpg", "furniture", TROUGH),
                     ("e1.jpg", "empty", vec(0, 0, 0, 1)),
                     ("p1.jpg", "person", vec(0, 0, 0, 0, 1))])
    M = AnimalMatcher(path, threshold=0.60, margin=0.05)

    print("1. it names an animal it has references for")
    check("the matcher loaded", M.available, M.describe())
    name, sc, gap = M.match(HOG)
    check("a hedgehog is named", name == "hedgehog", f"{name} {sc:.2f}")
    check("…with the score it matched at", sc > 0.99, f"{sc:.3f}")
    check("a cat is named", M.match(CAT)[0] == "cat")

    print("\n2. a NEGATIVE reference vetoes — this is what keeps the trough quiet")
    # The stone trough is in the references, labelled furniture. A new trough detection
    # matches it almost perfectly, and must produce NO name rather than the runner-up.
    name, sc, _ = M.match(TROUGH)
    check("furniture names nothing", name is None, f"{name} at {sc:.2f}")
    check("…even though it matched confidently", sc > 0.99, f"{sc:.3f}")
    check("empty pavement names nothing", M.match(vec(0, 0, 0, 1))[0] is None)
    check("a person names nothing (never announced as an animal)",
          M.match(vec(0, 0, 0, 0, 1))[0] is None)
    for lab in ("furniture", "empty", "person", "_drop"):
        check(f"{lab!r} is a negative label", lab in NOT_AN_ANIMAL)

    print("\n3. below the threshold it says nothing")
    far = vec(1, 1, 1, 1, 1, 1)                      # cos ~0.41 to hedgehog
    name, sc, _ = M.match(far)
    check("a weak match is not named", name is None, f"{name} at {sc:.2f}")
    check("…and the score is reported anyway, for the log", 0.0 < sc < 0.60, f"{sc:.3f}")

    print("\n4. an ambiguous match is not guessed at")
    tight = AnimalMatcher(path, threshold=0.30, margin=0.20)
    between = vec(1, 1, 0)                           # equidistant hedgehog/cat
    name, sc, gap = tight.match(between)
    check("a crop between two animals is not named", name is None, f"{name} gap={gap:.3f}")
    loose = AnimalMatcher(path, threshold=0.30, margin=0.0)
    check("…but a zero margin would name it", loose.match(between)[0] is not None)

    print("\n5. it fails safe when there is nothing to match against")
    empty_path = os.path.join(d, "none.npz")
    save_refs(empty_path, [])
    E = AnimalMatcher(empty_path)
    check("empty refs -> unavailable", not E.available)
    check("…and matching is quiet rather than an error", E.match(HOG) == (None, 0.0, 0.0))
    check("a missing file -> unavailable", not AnimalMatcher(os.path.join(d, "nope.npz")).available)
    check("a None embedding is quiet", M.match(None) == (None, 0.0, 0.0))

    print("\n6. only-negative references can never name anything")
    neg_path = os.path.join(d, "neg.npz")
    save_refs(neg_path, [("f.jpg", "furniture", TROUGH), ("e.jpg", "empty", vec(0, 0, 0, 1))])
    N = AnimalMatcher(neg_path, threshold=0.30, margin=0.0)
    check("furniture-only refs name nothing", N.match(TROUGH)[0] is None)
    check("…and do not invent a name for an animal either", N.match(HOG)[0] is None)

print("\n7. the measured thresholds, on this garden's own labelled crops")
REFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animal_refs.npz")
if os.path.exists(REFS):
    M = AnimalMatcher(REFS)
    check("the real reference set loads", M.available, M.describe())
    check("it holds hedgehog references", "hedgehog" in M.labels, str(sorted(M.labels)))
    check("…and negative references, or nothing would veto",
          bool(NOT_AN_ANIMAL & set(M.labels)), str(sorted(M.labels)))
    # The default threshold is chosen from the measured cross-visit separation:
    #   same animal, different visit   mean 0.518  p90 0.746  max 0.933
    #   different thing, different visit  mean 0.285  p90 0.414  MAX 0.552
    # 0.60 sits above the whole observed different-thing range. Per-frame cross-visit
    # recall: 0.55 -> 75% but calls a PERSON a raccoon; 0.60 -> 50% and nothing wrong;
    # 0.65 -> 17%; 0.70 -> 0%.
    check("the default threshold clears the measured different-thing max (0.552)",
          M.threshold > 0.552, f"threshold={M.threshold}")
    check("…and is not so high it can never fire (0.70 scored 0%)",
          M.threshold <= 0.65, f"threshold={M.threshold}")
else:
    print("  SKIP  no animal_refs.npz — run `h32 harvest` then `h32 label`")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all animal-matcher checks passed")
