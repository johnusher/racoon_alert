#!/usr/bin/env python3
"""
Cluster the harvested animal crops, show them all on one sheet, and name them.

WHY

SpeciesNet names the raccoon and the cat. It cannot name the hedgehog that walked through
at 03:42 on 2026-08-17 — `western european hedgehog` 0.0001 against `blank` 0.9, and no
padding, enhancement or upscale moves it (see harvest_refs.py for the full measurement).
For animals like that, recognition has to come from crops labelled here: the 1280-d
SpeciesNet feature banked with each crop is the thing a later matcher compares against.

This is the animal half of the cluster-and-label step in TODO.md. It does not decide
anything at detection time — it only turns a pile of crops into named references.

HOW

Crops with an embedding are grouped by average-linkage cosine, biggest cluster first.
EVERY cluster goes onto ONE labelled sheet, which is opened once (see overview_sheet for
why one file rather than one per cluster), then you are prompted per cluster:

    <name>    label the whole cluster
    s         skip it, decide later
    d         drop it — furniture, an empty frame, a false positive
    q         stop here and write what has been named so far

Or skip the prompting entirely and label clusters by number with --set.

Labels are written to detector/animal_refs.npz (embeddings + label per crop), so
re-running picks up where you left off rather than asking about crops already named.

Usage:
    h32 label                                 # label what is unlabelled
    h32 label --set 1=hedgehog,5=_drop        # label by cluster number, no prompting
    h32 label --min-cos 0.70                  # fussier grouping (default 0.55)
    h32 label --review                        # show what is already labelled and stop
"""
import argparse
import json
import os
import subprocess
import sys

try:
    import numpy as np
except ModuleNotFoundError:                  # the shebang finds system python, which has
    sys.exit("this needs the project venv — run it as:  h32 label\n"      # none of our deps
             "                       or:  ./.venv/bin/python detector/label_animals.py")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

REFS = os.path.join(BASE, "animal_refs.npz")
SHEET = os.path.join(BASE, "gallery", "_clusters.jpg")   # ONE sheet — see overview_sheet


def unit(v):
    """Row-normalise for cosine. A zero vector stays zero rather than becoming NaN."""
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


MIN_COS = 0.55        # measured below — do not raise it on intuition


def cluster(embs, min_cos=MIN_COS):
    """Average-linkage agglomerative clustering on cosine similarity.

    → (n,) int labels, renumbered so cluster 0 is the largest: the animal that visits
    most gets named first, and a one-off sits at the end where it is easy to skip.

    Average linkage rather than centroid because a cluster of near-duplicate crops (the
    harvester keeps poses that differ by cos<0.94) should not be dragged into its
    neighbour by a single outlying frame. n is in the hundreds at most, so the naive
    O(n^3) merge loop is far simpler than it is slow, and needs no new dependency.

    min_cos 0.55 is measured, not guessed. On the 37 harvested crops of 2026-08-17,
    grouping the 13 hedgehog crops from 20260817_034249:

        cos>=0.80   8 clusters for one animal, biggest holds  3/13   (unusable)
        cos>=0.70   4 clusters,                biggest holds  7/13
        cos>=0.60   3 clusters,                biggest holds  7/13
        cos>=0.55   2 clusters,                biggest holds 12/13   <- and pure
        cos>=0.50   same 12/13, but starts merging unrelated events elsewhere

    These crops are 80-142px of night IR, so within-hedgehog cosine runs 0.24-0.93 —
    a threshold tuned for clean daylight faces shatters one animal into eight piles.
    """
    embs = unit(np.asarray(embs, np.float32))
    n = len(embs)
    if n == 0:
        return np.zeros(0, np.int32)
    if n == 1:
        return np.zeros(1, np.int32)
    sim = embs @ embs.T
    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        best, pair = -2.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = float(sim[np.ix_(groups[a], groups[b])].mean())
                if s > best:
                    best, pair = s, (a, b)
        if best < min_cos:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        groups.pop(b)
    groups.sort(key=len, reverse=True)
    lab = np.zeros(n, np.int32)
    for k, g in enumerate(groups):
        for i in g:
            lab[i] = k
    return lab


def overview_sheet(blocks, out, cols=6, tile=110):
    """Every cluster on ONE labelled image — the thing a person actually looks at.

    blocks: [(title, [crop paths])].

    Not one file per cluster. macOS `open` on a second image either refocuses the window
    Preview already has up or adds a background tab, so eleven separate sheets all showed
    cluster 1: on 2026-08-17 the cat, the raccoon, the stone trough, the watering can and
    two patches of empty pavement all appeared to be hedgehogs. One sheet, opened once,
    has no window-management failure mode to get wrong.
    """
    import cv2
    made = []
    for title, paths in blocks:
        row = contact_sheet(paths, None, cols=cols, tile=tile, limit=cols)
        if row is None:
            continue
        row = cv2.copyMakeBorder(row, 24, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(row, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        made.append(row)
    if not made:
        return None
    w = max(r.shape[1] for r in made)
    made = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                               value=(30, 30, 30)) for r in made]
    # Two columns, so twenty clusters still fit on a screen rather than scrolling.
    half = (len(made) + 1) // 2
    left, right = cv2.vconcat(made[:half]), None
    if made[half:]:
        right = cv2.vconcat(made[half:])
        h = max(left.shape[0], right.shape[0])
        left = cv2.copyMakeBorder(left, 0, h - left.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                  value=(30, 30, 30))
        right = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 8, cv2.BORDER_CONSTANT,
                                   value=(30, 30, 30))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.hconcat([left, right]) if right is not None else left)
    return out


def contact_sheet(paths, out, cols=6, tile=180, limit=None):
    """Montage crops into one image. → the image (and writes `out`, if given)."""
    import cv2
    tiles = []
    for p in paths[:limit or cols * 5]:               # 30 crops is plenty to judge by
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        pad = cv2.copyMakeBorder(im, 0, tile - im.shape[0], 0, tile - im.shape[1],
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
        tiles.append(cv2.copyMakeBorder(pad, 2, 2, 2, 2, cv2.BORDER_CONSTANT,
                                        value=(60, 60, 60)))
    if not tiles:
        return None
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    blank = np.full_like(tiles[0], 20)
    rows[-1] += [blank] * (cols - len(rows[-1]))
    sheet = cv2.vconcat([cv2.hconcat(r) for r in rows])
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, sheet)
    return sheet


def load_refs():
    """→ {crop filename: label} already decided."""
    if not os.path.exists(REFS):
        return {}
    with np.load(REFS, allow_pickle=True) as z:
        return dict(zip(z["files"].tolist(), z["labels"].tolist()))


def save_refs(rows):
    """rows: [(file, label, embedding)] → animal_refs.npz."""
    np.savez(REFS,
             files=np.array([r[0] for r in rows], dtype=object),
             labels=np.array([r[1] for r in rows], dtype=object),
             embs=np.stack([unit(r[2]) for r in rows]).astype(np.float32)
             if rows else np.zeros((0, 0), np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-cos", type=float, default=MIN_COS,
                    help=f"how alike two crops must be to group (default {MIN_COS})")
    ap.add_argument("--review", action="store_true", help="show existing labels and stop")
    ap.add_argument("--set", metavar="N=LABEL,…",
                    help="label whole clusters by number without prompting, e.g. "
                         "--set 1=hedgehog,2=cat,5=_drop")
    ap.add_argument("--relabel", action="store_true", help="ask again about labelled crops")
    args = ap.parse_args()

    from gallery import Gallery
    gal = Gallery(os.path.join(BASE, "gallery"))
    embs, metas = gal.embeddings("animal")
    known = load_refs()

    if args.review:
        if not known:
            sys.exit("nothing labelled yet — run without --review")
        counts = {}
        for lab in known.values():
            counts[lab] = counts.get(lab, 0) + 1
        print(f"{len(known)} labelled crop(s) in {os.path.relpath(REFS, os.getcwd())}:")
        for lab, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4}  {lab}")
        return

    if not len(embs):
        sys.exit("no animal crops with embeddings yet — run detector/harvest_refs.py first")

    keep = [i for i, m in enumerate(metas)
            if args.relabel or m.get("file") not in known]
    if not keep:
        print(f"all {len(metas)} crop(s) already labelled — --review to see them, "
              f"--relabel to redo")
        return

    sub = embs[keep]
    lab = cluster(sub, args.min_cos)
    n_clusters = len(set(lab.tolist()))
    print(f"{len(keep)} unlabelled crop(s) -> {n_clusters} cluster(s) at cos>={args.min_cos}"
          f"  ({len(known)} already named)\n")

    rows = [(f, l, np.zeros(embs.shape[1], np.float32)) for f, l in known.items()]
    # Existing labels keep their real vectors where we still have them.
    by_file = {m.get("file"): embs[i] for i, m in enumerate(metas)}
    rows = [(f, l, by_file.get(f, e)) for f, l, e in rows]

    crops_dir = os.path.join(BASE, "gallery", "animals")
    members = {k: [keep[j] for j in np.flatnonzero(lab == k)] for k in range(n_clusters)}

    def describe(k):
        idx = members[k]
        events = sorted({metas[i].get("event", "live") for i in idx})
        tops = sorted({metas[i].get("top", "?") for i in idx})
        return (f"cluster {k + 1}: {len(idx)} crop(s)",
                f"    from: {', '.join(events[:3])}{' …' if len(events) > 3 else ''}",
                f"    speciesnet said: {', '.join(tops[:4])}")

    def apply(k, label):
        for i in members[k]:
            f = metas[i].get("file")
            if f:
                rows.append((f, label, embs[i]))
        return len(members[k])

    # One sheet with every cluster on it, opened once — see overview_sheet().
    sheet = overview_sheet([(f"cluster {k + 1}: {len(members[k])} crops",
                             [os.path.join(crops_dir, metas[i]["file"])
                              for i in members[k] if metas[i].get("file")])
                            for k in range(n_clusters)], SHEET)

    if args.set:
        # Non-interactive: label whole clusters by number, e.g. --set 1=hedgehog,5=_drop
        chosen = {}
        for part in args.set.split(","):
            if "=" not in part:
                sys.exit(f"--set wants cluster=label pairs, got {part!r}")
            num, name = part.split("=", 1)
            try:
                k = int(num.strip()) - 1
            except ValueError:
                sys.exit(f"--set cluster numbers must be integers, got {num!r}")
            if not 0 <= k < n_clusters:
                sys.exit(f"--set: cluster {num.strip()} does not exist "
                         f"(there are {n_clusters})")
            chosen[k] = name.strip().lower()
        for k in sorted(chosen):
            n = apply(k, chosen[k])
            print(f"cluster {k + 1}: {n:>3} crop(s) -> {chosen[k]!r}")
        for k in range(n_clusters):
            if k not in chosen:
                print(f"cluster {k + 1}: {len(members[k]):>3} crop(s) left unlabelled")
        stopped = False
    else:
        if sheet:
            print(f"all {n_clusters} clusters on one sheet: {sheet}\n")
            if sys.platform == "darwin" and sys.stdin.isatty():
                subprocess.run(["open", sheet], check=False)
        stopped = False
        for k in range(n_clusters):
            for line in describe(k):
                print(line)
            try:
                ans = input("    name (or s=skip, d=drop, q=quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n    stopping")
                stopped = True
                break
            if ans == "q":
                stopped = True
                break
            if ans in ("", "s"):
                continue
            label = "_drop" if ans == "d" else ans.lower()
            print(f"    -> {apply(k, label)} crop(s) labelled {label!r}")

    save_refs(rows)
    named = [r for r in rows if r[1] != "_drop"]
    kinds = sorted({r[1] for r in named})
    print(f"\nwrote {os.path.relpath(REFS, os.getcwd())}: {len(named)} labelled crop(s) "
          f"across {len(kinds)} label(s){' (stopped early)' if stopped else ''}")
    if kinds:
        print("  " + ", ".join(kinds))


if __name__ == "__main__":
    main()
