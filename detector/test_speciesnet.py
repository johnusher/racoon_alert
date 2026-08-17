#!/usr/bin/env python3
"""
Tests for the SpeciesNet RULES — the bit that turns a 2498-way softmax into the three
answers the detector actually needs: is it a human, is it not a human, what species is it.
Model-free (synthetic probability vectors), so no 224MB download needed to run these.

The thresholds asserted here are the ones measured on the real archive — see the numbers
quoted in speciesnet.py.

Run:  ../.venv/bin/python detector/test_speciesnet.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speciesnet import SpeciesRules, short_name

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


LABELS = ["northern raccoon", "domestic cat", "human", "blank", "vehicle", "rodent", "red fox"]
R = SpeciesRules(LABELS)


def p(**kw):
    """A probability vector by label name; the remainder is spread over the rest."""
    v = np.zeros(len(LABELS), np.float32)
    for k, x in kw.items():
        v[LABELS.index(k.replace("_", " "))] = x
    left = max(0.0, 1.0 - float(v.sum()))
    zeros = [i for i in range(len(LABELS)) if v[i] == 0]
    for i in zeros:
        v[i] = left / len(zeros)
    return v


print("1. a real person is a person — the veto must never fire on them")
# measured: the 9 real humans in the archive ran 0.49 … 0.999
for name, hp in (("clear daylight human", 0.999), ("night human", 0.975), ("worst real human", 0.491)):
    v = R.verdict(p(human=hp))
    check(f"{name} ({hp}) is human", v.is_human and not v.not_human, f"human_p={v.human_p:.3f}")

print("\n2. the cat MegaDetector called a person is vetoed")
# measured: all 29 crops of the 21:18 cat scored P(human) 0.0017 … 0.0855
for name, hp in (("cat, worst case", 0.0855), ("cat, typical", 0.009), ("cat, best case", 0.0017)):
    v = R.verdict(p(human=hp, rodent=0.30, blank=0.20))
    check(f"{name} ({hp}) is NOT human", v.not_human and not v.is_human, f"human_p={v.human_p:.3f}")

print("\n3. an unsure verdict fails SAFE — it keeps the person event")
v = R.verdict(p(human=0.35, blank=0.30))
check("mid-range P(human) does not veto", not v.not_human, f"human_p={v.human_p:.3f}")
check("…and does not assert human either", not v.is_human)

print("\n4. species naming: confident only")
v = R.verdict(p(northern_raccoon=0.978))
check("the 03:53 raccoon is named", v.species == "northern raccoon", f"{v.species} p={v.top_p:.3f}")
v = R.verdict(p(domestic_cat=0.342, blank=0.30))
check("an unsure cat is NOT named", v.species is None, f"top={v.top_label} p={v.top_p:.3f}")
check("…but is still reported as top_label", v.top_label == "domestic cat")

print("\n5. 'blank' and 'vehicle' are not species")
v = R.verdict(p(blank=0.983))
check("empty night pavement names no species", v.species is None, f"top={v.top_label}")
check("…and is flagged not-human", v.not_human)
check("blank_p is exposed for logging", abs(v.blank_p - 0.983) < 1e-3, f"{v.blank_p:.3f}")
v = R.verdict(p(vehicle=0.9))
check("a vehicle names no species", v.species is None)

print("\n6. a human is never returned as a 'species'")
v = R.verdict(p(human=0.99))
check("human is not a species label", v.species is None, f"{v.species}")

print("\n7. thresholds are configurable and honoured")
strict = SpeciesRules(LABELS, human_veto=0.05, species_min=0.90)
v = strict.verdict(p(human=0.0855, rodent=0.3))
check("a stricter veto lets the worst cat through", not v.not_human, f"human_p={v.human_p:.3f}")
v = strict.verdict(p(northern_raccoon=0.978))
check("a stricter species_min still names 0.978", v.species == "northern raccoon")
v = strict.verdict(p(northern_raccoon=0.80))
check("…but not 0.80", v.species is None)

print("\n8. a label set with no 'human' entry degrades safely, never vetoes")
odd = SpeciesRules(["blank", "rodent"])
v = odd.verdict(np.array([0.5, 0.5], np.float32))
check("no human label -> never claims not-human", not v.not_human and not v.is_human)

print("\n9. common names shorten to a usable one-word tag")
for full, want in (("northern raccoon", "raccoon"), ("domestic cat", "cat"),
                   ("red fox", "fox"), ("white-tailed deer", "deer"),
                   ("european hedgehog", "hedgehog"), ("domestic dog", "dog")):
    check(f"{full!r} -> {want!r}", short_name(full) == want, str(short_name(full)))
check("'raccoon dog' is NOT shortened to 'dog'", short_name("raccoon dog") == "raccoondog",
      str(short_name("raccoon dog")))
check("None stays None", short_name(None) is None)
check("a tag is filename-safe", all(c.isalnum() for c in short_name("white-tailed deer")))

print("\n10. the verdict carries the tag the event will actually use")
v = R.verdict(p(northern_raccoon=0.978))
check("raccoon verdict tags RACCOON", v.tag == "RACCOON", str(v.tag))
v = R.verdict(p(domestic_cat=0.342, blank=0.3))
check("an unnamed species has no tag", v.tag is None, str(v.tag))

print("\n11. .identified — the promote side, for someone who stands still")
# The friend who walked out at 22:48 on 2026-08-16 and stared at the camera got no
# alert: standing still, their box never moved, so the movement gate held them back
# as 'unproven' for ever. A positive identification has to be able to fire on its own.
check("a real person is identified", R.verdict(p(human=0.975)).identified)
check("the worst real person is still identified", R.verdict(p(human=0.491)).identified)
check("a named raccoon is identified", R.verdict(p(northern_raccoon=0.978)).identified)
# …and the things the movement gate exists to suppress must NOT be promoted. These are
# the measured verdicts for the bench, the rock, the plant pot and the orange bucket.
for name, pr in (("bench", 0.921), ("plant pot", 0.99), ("orange bucket", 0.97),
                 ("night pavement", 0.983), ("night bush", 0.947)):
    v = R.verdict(p(blank=pr))
    check(f"{name} (blank={pr}) is NOT promoted", not v.identified, repr(v))
check("an unsure animal is not promoted", not R.verdict(p(domestic_cat=0.342, blank=0.3)).identified)
check("a mid-range human is not promoted on its own",
      not R.verdict(p(human=0.35, blank=0.3)).identified)

print("\n12. a taxonomic ROLLUP is not a species (the 2026-08-17 garden trough)")
# MegaDetector called the stone trough at [1263,568,1653,1070] `person` 0.22-0.80 for
# 100 minutes. SpeciesNet read that crop as `bird` 0.49-0.72 — and `bird` is the
# CLASS-level rollup `aves;;;;;bird`, not a species. NOT_A_SPECIES listed only
# blank/vehicle/human/animal, so `bird 0.55` counted as a positive identification,
# promoted the box straight past the movement gate and tagged 12 BIRD events in ten
# minutes. 432 of the 2498 labels (17%) are rollups like this: 282 `… species`,
# 108 `… family`, 18 `… order`, and 24 bare ones (bird, mammal, rodent, primate, bat…).
ROLLUPS = ["bird", "mammal", "rodent", "carnivorous mammal", "primate", "bat",
           "procyon species", "icteridae family", "galliformes order"]
ROLL_LABELS = ROLLUPS + ["human", "blank", "northern raccoon"]
RR = SpeciesRules(ROLL_LABELS)


def rp(label, prob):
    """A vector peaked on one label of ROLL_LABELS."""
    v = np.zeros(len(ROLL_LABELS), np.float32)
    v[ROLL_LABELS.index(label)] = prob
    rest = [i for i in range(len(ROLL_LABELS)) if v[i] == 0]
    for i in rest:
        v[i] = max(0.0, 1.0 - prob) / len(rest)
    return v


for label in ROLLUPS:
    v = RR.verdict(rp(label, 0.72))                 # 0.72 = the trough's best `bird` score
    check(f"{label!r} names no species", v.species is None, f"species={v.species!r}")
    check(f"{label!r} does not promote past the movement gate", not v.identified, repr(v))
check("a real species is still named next to them",
      RR.verdict(rp("northern raccoon", 0.978)).species == "northern raccoon")
check("…and still promotes", RR.verdict(rp("northern raccoon", 0.978)).identified)

print("\n13. the real label file decides what a species is, by taxonomy")
_labels_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "speciesnet_labels.txt")
if os.path.exists(_labels_file):
    from speciesnet import load_labels, load_species_names
    names, spp = load_labels(_labels_file), load_species_names(_labels_file)
    fr = SpeciesRules(names, species_labels=spp)
    check("every label is classified", len(names) == 2498, f"{len(names)}")
    # 2498 labels = 2066 at species rank + 432 roll-ups; the species names collapse to
    # 2065 because 'domestic water buffalo' is listed twice, under two UUIDs.
    check("the taxonomy roll-ups are found", len(names) - len(spp) == 433,
          f"{len(names)} labels - {len(spp)} species names")
    check("…and that is 432 roll-ups plus the one duplicated name",
          sum(1 for n in names if n in spp) == 2066, f"{sum(1 for n in names if n in spp)}")
    for label in ("bird", "mammal", "blank", "vehicle", "reptile",
                  "icteridae family", "galliformes order"):
        check(f"{label!r} is not a species", not fr.is_species(label))
    # homo sapiens IS a species by taxonomy — but a person is reported by is_human, and
    # naming them as a species would tag the event with their genus.
    check("'human' is a species to the taxonomy", "human" in spp)
    check("…but is never announced as one", not fr.is_species("human"))
    for label in ("northern raccoon", "domestic cat", "red fox",
                  "western european hedgehog"):
        check(f"{label!r} IS a species", fr.is_species(label))
    i_bird = names.index("bird")
    probs = np.full(len(names), 0.28 / (len(names) - 1), np.float32); probs[i_bird] = 0.72
    v = fr.verdict(probs)
    check("the trough's `bird 0.72` names no species", v.species is None, repr(v))
    check("…and cannot promote past the movement gate", not v.identified)
else:
    print("  SKIP  labels file absent — detector/get-speciesnet.sh")

print("\n14. `blank` is 'nothing is there', not 'therefore an animal'")
# The 07:45:19 ANIMAL event: SpeciesNet said blank=0.62 human=0.00 on the trough crop.
# human_p 0.00 satisfies the not_human veto, so detect.py re-tagged the person box
# `animal` and fired an ANIMAL alert about an empty stone trough. A blank crop has to be
# distinguishable from one holding an animal, or "not a person" silently means "animal".
v = R.verdict(p(blank=0.62, human=0.0))
check("a blank crop is flagged blank", v.is_blank, repr(v))
check("…and is still not-human", v.not_human)
check("…and names no species", v.species is None)
v = R.verdict(p(northern_raccoon=0.978))
check("a named animal is NOT blank", not v.is_blank, repr(v))
v = R.verdict(p(human=0.975))
check("a human is NOT blank", not v.is_blank)
# `blank` need not win outright to mean nothing is there, but the top label decides.
v = R.verdict(p(blank=0.48, domestic_cat=0.30))
check("blank on top at low confidence still reads blank", v.is_blank, repr(v))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all speciesnet-rules checks passed")
