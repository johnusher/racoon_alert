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

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all speciesnet-rules checks passed")
