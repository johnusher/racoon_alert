#!/usr/bin/env python3
"""
Child or adult, measured against the gate.

The literature does this two ways. One estimates real height through a ground-plane
homography, which needs the camera calibrated — tilt, height, a reference length — and
recalibrated every time it moves. The other takes the head-to-body ratio from a pose
model, which is scale-invariant and needs no calibration at all, but costs a second model
and depends on keypoints landing correctly on a small child at distance.

Neither is necessary here, because the scene contains a ruler. A person standing AT the
gate is at the same depth as the gate, so perspective cancels: whatever shrinks them
shrinks the gate by the same factor. Head above the top rail, adult; head below it,
child. One number — the row of the rail — and it is a number we already need for the
gate's aperture.

A garden gate is about 1.10 m, a 2-year-old about 87 cm (WHO median; 82-93 cm covers the
3rd to 97th percentile), an adult about 1.70 m. Both fall the right side of the rail, but
NOT by the same amount, and the asymmetry is the thing to know: the toddler is only 0.23 m
under the rail while an adult is 0.60 m over it. The child side is what limits `margin` —
at the shipped 0.15 (0.165 m) he clears by 0.065 m, and anything above about 0.20 is wider
than his clearance and quietly turns the feature off. Widening the margin is therefore not
a free way to buy caution; it buys silence.

⚠️ There is a ceiling on how tall the gate may be. An adult of height A against a gate of
height G reads as a CHILD once (G - A) > margin*G, i.e. once G > A/(1 - margin) — with the
shipped margin that is 1.70/0.85 = 2.0 m. Below that the margin scales with the gate and
absorbs the error into "unsure" first, so a 1.10 m garden gate has enormous room and even
a head-height gate merely goes quiet rather than wrong. Re-check this if the gate is ever
replaced with something over 2 m.

Three things make it say "unsure" instead of guessing, and all three are deliberate:
  • a head within `margin` of the rail — that band is roughly a 0.9-1.3 m person, a
    3-to-7-year-old, and this system is built to spot ONE toddler, not to grade children;
  • a box cut off by the top of the frame, where the measured height is only a lower
    bound and a distant adult would read as a child;
  • a gate that has not been measured yet.

⚠️ Known weakness: a crouching or bending adult is genuinely child-shaped, and no
size-only measure can tell them apart. The answer if that ever fires in practice is the
pose model — head-to-body ratio sees the crouch. Deliberately not built yet: the camera
is not even mounted, so whether it happens is unmeasured, and guessing at it now would
mean carrying a second model for a problem that may not exist.
"""

CHILD, ADULT, UNSURE = "child", "adult", "unsure"


def classify(box, gate_top_row, gate_height_px, margin=0.15, frame_h=None, edge_slack=2):
    """Is the person in `box` shorter than the gate?

    box            — [x1, y1, x2, y2]; y1 is the top of the head
    gate_top_row   — image row of the gate's top rail, the ruler
    gate_height_px — the gate's height in the image, which sets the margin
    margin         — how far from the rail counts as too close to call, as a
                     fraction of the gate's height
    frame_h        — the frame height, so a head cut off by the top can be refused
    Returns (verdict, detail).
    """
    if gate_top_row is None or not gate_height_px or gate_height_px <= 0:
        return UNSURE, "the gate has not been measured"
    head = box[1]
    if head <= edge_slack:
        return UNSURE, f"head cut off by the top of the frame (row {head})"
    if frame_h and box[3] >= frame_h - edge_slack and head <= edge_slack:
        return UNSURE, "box fills the frame vertically"
    slack = margin * gate_height_px
    drop = head - gate_top_row              # positive = head below the rail = shorter
    where = (f"head {abs(drop):.0f}px {'below' if drop > 0 else 'above'} the gate's top "
             f"rail (gate {gate_height_px:.0f}px, needs {slack:.0f}px)")
    if drop > slack:
        return CHILD, where
    if drop < -slack:
        return ADULT, where
    return UNSURE, where + " — too close to the rail to call"


def describe(gate_top_row, gate_height_px, margin=0.15):
    if gate_top_row is None or not gate_height_px:
        return "off (the gate has not been measured)"
    return (f"child = head below row {gate_top_row:.0f} by more than "
            f"{margin * gate_height_px:.0f}px")
