# Archery Target Detection — Research Notes

---

## 1. Current algorithm summary

`findTarget` runs this pipeline:

```
pretreat (Gaussian blur 15×15 + erode×1 + dilate×3)
    │
    ├─ applyColorFilter(yellow) ─┐
    ├─ applyColorFilter(red)     ├─ aggregateBlobs → extractBoundary → cleanupContourPoints → fitEllipsePCA
    └─ applyColorFilter(blue)   ─┘
              │
       3 RotatedRect (yellow, red, blue)
              │
       findWeakestElement → discard worst
              │
       linearInterpolateTo10Rings (least-squares on 2–3 anchor points)
              │
       EllipseData[10]
```

HSV thresholds (HSV_FULL 0-255, with a BGR-as-RGB channel swap):

| Color  | H range  | Range width |
|--------|----------|-------------|
| Yellow | 136 – 140 | 4           |
| Red    | 168 – 171 | 3           |
| Blue   | 24  – 32  | 8           |

---

## 2. Observed failure modes

Diagnostic output from `scripts/diag.ts` run on all 10 test images:

### 2a. Complete color-detection miss → all rings identical

Three images produce all 10 rings with identical geometry:

| Image | cx | cy | w | h | Symptom |
|-------|----|----|---|---|---------|
| 20190325_193217.jpg | 307.8 | 598.6 | 201.8 | 180.3 | all rings same |
| 20190325_202607.jpg | 292.2 | 602.0 | 282.7 | 261.7 | all rings same |
| 20190325_204137.jpg | 298.6 | 588.9 | 265.4 | 237.7 | all rings same |

**Root cause**: When only 1 of the 3 color masks succeeds, `linearInterpolateTo10Rings`
receives n=1 valid anchor. With a single point, the least-squares denominator
`xSq = n·xxSum − xSum²` equals zero, so every regression function degenerates to
`() => constant`. All 10 rings collapse to the single detected ellipse.

### 2b. Degenerate inner ring → bad extrapolation inward

Three images produce a near-zero or wildly distorted bullseye:

| Image | ring[0] w | ring[0] h | ring[0] ar | Note |
|-------|-----------|-----------|------------|------|
| 20190325_193820.jpg | 84.4 | 23.5 | 3.59 | very elongated |
| 20190325_195129.jpg |  4.6 |  1.0 | 4.62 | effectively a point |
| 20190325_195801.jpg |  8.4 |  1.0 | 8.43 | effectively a point |

**Root cause**: Linear regression over yellow (x=0), red (x=1), blue (x=2) then
evaluates at `x = −0.5` for ring 0. When the yellow ellipse itself is a poor fit
(partial detection, wrong blob, or only 1–2 anchors), the extrapolated bullseye is
either extremely small or highly eccentric.

### 2c. Tests pass despite wrong results

The current Jest tests only check:
- `success: true`
- `rings.length === 10`
- Each ring has positive dimensions and in-bounds center
- All centers within 100 px of ring[0]

Failure modes 2a and 2b both satisfy all four conditions (identical rings are 0 px
apart from each other; near-zero rings still have w > 0 after `Math.max(1, ...)` in
the interpolation). **The test suite gives false confidence.**

---

## 3. Root cause analysis

### 3a. HSV thresholds are too narrow and too brittle

A range of 3–4 HSV_FULL hue units (out of 0–255) leaves almost no tolerance for:

- Exposure variation (bright sun vs. overcast / indoor)
- White-balance differences between cameras
- JPEG compression artifacts (DCT ringing near color edges)
- Target aging / fading / different print batches

### 3b. The BGR-as-RGB convention multiplies fragility

The thresholds were calibrated by running OpenCV's `COLOR_RGB2HSV_FULL` on a
BGR-loaded image (R↔B channels swapped). `rgbToHsvFull` replicates this by
internally swapping R and B before converting. The thresholds are therefore only
valid for images decoded through exactly that same BGR path. Any other decoder
(libjpeg/Jimp decodes as RGB) that doesn't have the same quirk requires a different
calibration.

In practice this means: the thresholds were hand-tuned on a handful of images taken
with one specific phone, in one specific lighting condition, decoded in one specific
way. They do not generalise.

### 3c. Linear interpolation assumes equal ring spacing

The least-squares line is fit to 3 points at x = 0, 1, 2 (yellow, red, blue rings),
then extrapolated to x = −0.5 … 4.5 to produce the 10 output rings. This model
assumes all ring ellipses grow at a constant rate, which is only approximately true
when the target is viewed head-on. Any perspective or radial distortion breaks the
linearity — and even without distortion, the extrapolation to both ends (bullseye and
outer white rings) amplifies any error in the 3 anchor fits.

### 3d. PCA ellipse fit is fragile for partial blobs

The boundary pixels fed into `fitEllipsePCA` come from the aggregated color blob.
When lighting causes only part of a ring to match the color threshold (e.g., one side
of the red ring is in shadow), the boundary is a partial arc, not a complete ellipse.
PCA on a partial arc systematically underestimates the major axis and produces an
incorrect angle. `cleanupContourPoints` then filters out valid far-field points,
making the problem worse.

---

## 4. Candidate algorithms

### Algorithm A — Wider adaptive HSV thresholds

**Idea**: Replace the fixed narrow thresholds with a two-pass approach:
1. Apply a generous initial range (e.g., yellow H ± 15 units instead of ± 2)
2. If the largest blob covers less than a minimum area, widen the range and retry
3. Use the found blob's actual HSV centroid to re-centre the threshold for that image

**Pros**: Minimal code change; keeps the existing pipeline structure.
**Cons**: Still fundamentally HSV-threshold-based; will still fail when the target
color genuinely falls outside the range (e.g., faded targets, unusual lighting).
Does not fix the linear extrapolation problem. Does not fix PCA on partial arcs.

---

### Algorithm B — Contour hierarchy (nested contour approach)

**Idea**: Compute a Canny-like edge image over the full colour image, find all
contours, filter to those that are roughly elliptical, and look for a group of nested
concentric ellipses. The archery target has a distinctive set of 3–5 nested
colour-boundary contours.

**Pros**: Robust to lighting changes (edges are relative, not absolute HSV values).
Does not depend on specific colours at all — works with any colour target.
**Cons**: Edge detection and contour finding are non-trivial in pure TypeScript.
Matching nested contours to the archery target geometry requires additional heuristics.
Computationally heavier.

---

### Algorithm C — Radial profile sampling

**Idea**:
1. **Coarse center estimate** — use the existing colour-blob centroids (or their
   average if multiple colors are found) to get an approximate target center.
2. **Radial rays** — cast N rays (e.g., 180) from that center outward.
3. **Sample & detect transitions** — along each ray, read pixel hues and detect
   step changes (hue-gradient crossings). Each transition is a candidate ring boundary.
4. **Cluster transitions by radius** — at each expected ring boundary, the transitions
   across all rays should cluster at a consistent radius. Fit an ellipse to the
   clustered points.
5. **Validate with color pattern** — the sequence of colours inward → outward should
   follow white → black → blue → red → yellow.

**Pros**: Does not depend on absolute color values; tolerates partial coverage and
occlusion; naturally produces one ellipse per ring boundary; the concentric structure
becomes a signal rather than a constraint.
**Cons**: Center estimate must be at least roughly correct. Requires implementing
ray-casting + transition detection + per-ring ellipse fitting — moderate complexity.

---

### Algorithm D — Fixed physical ring ratios

**Idea**: All World Archery targets have rings with equal width — meaning the outer
radius of ring k is proportional to k (1× ring_width for ring 10, 10× for ring 1).
Once any one ring's ellipse is known, all others follow directly from this fixed
ratio, without regression.

Given detected ellipses for yellow, red, blue:
- Yellow ring = rings 9 + 10 combined zone: mean radius ≈ 1.5 × w
- Red ring = rings 7 + 8: mean radius ≈ 3.5 × w
- Blue ring = rings 5 + 6: mean radius ≈ 5.5 × w

(where w = single ring width in pixels)

Two unknowns (center + w) can be solved from any two detected rings. The third
provides a consistency check. No regression needed.


**Pros**: Physically grounded; one good detection is sufficient to reconstruct all
rings; eliminates runaway extrapolation.
**Cons**: Only valid for standard WA targets. Target size and camera distance must
vary, but the *ratio* is always fixed. Requires knowing which ring each colour
detection corresponds to.

---

### Algorithm E — Fitzgibbon direct algebraic ellipse fit (replaces PCA)

**Idea**: Replace `fitEllipsePCA` with the Fitzgibbon-Pilu-Fisher (1996) constrained
algebraic fit. It minimises the sum of squared algebraic distances subject to the
ellipse constraint `4ac − b² = 1`, making it robust to non-uniform point distribution
around the arc. Halir & Flusser (1998) provide a numerically stable formulation that
avoids the generalised eigenvalue problem.

**Pros**: Drop-in replacement for PCA; handles partial arcs correctly; better-conditioned
for skewed point distributions; ~6×6 linear algebra, fast.
**Cons**: Requires a 6×6 SVD or eigenvalue solve (can be implemented without external
libraries); slightly more code than PCA.

---

## 5. Recommendation

No single change is sufficient. The recommended approach combines fixes at each
failure point:

### Tier 1 — Quick wins (fix the immediate failures)

1. **Widen and de-quirk HSV thresholds** (Algorithm A)
   Correct the BGR-as-RGB swap so the thresholds are expressed in standard RGB→HSV
   space, and widen ranges significantly:
   - Yellow: H 25–40° (standard 0-360°, map accordingly)
   - Red: H 0–12° **and** H 348–360° (wraps around)
   - Blue: H 195–235°
   Add per-image adaptive re-centering: measure the median hue of the found blob and
   re-threshold ± 20 hue units around it, then re-run detection.

2. **Replace linear regression with fixed-ratio reconstruction** (Algorithm D)
   Use the known 1:3.5:5.5 radius ratios for yellow:red:blue to solve for ring width
   and center directly. Fall back to regression only when all three are found.

### Tier 2 — Accuracy improvements

3. **Replace PCA with direct algebraic ellipse fit** (Algorithm E)
   Fitzgibbon / Halir-Flusser fit. Especially improves accuracy on partial arcs
   (partial lighting, rings cut at image boundary).

4. **Radial profile validation pass** (Algorithm C, validation step only)
   After fitting ellipses, cast radial rays from the estimated center and check that
   colour transitions occur at the expected radii. Use this to score and re-rank
   candidate ellipses rather than relying on the "weakest element" heuristic.

### Tier 3 — Full robustness

5. **Full radial-profile primary detection** (Algorithm C)
   Use radial ray sampling as the *primary* detection path (not just validation).
   This makes the algorithm lighting-independent and removes the dependency on HSV
   thresholds entirely.

---

## 6. Implementation priority and estimated impact

| Change | Fixes | Complexity |
|--------|-------|------------|
| Wider + standard HSV thresholds | Failures 2a (colour miss) | Low — tune constants |
| Adaptive per-image threshold re-centering | Remaining 2a cases | Low-Medium |
| Fixed-ratio ring reconstruction (Algorithm D) | Failure 2b (bad extrapolation) | Medium |
| Fitzgibbon ellipse fit (Algorithm E) | Failure 2b + partial arcs | Medium |
| Radial-profile validation | All failure modes | Medium-High |
| Full radial-profile detection | All failure modes, any lighting | High |

The quickest path to getting all 10 images correct is: **fix HSV thresholds first**,
then **replace linear regression with fixed-ratio reconstruction**. Together these
address the two root causes behind 6 of 10 failures.

---

## 7. Test suite improvements needed

The current tests pass even when results are wrong. New assertions needed:

- Ring widths must be **strictly increasing** (ring[i].width < ring[i+1].width)
- Successive ring width ratios must be within a reasonable range of the expected
  physical ratio (e.g., each ring ~10% wider than the previous, ± some tolerance)
- Bullseye (ring[0]) must have a realistic minimum size relative to the image
  (e.g., > 1% of image width)
- No two consecutive rings should have identical geometry (catches the degenerate
  constant-regression case)

---

*Last updated: 2026-03-13*
