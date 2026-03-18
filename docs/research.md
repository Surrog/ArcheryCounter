# ArcheryCounter — Research Notes

---

## §1 Why ellipses and quadrilaterals are wrong (summary)

An archery target face is non-planar: gravity sag, arrow damage, edge curl, and humidity deform it. Its image is therefore **not** a projective transform of the canonical flat target.

- **Ellipse fit fails** because rings are only true conics if the surface is flat. Inner rings (bullseye) are most deformed — precisely where accuracy matters most.
- **4-corner quadrilateral fails** because curved/bowing edges between pins cannot be represented by four straight line segments.

**Resolution:** free-form closed spline rings (§2) for ring boundaries; a 4–8 vertex polygon for the target boundary; radial-profile sampling as the internal detection representation.

---

## §2 Representations

| Representation | Use | Notes |
|---|---|---|
| Radial profile (N×10 distances) | Internal detection | 3 600 values/image — impractical to annotate manually |
| Free-form closed spline (K=12 Catmull-Rom control points) | Ring output + annotation ground truth | Expressive, compact, practical to drag-annotate |
| 4–8 vertex polygon | Target boundary | Handles moderate edge curvature; gift-wrap + simplify |
| Thin-plate spline warp | Future fallback | Handles arbitrary deformation but requires solving a harder sub-problem |

---

## §3 Detection algorithm (implemented — `targetDetection.ts`)

### 3.1 Pipeline

1. **Pretreatment** — Gaussian blur (15×15, σ=1.5) + erode×1 + dilate×3 on 2× downsampled image (BOOTSTRAP_SCALE=2). Float64Array HSV cache computed once per pixel for speed.
2. **Boundary scan** — 180 rays from image centre; walk outward until hay-bale colour (H∈[15°,65°], S>0.25, V>0.20) or image edge. Circular median filter (±10 rays); convex hull + simplify to ≤8 vertices.
3. **Colour calibration** — 8 rays × 2 samples/zone. Circular-mean hue + median S/V per zone; von Kries white-balance correction.
4. **Colour-guided ring detection** — 32 rays; 5-point mode-smooth zone classifications; detect 4 colour-zone transitions (gold→red, red→blue, blue→black, black→white) per ray with MIN_STREAK=3 and 50%-of-expected minimum-distance gate (prevents arrow-hole false positives near centre).
5. **White ring closure** — black-line scan outward from white zone start; fallback to OLS linear regression through known colour-transition distances.
6. **Monotonicity enforcement** — forward pass on detected `transitionDist[]` before commit, then final pass on full `result[0..9]`. Violations nulled; missing rings filled by spline interpolation from neighbouring rays.
7. **Spline construction** — detected rings [1,3,5,7,9] → Catmull-Rom SplineRings directly; interpolated rings [0,2,4,6,8] via point-wise spline interpolation from adjacent detected rings.

### 3.2 HSV convention

Standard RGB→HSV (H 0–360°, S/V 0–1). Wide initial ranges (yellow 20–70°, red 0–18°+342–360°, blue 190–245°) adaptively re-centred around the measured median hue per image.

### 3.3 Known failure mode (fixed): monotonicity violations

Each of the 4 colour-zone transitions was scanned independently from ray origin. A colour false positive (hay bale, arrow shaft, specular reflection) could place an outer transition *closer* than an inner one. Similarly, regression-derived r8/r9 using scale estimate `w` could fall below a detected r5. Fixed by the two-pass monotonicity enforcement described above. Verified by the per-ray distance ordering test in `targetDetection.test.ts`.

---

## §4 Scoring approach (deferred — see plan.md §P8)

**Primary: colour-zone classification at the arrow tip.**

1. Sample HSV in a small annular patch around the tip; exclude hay-coloured pixels; take modal zone → score range (pair).
2. Disambiguate within zone: distance to inner and outer SplineRing boundaries → exact score.
3. X-ring: distance from centre < ~40% of gold zone radius.

Per-image colour calibration (§3 above) is critical; hardcoded ranges are insufficient under mixed lighting.

---

## §5 Migration history

| Phase | Representation | Status |
|---|---|---|
| v1 | Native C++ / OpenCV (RotatedRect ellipse fit) | Removed |
| v2 | Pure-TS ellipse fit (Fitzgibbon/Halir-Flusser) + 4-point quad | Removed |
| v3 | Boundary-first polygon mask → colour-zone scoring + spline ring output | **Current** |
| v4 | Thin-plate spline non-rigid warp | Future fallback if needed |

---

## §6 Explicitly excluded approaches

| Approach | Reason |
|---|---|
| Homography-based canonical scoring | Corrects perspective only; does not address surface deformation |
| Lens distortion correction | Algorithm works in image space; no rectified input assumed |
| Radial profile as annotation format | 3 600 values/image; impractical to annotate manually |
| Perspective rejection / user notification | Derive from observed failures, not preemptively |

---

## §7 Arrow detection (research)

**Deferred until ring and boundary detection are stable (phases 1–7 complete).**

### 7.1 Physical description

An arrow in a target photo has three visually distinct regions:

| Region | Appearance | Notes |
|---|---|---|
| **Nock** | Small (~8–12 mm dia.), brightly coloured (yellow, orange, green, blue, red) near-circular disc at the rear end | Most visually distinct element; highest-saturation blob not matching any target zone |
| **Shaft** | Thin (6–10 mm dia.) line, typically glossy **black** carbon fibre; extends from nock into the target face | 1–5 px wide at 1200 px image width; glossy surface produces a bright specular highlight streak alongside the dark body |
| **Vanes / fletching** | 3 thin plastic fins (often white or translucent) extending from just behind the nock | ~2–3 cm long; may be folded against the shaft in a tight group |

The **impact point** — where the shaft enters the paper — is the scoring location. The shaft protrudes perpendicular (or near-perpendicular) to the target face. From a slightly-above-centre camera position, all shafts in the same image project as **roughly parallel lines** pointing toward the vanishing point of the camera direction, converging slightly toward the image centre.

### 7.2 Key constraints

- **All shafts are parallel** (to first order): arrows shot at the same target face all travel the same direction, so their projections in the image share a common vanishing point. This is the strongest structural prior for detection.
- **Nocks float in front of the target plane**: their image position is offset from the impact point by an amount that depends on shaft length and camera angle. For a typical 28" arrow at 30° oblique view, the nock appears 5–30 px displaced from the entry point (at 1200 px image width).
- **Shafts do not cross colour boundaries**: the entry point is always within the target face; the shaft overlays but does not belong to any ring zone.
- **Multiple arrows**: 3 or 6 arrows per photo in competition; all from the same direction.
- **Arrow holes (post-removal)**: 5–15 px dark/warm circular holes (exposed hay) cluster near the centre. Simpler to detect than shafts but less accurate in position.

### 7.3 Detection approaches

#### A. Nock-first (recommended primary approach)

Nocks are the visually cleanest element:
1. **Detect nock blobs**: look for small (5–20 px diameter), near-circular, highly-saturated blobs inside or near the target boundary. Prefer colors outside the known target-zone hue ranges (see calibration), but any high-saturation small circle is a candidate.
2. **Estimate shaft direction**: cast multiple oriented line-scan rays from each nock candidate toward the target center. The direction minimizing mean luminance (dark shaft) is the shaft axis. All shaft directions should agree within ~5°; use the modal direction as the prior for all arrows.
3. **Trace shaft to impact point**: follow the shaft axis from the nock inward until luminance jumps to match the local target-zone background. The last sub-threshold pixel is the entry point.

**Strengths**: robust to shaft color, works even with specular highlights (the dark body is always darker than the background), natural multi-arrow handling.

**Weaknesses**: nock color can overlap target zone colors (e.g., a red nock on a red zone → hard to distinguish). Mitigation: require the blob to be isolated (not part of a large uniform region) and slightly above the target plane (displaced from the target surface by the shaft projection).

#### B. Shaft line detection (Hough / LSD)

The shaft appears as a thin line segment. A Hough line transform (or Line Segment Detector) on a luminance-edge image finds candidate lines. Filter:
- Within the target boundary
- Appropriate length (20–200 px at 1200 px width)
- Consistent direction (within 10° of the estimated shaft vanishing point)
- One endpoint on or near the target surface (impact point), other endpoint free (nock end)

**Strengths**: directly finds the shaft regardless of nock color.

**Weaknesses**: many false positives (ring-outline edge segments, ring divider lines). Requires a Hough transform implementation in pure TypeScript. The shaft direction prior from nock detection (§A) dramatically reduces false positives.

#### C. Doublet filter (specular shaft signature)

A glossy shaft under directional light shows a bright specular streak ≈1 px to one side of a dark line. This "doublet" (dark–bright or bright–dark depending on light direction) is distinctive. A matched filter tuned to this pattern (width ≈3 px, oriented along the shaft direction) would have high specificity.

**Strengths**: high selectivity against ring-outline edges, which are dark–bright on one side only.

**Weaknesses**: requires knowing the shaft direction and light direction first; a good refinement step after A or B, not a standalone detector.

#### D. Arrow-hole detection (fallback / post-pull mode)

After arrows are removed, circular holes (5–15 px diameter) expose the hay bale (H∈[15°,65°], S>0.25, V<0.6). These appear as small warm-dark blobs within the target boundary.

1. Subtract the expected colour of each target zone (from calibration) to suppress background.
2. Look for residual warm-dark blobs of roughly circular shape.
3. Cluster nearby candidates (arrows land within a few mm of each other in a tight group).

**Strengths**: no shaft or nock needed; works on post-session photos; simpler blob detector.

**Weaknesses**: less accurate (hole position ≈ shaft entry point ± 3–5 px, acceptable for scoring); faint on thick paper; multiple holes from same arrow position (re-shooting) creates ambiguity.

### 7.4 Recommended implementation order

1. **Nock detector** (P9-T1): start with a simple thresholded blob detector (find connected components with saturation > 0.5, V > 0.4, area 20–400 px², eccentricity < 0.7).
2. **Shaft direction prior** (P9-T2): from 2+ nock candidates, estimate the common vanishing point direction.
3. **Impact point localisation** (P9-T3): trace shaft ray inward from each nock to the paper surface.
4. **Arrow-hole fallback** (P9-T5): implement as alternative path when no nocks are found.
5. **Doublet refinement** (later): once the direction is known, use the doublet filter to sub-pixel refine the impact point.

### 7.5 Test dataset requirements

The current test images (`images/*.jpg`) do not contain arrows. A separate dataset is needed:

- 10–20 images with arrows in place (3 or 6 arrows each), varied lighting conditions
- 5–10 images taken after arrows are removed (hole mode)
- Ground truth: manually annotated impact points (pixel coordinates) per arrow

Store annotations in the same PostgreSQL `annotations` table used for ring ground truth, adding an `arrows` column: `[{ tip: [x, y], nock: [x, y] | null }]`.

### 7.6 Open questions

| Question | Notes |
|---|---|
| Nock color variability | Can be any saturated color including those that appear on the target. A color classifier trained per-image (like zone calibration) may be needed. |
| Indoor vs outdoor lighting | Indoor: diffuse, low specular; outdoor: directional sun → strong specular on shaft. The doublet filter is more useful outdoors. |
| Arrow damage to target around impact | The paper tears and crumples; the entry point is not a clean circle. Use the shaft-ray method rather than trying to detect the hole shape. |
| Arrows covering ring boundaries | A shaft crossing a ring boundary can occlude detection; the shaft mask should be subtracted before ring detection if shafts are detected first. (Future: run arrow detection before ring detection for images where arrows are visible.) |
