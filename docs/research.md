# ArcheryCounter — Research Notes

## §1 Why ellipses and quadrilaterals are wrong

### 1.1 The physical reality

An archery target face is a flat, printed piece of paper mounted on a hay bale. In an ideal photo it would be perfectly flat and at a known angle to the camera. In practice:

- **Gravity sag** — the face bows between its pin attachment points; the centre (where arrows accumulate and the paper weakens) sags more than the edges.
- **Arrow damage** — each arrow piercing the face tears and ripples the paper around the hole, especially near the centre where density is highest.
- **Edge curl** — unclipped outer edges fold forward; corners are the worst offenders.
- **Rain / humidity** — a wet target dries unevenly, leaving creases and waves across the full face.
- **Stacking** — targets bolted on the same bale can overlap and push adjacent faces out of plane.

The resulting shape in a photo is a **non-planar surface** projected through a camera. It is not a plane; therefore its image is not a projective transform of the canonical flat target.

### 1.2 Why ellipses fail for rings

If the target were perfectly flat and tilted, the printed circles (rings) would project as **exact conics** (ellipses under normal perspective). The current algorithm assumes this and fits one ellipse per ring.

The assumption breaks in two ways:

| Failure mode | Effect |
|---|---|
| Surface bowing / curvature | Circle → curve that is *wider* than the fitted ellipse at the belly of the bow, narrower at the edges |
| Arrow-induced ripple near centre | Innermost rings are most deformed; outer rings (away from arrows) are relatively stable |
| Edge curl at corners | Outermost ring appears squashed or folded back |

The inner rings (bullseye, 9-ring) are the *most important* for scoring and also the *most deformed* — an ellipse fit there is doubly unreliable.

### 1.3 Why a quadrilateral fails for the boundary

The target boundary is a rectangle in 3D space. Under pure perspective projection it maps to a quadrilateral. In practice:

- Each corner is pinned independently; if one pin sits higher than another, the edge between them curves.
- A bowing face means the edges between corners are **concave or convex curves**, not straight lines.
- Heavy rain causes the bottom edge to sag significantly.

A 4-vertex polygon cannot represent a curved boundary.

---

## §2 Candidate representations

### 2.1 Radial profile (polar boundary sampling)

**Representation:** For each of N uniformly-spaced angles θᵢ from the target centre, store the distance rᵢⱼ at which the image crosses each of the J ring boundaries. This gives an N × J matrix.

**Strengths:**
- Zero geometric assumptions — works for any ring shape.
- Natural fit for the algorithm's existing radial sampling (Phase 4 in `targetDetection.ts`).
- Scoring an arrow: compute its distance from centre at the relevant angle, compare to rᵢⱼ for that angle → exact ring assignment.

**Weaknesses:**
- Storage: 360 × 10 = 3 600 values per image.
- Manual annotation is impractical — you cannot drag 3 600 points.
- Interpolation artefacts if N is small.

**Verdict:** Good as an *internal algorithm representation* for detection; not suitable as a ground-truth annotation format.

### 2.2 Free-form closed spline rings

**Representation:** Each ring boundary is a closed **Catmull-Rom** (or cubic B-spline) curve defined by K control points (K ≈ 8–16). The curve interpolates or approximates the control points and closes on itself.

**Strengths:**
- Far more expressive than an ellipse; can represent any smooth closed curve.
- K ≈ 12 gives 120 values per image (vs 3 600 for full radial), easily stored as JSON.
- Annotation is practical: drag 12 points per ring.
- Scoring: standard point-in-closed-curve test (ray casting or winding number).

**Weaknesses:**
- A fold or sharp crease requires many control points to capture; a spline stays smooth.
- The algorithm must detect where to place the initial control points automatically.

**Verdict:** Best candidate for the **annotation ground-truth format** and as a scoring primitive. Replaces ellipses in both the annotation tool and the test suite.

### 2.3 Color-zone segmentation (direct colour scoring)

**Representation:** Instead of storing ring *boundaries*, classify each image pixel inside the target boundary as belonging to a colour zone: gold (9–10), red (7–8), blue (5–6), black (3–4), white (1–2).

**Within-zone score disambiguation:** Each colour zone contains exactly two adjacent scoring rings of the same colour. Once the zone is identified, determine which of the two rings the arrow belongs to by computing the arrow's distance to the inner zone boundary and to the outer zone boundary, then checking which half it falls in. If it is closer to the inner boundary → higher score; closer to the outer boundary → lower score. The inner and outer zone boundaries come from the spline ring representation (§2.2).

**Strengths:**
- Completely bypasses the geometry problem. Colour is printed on the target; deformation moves the colour with the paper — no modelling needed.
- Extremely robust: a bowed, creased, folded face still has its correct colours.
- The algorithm already has colour blob detection (Phase 2 in `targetDetection.ts`); extending it to full segmentation is natural.
- Works in image space — no perspective correction required; oblique photos are handled naturally.

**Weaknesses and mitigations:**

*Lighting variation across the image.* Mixed lighting (sun + shadow, or indoor mixed sources) shifts apparent hue differently across the face, making a single global HSV threshold unreliable. **Mitigation:** per-image adaptive colour calibration — sample the actual HSV median of each detected colour zone and store these as per-image references rather than hardcoded ranges. Multiple samples at different angles within each zone account for gradients across the image (see §3.1).

*Arrow holes removing colour.* When arrows are pulled out they leave holes through the paper, exposing the hay behind (brown/dark). Holes cluster near the centre where scoring matters most. **Mitigation:** sample colour in a small annular patch *around* the hole rather than at its centre; exclude pixels that match hay-bale HSV (H ∈ [15°, 65°], S > 0.25) from the zone classification vote. The mode (not mean) of the patch discards outlier hole pixels.

*White zones and paper margin.* Rings 1 and 2 are white. Complicating this further, the target paper has a white margin between ring 1 and the paper edge — this margin is *inside* the boundary polygon but is not a scoring ring. White pixels alone therefore cannot identify scoring rings 1 and 2. Two detection strategies, in order of preference:

  1. **Detect the outermost black printed line.** All colour-zone transitions on a WA target are delimited by a thin printed black ring. The outermost such line marks the outer boundary of ring 1. Detect it as a radial luminance minimum outward from the blue/black/red zone group.
  2. **Extrapolate from inner ring geometry.** WA targets follow a strict proportional layout: ring radii are in the ratio 1 : 2 : 3 : … : 10 from centre. Measuring the ring width `w` from the reliably detected inner rings (gold, red, blue), extrapolate outward: ring 1 outer radius ≈ 10 w, ring 2 outer radius ≈ 9 w, ring 1/2 dividing line ≈ 9.5 w. This places rings 1 and 2 without relying on their colour or black-line detection.

*Worn or faded targets.* Older targets lose saturation; all zones tend toward grey. **Mitigation:** adaptive calibration helps; as a fallback, use ring geometry (spline boundaries §2.2) to assign zone membership.

**Verdict:** The most robust approach for **direct arrow scoring**. Adopted as the primary scoring signal. Requires the boundary mask (§3.3) to be computed first.

### 2.4 Thin-plate spline warp (non-rigid rectification)

**Representation:** A thin-plate spline (TPS) mapping from image coordinates to canonical circular coordinates, estimated from M control-point correspondences (M ≈ 20–40).

**Strengths:**
- Handles **arbitrary smooth surface deformation**, not just perspective.
- Under the TPS warp, rings become concentric circles; scoring is trivial.
- Mathematically minimises bending energy — the smoothest warp consistent with correspondences.

**Weaknesses:**
- Requires M correspondences whose canonical positions are known; automatic detection of these is as hard as the current ring detection problem.
- TPS fitting is O(M³) — feasible but non-trivial in pure JS; runtime on mobile is uncertain.
- Extrapolates poorly outside the convex hull of control points.

**Verdict:** Most general solution but requires solving a harder sub-problem to use it. Kept as a **last-resort future direction** if all simpler approaches prove insufficient.

---

## §3 Recommended approach

### 3.1 For scoring (production algorithm)

**Primary: colour-zone classification at the arrow point.**

1. Detect the boundary and compute the per-image colour calibration (see §3.3).
2. Detect the arrow tip location (see §4.1).
3. Sample HSV in a small annular patch around the tip, excluding hay-coloured pixels.
4. Classify the mode colour into gold / red / blue / black / white zone → assign score range (pair of adjacent values).
5. Disambiguate within the zone: compute the arrow's distance to the zone's inner and outer spline boundaries; the half it falls in determines the exact score.
6. For the X-ring within ring 10: use centre-distance threshold (inner ~40% of the gold zone radius); high precision is not required.

**Per-image colour calibration** is critical: because hue can shift across the image due to mixed lighting, hardcoded HSV ranges are insufficient. The algorithm should:
- Sample the colour median of each zone at multiple angular positions from the centre.
- Store these per-image references (e.g., `{ gold: [h, s, v], red: [...], ... }`).
- Use these as the zone classification thresholds for that image.
- The white zone sampled from inside the boundary provides a per-image white reference for lighting normalisation (see §4.2).

### 3.2 For ring boundary display and annotation ground truth

**Free-form closed splines (§2.2) with K ≈ 12 control points per ring.**

- Replace ellipses in the annotation tool with draggable spline control points.
- Replace `EllipseData` in the algorithm output with `SplineRing` (K control points + interpolation type).
- Keep the radial profile as the *internal* detection representation (it already is); convert to spline for storage and display by distributing K control points uniformly in angle and initialising each from the radial sample at that angle.
- For scoring, point-in-ring is tested against the spline boundary (winding number or ray cast).

### 3.3 For target boundary detection and masking

**A polygon with 4–8 vertices (a 4-anchor spline with optional mid-edge control points).**

**Critically, boundary detection is performed first, before any ring detection.** Its output polygon serves as the mask for all subsequent processing:
- Ring colour segmentation (§2.3) operates only inside this mask.
- Ring boundary detection (§2.2) is constrained to the masked region.
- Hay-bale pixels outside the mask are ignored.

**Detection strategy:**
- The hay bale has a distinctive warm-straw colour (H ∈ [15°, 65°], S > 0.2) and texture distinct from the white/coloured target face.
- Detection should look for the *hay-to-paper transition* rather than the paper edge colour specifically, because the outer white rings share hue with some backgrounds.
- Fit the 4 corner anchors first (existing `fitQuadrilateral` approach), then optionally add mid-edge handles where edge curvature is significant (detected by comparing the straight edge to the actual image boundary transition).
- 4–8 vertices are sufficient for the deformations expected in practice.

**Lighting robustness note:** White hue varies across the image under mixed lighting. The boundary detection must not rely on white uniformity; using the hay-bale colour as the "outside" signal is more reliable than using the target's white as the "inside" signal.

---

## §4 Open problems (not yet addressed)

### 4.1 Arrow detection

*Deferred — to be tackled once ring boundary detection is stable.*

The current algorithm detects ring boundaries but not arrow locations. Scoring requires both. Arrow detection approaches to investigate:

- **Shaft segmentation:** The arrow shaft is a thin, high-contrast line entering the target. Hough line detection or oriented gradient filtering could locate it.
- **Tip localisation:** The endpoint of the detected shaft segment closest to the target centre.
- **Multiple arrows:** A real end has 3–6 arrows; the algorithm must find all tips simultaneously.
- **Arrow-hole detection (post-pull):** If the photo is taken after arrows are removed, the holes (exposed hay, ~6–9 mm diameter) must be found instead of shafts. The hole cluster positions then drive scoring.

### 4.2 Lighting normalisation

Colour-zone classification depends on consistent hue perception. Confirmed approach:

- Use the white zone (rings 1–2), sampled inside the boundary mask, as a per-image white reference.
- Apply a simple von Kries-style channel scaling to shift the sampled white to a canonical white before computing all other zone HSV references.
- This handles colour temperature shifts (warm sun vs. cool shade) robustly.

### 4.3 X-ring (10 vs X)

The inner gold ring is subdivided into the 10-ring and the X-ring (used for tiebreaking in competition). They are the same colour; colour-zone scoring cannot distinguish them.

The centre of the target has minimal deformation (it is the most rigidly supported region, usually pinned directly). Computing distance from the target centre and applying a threshold (e.g., "arrow within the inner ~40% of the gold zone radius → X") is sufficient. High accuracy is not required for this distinction.

---

## §5 Algorithm migration history

| Phase | Representation | Status |
|---|---|---|
| v1 | Native C++ / OpenCV (RotatedRect ellipse fit) | Removed (see `docs/plan.md`) |
| v2 | Pure-TS ellipse fit (Fitzgibbon/Halir-Flusser) + 4-point quad boundary | Current (`targetDetection.ts`) |
| v3 (proposed) | Boundary-first polygon mask → colour-zone scoring + spline ring display | Next |
| v4 (future) | Thin-plate spline non-rigid warp for full deformation correction | If needed |

---

## §6 Explicitly excluded approaches

| Approach | Reason excluded |
|---|---|
| Homography-based canonical scoring | Corrects perspective only; does not address surface deformation; accuracy gain insufficient |
| Lens distortion correction | Excluded; the algorithm works in image space and does not assume a rectified input |
| Radial profile as annotation format | 3 600 values per image; impractical to annotate manually |

### Deferred — revisit when real failures are found

| Approach | Trigger to revisit |
|---|---|
| Perspective rejection / user notification | If specific images fail due to extreme oblique angle, work backward from the failure to define a threshold and a user-facing message |
| Photography guidelines | Same — derive constraints from observed failure modes rather than preemptively restricting the user |
