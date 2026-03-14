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

### 2.3 Homography-based canonical scoring

**Representation:** A 3×3 projective matrix H that maps image coordinates to a canonical circular target (e.g., 500 px radius, concentric circles at WA standard ratios).

**Strengths:**
- Under H, rings are perfect concentric circles → scoring is `distance_from_centre / ring_radius`.
- Handles arbitrary camera angle (pure perspective) perfectly.
- Only 8 degrees of freedom (H is defined up to scale).

**Weaknesses:**
- Only corrects **perspective**, not surface deformation. A bowed target still produces non-circular rings in canonical space.
- Requires knowing correspondences: at least 4 points whose canonical positions are known (e.g., the four outermost ring intersections at cardinal directions).
- Automatic detection of those correspondences is as hard as the current ring detection.

**Verdict:** Theoretically elegant for flat targets at an angle. In practice, surface deformation limits accuracy exactly where it matters most. Useful as a **pre-processing step** (rectify the image before ring detection) rather than a complete solution.

### 2.4 Color-zone segmentation (direct colour scoring)

**Representation:** Instead of storing ring *boundaries*, classify each image pixel as belonging to a colour zone: gold (9–10), red (7–8), blue (5–6), black (3–4), white (1–2), or background.

**Scoring:** To score an arrow at pixel p, look up the colour zone at p (or average a small neighbourhood around the arrow tip).

**Strengths:**
- Completely bypasses the geometry problem. Colour is printed on the target; deformation moves the colour with the paper — no modelling needed.
- Extremely robust: a bowed, creased, folded face still has its correct colours.
- The algorithm already has colour blob detection (Phase 2 in `targetDetection.ts`); extending it to full segmentation is natural.
- No ring boundary needed for scoring — just colour at the arrow point.

**Weaknesses:**
- Colour accuracy depends on lighting. Mixed light (sun + shadow across the face) shifts apparent hue.
- Worn or dirty targets lose saturation; older targets fade to similar hues.
- Does not give a boundary representation — hard to display ring overlays from colour alone.
- The X-ring (innermost gold subdivision, 10 vs X) has no distinct colour; requires geometry.

**Verdict:** The most robust approach for **direct arrow scoring**. Should be primary scoring signal, with geometry as a fallback for ambiguous colours (worn targets) and for the X-ring.

### 2.5 Thin-plate spline warp (non-rigid rectification)

**Representation:** A thin-plate spline (TPS) mapping from image coordinates to canonical circular coordinates, estimated from M control-point correspondences (M ≈ 20–40; e.g., ring boundary intersections at multiple angles).

**Strengths:**
- Handles **arbitrary smooth surface deformation**, not just perspective.
- Under the TPS warp, rings become concentric circles; scoring is trivial.
- Mathematically minimises bending energy — the smoothest warp consistent with correspondences.

**Weaknesses:**
- Requires M correspondences; estimating them automatically is the hard part.
- TPS fitting is O(M³) and involves solving a linear system — feasible but non-trivial in pure JS.
- Extrapolates poorly outside the convex hull of control points.

**Verdict:** Most general solution; worth pursuing once robust automatic control-point detection exists. For now, better suited as a **future direction**.

---

## §3 Recommended approach

### 3.1 For scoring (production algorithm)

**Primary: colour-zone classification at the arrow point.**

1. Detect the arrow tip location in the image (separate problem; see §4).
2. Sample HSV colour in a small region around the tip.
3. Classify into gold / red / blue / black / white zone → assign score range.
4. Use measured ring geometry (radial profile) to disambiguate within a zone (e.g., 9 vs 10, both gold).

This is robust to all paper deformations because it reads the printed information directly.

### 3.2 For ring boundary display and annotation ground truth

**Free-form closed splines (§2.2) with K ≈ 12 control points per ring.**

- Replace ellipses in the annotation tool with draggable spline control points.
- Replace ellipses in the algorithm output (`EllipseData`) with `SplineRing` (K control points + interpolation type).
- Keep the radial profile as the internal detection representation (it already is); convert to spline for storage and display.
- For automated detection, distribute K control points uniformly in angle from the centre and initialise each one from the radial profile sample at that angle.

### 3.3 For target boundary display

**Multi-segment polygon (8 points per edge, 32 total) or a 4-anchor spline.**

- Replace the 4-vertex quadrilateral with a cubic spline through 4 corner anchors + 1–2 midpoint handles per edge (12–16 total control points).
- This can represent a bowing edge (concave/convex) and corner curl.
- For automated detection: fit the 4 corners first (existing `fitQuadrilateral`), then detect midpoint deviations from straight edges using edge detection along the boundary.

---

## §4 Open problems (not yet addressed)

### 4.1 Arrow detection

The current algorithm detects ring boundaries but not arrow locations. Scoring requires both. Arrow detection is a distinct computer-vision problem:

- **Shaft segmentation:** The arrow shaft is a thin, high-contrast line entering the target. Hough line detection or oriented gradient filtering could locate it.
- **Tip localisation:** The tip is where the shaft meets the target face. Defined as the endpoint of the detected shaft segment closest to the target centre.
- **Multiple arrows:** A real end has 3–6 arrows; the algorithm must find all tips simultaneously.

### 4.2 X-ring (10 vs X)

The inner gold ring is subdivided into the 10-ring and X-ring (for tiebreaking). They are the same colour; distinction requires geometric measurement. A reliable centre estimate + the innermost spline ring boundary (§3.2) would handle this.

### 4.3 Lighting normalisation

Colour-zone classification (§3.1) is sensitive to colour temperature and shadows cast by arrows. Adaptive white-balance correction (e.g., using the white ring as a reference) before colour classification would improve robustness.

### 4.4 Camera calibration / lens distortion

Wide-angle phone cameras introduce barrel distortion, which warps the apparent shape of the target. Correcting for lens distortion before any geometric processing (using the phone's camera calibration data, available on iOS/Android) would reduce ring detection error, especially near image edges.

---

## §5 Algorithm migration history

| Phase | Representation | Status |
|---|---|---|
| v1 | Native C++ / OpenCV (RotatedRect ellipse fit) | Removed (see `docs/plan.md`) |
| v2 | Pure-TS ellipse fit (Fitzgibbon/Halir-Flusser) + 4-point quad boundary | Current (`targetDetection.ts`) |
| v3 (proposed) | Radial profile detection → spline ring output + 12-point boundary spline | Next |
| v4 (future) | Colour-zone scoring + TPS warp for canonical space | Future |
