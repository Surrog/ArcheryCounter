# ArcheryCounter — Implementation Plan (v3)

See `docs/research.md` for the rationale behind every decision taken here.

---

## Architecture overview

The v3 pipeline replaces the current ellipse-fit approach with a boundary-first, colour-driven pipeline:

```
Photo
  │
  ▼
[Phase 1] Boundary detection
  → TargetBoundary (4–8 vertex polygon)
  │
  ▼
[Phase 2] Colour calibration (inside boundary mask)
  → ColourCalibration (per-image HSV references per zone)
  │
  ▼
[Phase 3] Target centre bootstrap
  → (cx, cy)                         ← already works; runs inside mask
  │
  ▼
[Phase 4] Radial ring detection (inside boundary, guided by colour calibration)
  → RadialProfile (360 × 10 distances)   ← internal only, not stored
  │
  ▼
[Phase 5] White ring boundaries
  → extend RadialProfile rings 1–2 via black-line detection or extrapolation
  │
  ▼
[Phase 6] Convert radial profile → SplineRing[10]
  → SplineRing[] (12 control points per ring, Catmull-Rom)
  │
  ▼
Output: { boundary: TargetBoundary, rings: SplineRing[], calibration: ColourCalibration }
```

Arrow scoring (deferred, depends on arrow detection):
```
Arrow tip pixel p
  → colour zone classification (using ColourCalibration)
  → within-zone disambiguation (distance to inner/outer SplineRing)
  → score (1–10) + X flag
```

---

## New data types

These replace `EllipseData` and the `[Pixel, Pixel, Pixel, Pixel]` boundary tuple.

```typescript
/** A single closed ring boundary represented as K Catmull-Rom control points. */
interface SplineRing {
  /** Control points in image pixel coordinates, ordered clockwise. K ≈ 12. */
  points: [number, number][];
  /** Interpolation type (only 'catmull-rom' supported for now). */
  interpolation: 'catmull-rom';
}

/** Target paper boundary as an ordered polygon (4–8 vertices, clockwise). */
interface TargetBoundary {
  points: [number, number][];
}

/** Per-image HSV colour references, one median sample per zone. */
interface ColourCalibration {
  gold:  [number, number, number]; // H 0–360, S 0–1, V 0–1
  red:   [number, number, number];
  blue:  [number, number, number];
  black: [number, number, number];
  white: [number, number, number];
}

/** Updated top-level result. */
interface TargetResult {
  success: true;
  boundary: TargetBoundary;
  rings: SplineRing[];          // index 0 = bullseye, index 9 = outermost
  calibration: ColourCalibration;
  centre: [number, number];
}
```

---

## Phase 1 — Boundary detection

**Goal:** Detect the target paper boundary using the hay-bale colour as the "outside" signal. Output a polygon accurate enough to serve as a mask for all subsequent processing.

**Approach:**
- Cast N rays from the image centre outward. Along each ray, walk outward until hitting hay-bale colour (H ∈ [15°, 65°], S > 0.2) or the image edge. This gives one boundary point per ray.
- Fit 4 corner anchors using the existing axis-aligned diagonal projection (`fitQuadrilateral`).
- Check each edge for curvature: compare the straight line between two adjacent corners to the actual boundary points along that edge. If max deviation > threshold (e.g., 5% of image width), insert a midpoint vertex.
- Repeat until no edge deviates significantly or 8 vertices are reached.

**Tasks:**

- [ ] **P1-T1** Extract the existing `scanTargetBoundary` ray-cast logic into a standalone function that returns only `{ points: Pixel[] }` (one point per ray at the hay transition), independent of ring detection.
- [ ] **P1-T2** Move boundary detection to run before any ring/colour logic in `findTarget`. Pass the resulting mask to all downstream phases.
- [ ] **P1-T3** Write `fitBoundaryPolygon(points: Pixel[]): TargetBoundary` that: (a) fits 4 corner anchors via axis-aligned diagonal projection, (b) checks each edge for curvature and inserts midpoint vertices where deviation > threshold, up to 8 vertices total.
- [ ] **P1-T4** Write `pointInPolygon(pt: Pixel, poly: TargetBoundary): boolean` (ray-cast test) — used to mask all downstream pixel access.
- [ ] **P1-T5** Update `scripts/visualize.ts` to render the new `TargetBoundary` polygon (replace the existing 4-point polygon drawing).
- [ ] **P1-T6** Visually validate boundary on all 10 test images in `report.html`. Annotate any failures in `images/README.md` (create if absent).

---

## Phase 2 — Per-image colour calibration

**Goal:** Produce per-image HSV references for each of the 5 colour zones, accounting for lighting gradients across the image. Apply a von Kries white-balance correction so all other zones are measured relative to the white reference.

**Approach:**
- After boundary and centre are known, cast 8 evenly-spaced rays from the centre.
- Along each ray, sample HSV at the expected radial positions of each zone (bootstrapped from the existing colour blob detection).
- Collect samples per zone; compute the median HSV across all directions.
- White-balance: compute scale factors (R, G, B) that map the median white sample to a canonical white (1, 1, 1 in normalised RGB), then apply to all other zone medians.

**Tasks:**

- [ ] **P2-T1** Write `sampleZoneColours(rgba, width, height, cx, cy, approxRingWidth, boundary): RawZoneSamples` — casts 8 rays, samples ~5 pixels per expected zone position per ray, returns raw HSV arrays per zone.
- [ ] **P2-T2** Write `computeCalibration(samples: RawZoneSamples): ColourCalibration` — takes the median per zone, applies von Kries white-balance correction relative to the white zone sample.
- [ ] **P2-T3** Integrate `sampleZoneColours` + `computeCalibration` into the `findTarget` pipeline, running after boundary + centre are established.
- [ ] **P2-T4** Add `calibration: ColourCalibration` to `ArcheryResult` (success branch) and expose it from `ArcheryCounter.processImage`.
- [ ] **P2-T5** Add calibration data to `scripts/visualize.ts` report (collapsible table per image showing the 5 zone HSV references).

---

## Phase 3 — Radial ring detection (colour-guided)

**Goal:** Replace the current luminance-transition-based ring detection with colour-zone-boundary detection guided by the per-image calibration. Detect where adjacent colour zones meet along each ray.

**Approach:**
- Cast N=360 rays from centre, capped at the boundary polygon.
- Along each ray, classify each pixel to its nearest colour zone using the calibration.
- Find the first pixel where the classification changes between adjacent zones (gold→red, red→blue, blue→black, black→white). These transitions correspond to ring boundaries 2/3, 4/5, 6/7, 8/9 (the colour-change boundaries).
- The dividing lines within each zone (e.g., rings 9 and 10 within gold) are detected as the midpoint between the inner and outer colour transitions of that zone, OR as a luminance minimum (the thin black printed divider).

**Special case — ring 10 (bullseye / innermost ring):**

Ring 10 is the most critical ring for scoring and also the hardest to detect reliably:
- It has **no inner colour transition** — the gold zone simply converges toward the target centre.
- It is the most **arrow-damaged** region: heavy arrow traffic tears and discolours the paper precisely where ring 10 needs to be measured.
- Its inner boundary (the X-ring printed line) is a very thin circle and may be faint or absent on worn targets.

Detection strategy for ring 10's inner boundary:
1. Look for the X-ring printed luminance minimum within the gold zone along each ray.
2. If no clear minimum is found, use the ring-width extrapolation from Phase 4 (linear regression on detected inner-zone boundaries) to predict the ring 10 inner radius at that angle.
3. The detected/extrapolated ring 10 inner boundary must be annotatable — it is a full spline ring in the annotation tool, not a single point.

**Tasks:**

- [ ] **P3-T1** Write `classifyPixelZone(hsv: [h,s,v], cal: ColourCalibration): Zone | null` — returns 'gold' | 'red' | 'blue' | 'black' | 'white' | null (hay/outside) using nearest-zone HSV distance.
- [ ] **P3-T2** Write `detectColourTransitions(rgba, width, height, cx, cy, cal, boundary): number[][]` — for each of 360 rays, returns the distances at which colour-zone transitions occur (4 major transitions + midpoints for minor dividers).
- [ ] **P3-T3** Write `detectZoneDivider(ray: RayPixels, zoneStart: number, zoneEnd: number): number` — find the luminance minimum between two zone-boundary distances; fall back to the midpoint if no clear minimum found.
- [ ] **P3-T4** Write `detectRing10Inner(ray: RayPixels, goldStart: number, goldMid: number): number` — scan for the X-ring luminance minimum between centre and `goldMid`; fall back to Phase 4 linear regression estimate if no minimum found.
- [ ] **P3-T5** Replace the existing `collectRingPoints` function with the colour-guided transition detection from P3-T2/T3/T4.
- [ ] **P3-T6** Ensure the output is still a 360×10 radial profile (`number[][]`) for compatibility with the spline conversion (Phase 6).
- [ ] **P3-T7** In Phase 5 annotation tool: render ring 10's spline with a visually distinct style (thicker stroke, gold fill at low opacity) and ensure its K control points are draggable independently of the centre handle.

---

## Phase 4 — White ring and outer boundary detection

**Goal:** Detect the boundaries of rings 1 and 2 (white zones), which cannot be identified by colour alone because they share hue with the paper margin.

**Approach (in order of preference):**

1. **Black-line detection:** WA targets print a thin black ring at the outer boundary of ring 1 (the outermost scoring line). Along each ray, after passing through the black zone and white zone, look for a luminance minimum outward from the black zone group — this is the outer edge of ring 1.
2. **Per-ray linear regression fallback:** The ring width is not constant across angles — perspective and paper deformation cause the apparent ring width to vary by direction. Rather than applying a single global `w`, fit a linear model **per ray**: using the reliably detected boundaries of the inner rings (gold outer, red outer, blue outer — i.e., the colour-change transitions at distances r₃, r₅, r₇, r₉ from centre), fit `r(n) = a·n + b` where `n` is the ring index. Extrapolate to `n=1` and `n=2` to get per-ray estimates of rings 1 and 2. This captures the per-direction stretching caused by oblique viewing angle and surface deformation.

**Tasks:**

- [ ] **P4-T1** Write `detectOutermostBlackLine(ray: RayPixels, blackZoneEnd: number, boundaryDist: number): number | null` — scan outward from the black zone end, find the first luminance minimum before the boundary; return the distance or null if not found.
- [ ] **P4-T2** Write `fitRingRadiusModel(knownBoundaries: { ringIdx: number, dist: number }[]): (n: number) => number` — fits a linear regression `r(n) = a·n + b` through the known ring boundary distances for one ray; returns a function that predicts the boundary distance for any ring index.
- [ ] **P4-T3** Write `extrapolateWhiteRings(radialProfile: number[][], knownBoundariesPerRay: number[][][]): void` — for each ray, call `fitRingRadiusModel` using the 4 reliably detected colour-transition boundaries, then use the model to fill in rings 1 and 2 for any ray where `detectOutermostBlackLine` returned null.
- [ ] **P4-T4** Integrate P4-T1 through P4-T3 into the ray-by-ray loop from Phase 3, completing all 10 ring boundaries.
- [ ] **P4-T5** Add visual validation in `scripts/visualize.ts`: overlay the detected white ring boundaries on the report images. Confirm correct placement on all test images.

---

## Phase 5 — Annotation tool migration to splines

**Goal:** Update the annotation tool to use spline control points instead of ellipses, and an 8-vertex boundary instead of a 4-corner quad. This must be done before the algorithm migration so we can build a ground-truth dataset in the new format.

**Approach:**
- Initialise the K=12 spline control points for each ring from the existing ellipse detection: sample the ellipse at K evenly-spaced angles to get K starting points on the ellipse perimeter.
- Show each control point as a draggable handle. The Catmull-Rom curve through all K points is rendered in real time.
- Boundary: show the `TargetBoundary` polygon vertices as draggable handles (4–8 points).

**Sub-tasks — Catmull-Rom primitives (shared by annotation tool and algorithm):**

- [ ] **P5-T1** Write `evalCatmullRom(p0, p1, p2, p3, t): [number, number]` — evaluates one segment at parameter t ∈ [0,1].
- [ ] **P5-T2** Write `sampleClosedSpline(points: [number, number][], nSamples: number): [number, number][]` — samples a closed Catmull-Rom spline evenly. Used for rendering and point-in-ring tests.
- [ ] **P5-T3** Write `pointInClosedSpline(pt, points): boolean` — winding number test using a polygon approximation sampled from the spline (N=60 samples is sufficient).
- [ ] **P5-T4** Write `ellipseToSplinePoints(cx, cy, rx, ry, angleDeg, K=12): [number, number][]` — samples a rotated ellipse at K evenly-spaced angles to produce K control points; used as the initial spline approximation from the existing ellipse output.

**Sub-tasks — Annotation tool:**

- [ ] **P5-T5** In `scripts/annotate.ts` data model: change `rings` from `EllipseData[]` to `{ points: [number,number][] }[]` (K control points per ring). On load, convert existing ellipse output to control points using `ellipseToSplinePoints`.
- [ ] **P5-T6** Replace the rx/ry/rot per-ring handles with K draggable control point handles per ring. Each control point is a small circle; dragging it moves that point.
- [ ] **P5-T7** Render the Catmull-Rom spline for each ring in the SVG overlay using `sampleClosedSpline` (polyline through sampled points).
- [ ] **P5-T8** Replace the 4-corner boundary handles with N-vertex handles (one handle per vertex of `TargetBoundary`). Support adding/removing vertices (right-click or dedicated UI button).
- [ ] **P5-T9** Update the `Export JSON` function to write `SplineRing[]` format (array of `{ points: [[x,y],...] }`) and `TargetBoundary` format.
- [ ] **P5-T10** Update the `Load JSON` function to read the new format, with a migration shim for old ellipse-based annotations.
- [ ] **P5-T11** Re-run `npm run annotate` to regenerate `annotate.html` and validate visually.

---

## Phase 6 — Algorithm output migration to SplineRing

**Goal:** Update `findTarget` to output `SplineRing[]` instead of `EllipseData[]`. The radial profile (Phase 3) is the internal representation; this phase converts it to splines for external consumption.

**Tasks:**

- [ ] **P6-T1** Write `radialProfileToSpline(profile: number[][], cx: number, cy: number, ringIdx: number, K=12): SplineRing` — distributes K control points uniformly in angle, each at the corresponding distance from the profile.
- [ ] **P6-T2** Update `findTarget` return type: replace `EllipseData[]` with `SplineRing[]` and `[Pixel,Pixel,Pixel,Pixel] | undefined` with `TargetBoundary | undefined`.
- [ ] **P6-T3** Update `ArcheryCounter.ts` `processImage` return type and call sites.
- [ ] **P6-T4** Update `src/useArcheryScorer.ts` state type for `rings` and `boundary`.
- [ ] **P6-T5** Update `src/components/RingOverlay.tsx` to render Catmull-Rom splines instead of `<Ellipse>` elements — use `sampleClosedSpline` and render as `<Polyline>` or `<Path>`.
- [ ] **P6-T6** Update `scripts/visualize.ts` to render splines instead of SVG `<ellipse>` elements.
- [ ] **P6-T7** Remove all `EllipseData` references from the codebase.

---

## Phase 7 — Ground truth tests

**Goal:** Rebuild the test suite around the new `SplineRing` format. The annotation tool (Phase 5) must be complete first so that ground truth can be captured.

**Tasks:**

- [ ] **P7-T1** Annotate all 10 test images in `annotate.html` using the new spline handles. Export `annotations.json`.
- [ ] **P7-T2** Update `src/__tests__/groundTruth.test.ts` to load `SplineRing[]` from annotations.
- [ ] **P7-T3** Define tolerances for spline tests: each detected control point must be within N px of the annotated control point at the same angle (suggested: 30 px; tighten based on results).
- [ ] **P7-T4** Add a `boundaryTest`: each boundary vertex within 60 px of the annotated vertex (same as current tolerance).
- [ ] **P7-T5** Add a colour calibration sanity test: the 5 detected zone HSV references should match their expected canonical hue ranges (gold 20–70°, red 340–18°, blue 190–245°, black V < 0.3, white S < 0.2).
- [ ] **P7-T6** Run full test suite, review failures, and tighten or adjust algorithm until all 10 images pass.

---

## Phase 8 — Scoring pipeline (deferred — needs arrow detection)

*Do not implement until Phase 9 (arrow detection) is underway.*

**Tasks:**

- [ ] **P8-T1** Write `classifyColourZone(hsv: [h,s,v], cal: ColourCalibration): ColourZone | null` — returns 'gold' | 'red' | 'blue' | 'black' | 'white' | null.
- [ ] **P8-T2** Write `samplePatchZone(rgba, width, height, pt, radius, cal): ColourZone | null` — samples a small annular patch around a point, excludes hay-coloured pixels, returns the modal zone.
- [ ] **P8-T3** Write `disambiguateScore(zone: ColourZone, pt: Pixel, innerRing: SplineRing, outerRing: SplineRing): number` — computes distances to inner and outer spline boundaries, returns the score (higher or lower of the two adjacent scores in the zone).
- [ ] **P8-T4** Write `isXRing(pt: Pixel, centre: [number, number], goldInnerSpline: SplineRing): boolean` — returns true if pt is within the inner 40% of the gold zone radius.
- [ ] **P8-T5** Write `scoreArrow(rgba, width, height, arrowTip: Pixel, result: TargetResult): number | 'X' | 0` — orchestrates P8-T1 through P8-T4.
- [ ] **P8-T6** Add `scoreArrow` to `ArcheryCounter.processImage` API once arrow tips are available.

---

## Phase 9 — Arrow detection (fully deferred)

*Tackle once all ring and boundary detection is stable and tested.*

**Tasks:**

- [ ] **P9-T1** Research and prototype shaft detection: try Hough line transform on a preprocessed (edge-detected) version of the target region.
- [ ] **P9-T2** Implement shaft endpoint localisation (tip = endpoint of shaft segment closest to target centre).
- [ ] **P9-T3** Handle multiple arrows: detect all shaft lines within the target boundary, deduplicate nearby detections.
- [ ] **P9-T4** Implement arrow-hole mode: if no shafts are detected, find small dark circular regions (hay exposed through paper holes, ~6–9 px at 1200 px scale) as candidate arrow positions.
- [ ] **P9-T5** Wire arrow detection into the full scoring pipeline (Phase 8).
- [ ] **P9-T6** Collect images with arrows in place (or holes) to build a test dataset.

---

## Task summary by dependency order

```
P1 (Boundary) ──► P2 (Calibration) ──► P3 (Ring detection) ──► P4 (White rings)
                                                                        │
                                                                        ▼
P5 (Annotate tool) ◄──────────────────────────────────────────── P6 (Algorithm output)
       │                                                                │
       ▼                                                                ▼
P7 (Tests) ◄────────────────────────────────────────────────────────────
       │
       (all above done)
       │
       ▼
P8 (Scoring) ◄── P9 (Arrow detection)
```

Phases 1–4 can proceed in parallel with Phase 5 (annotation tool), since Phase 5 initialises from the existing ellipse output. Phases 6 and 7 require Phases 1–5 to be complete.
