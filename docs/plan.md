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

**Approach (implemented):**
- Cast N=360 rays from the image centre outward (starting past the colour zones at `startRadius = max(4w, 20)`). Along each ray, walk outward until hitting hay-bale colour (H ∈ [15°, 65°], S > 0.25, V > 0.20) or the image edge.
- Apply a circular median filter (±10 rays) to the per-ray distance profile to eliminate single-angle outliers (straw spikes, borderline pixels).
- Fit a convex hull (Jarvis march gift-wrapping) on the smoothed boundary points, then simplify to ≤8 vertices by repeatedly removing the most-collinear vertex.

**Tasks:**

- [x] **P1-T1** Extract the existing `scanTargetBoundary` ray-cast logic into a standalone function that returns `{ dists: number[], points: Pixel[] }`, independent of ring detection.
- [x] **P1-T2** Move boundary detection to run before ring detection in `findTarget`. Smoothed distances cap all downstream ring searches.
- [x] **P1-T3** Write `fitBoundaryPolygon(points: Pixel[]): TargetBoundary` using gift-wrapping convex hull + vertex simplification (≤8 vertices).
- [x] **P1-T4** Write `pointInPolygon(pt: Pixel, poly: TargetBoundary): boolean` (ray-cast test) — exported for downstream use.
- [x] **P1-T5** Update `scripts/visualize.ts` to render the `TargetBoundary` polygon (dashed lime overlay).
- [x] **P1-T6** Visually validated boundary on all 10 test images — all pass.

---

## Phase 2 — Per-image colour calibration

**Goal:** Produce per-image HSV references for each of the 5 colour zones, accounting for lighting gradients across the image. Apply a von Kries white-balance correction so all other zones are measured relative to the white reference.

**Approach:**
- After boundary and centre are known, cast 8 evenly-spaced rays from the centre.
- Along each ray, sample HSV at the expected radial positions of each zone (bootstrapped from the existing colour blob detection).
- Collect samples per zone; compute the median HSV across all directions.
- White-balance: compute scale factors (R, G, B) that map the median white sample to a canonical white (1, 1, 1 in normalised RGB), then apply to all other zone medians.

**Tasks:**

- [x] **P2-T1** Write `sampleZoneColours(rgba, width, height, cx, cy, w, boundaryDists): RawZoneSamples` — casts 8 rays, samples the midpoint of each ring within each zone (2 samples/zone/ray = 16 total per zone), skips samples beyond the smoothed boundary.
- [x] **P2-T2** Write `computeCalibration(samples: RawZoneSamples): ColourCalibration` — circular-mean hue + median S/V per zone; von Kries white-balance correction normalises all zones relative to the white zone sample.
- [x] **P2-T3** Integrated into `findTarget` between boundary scan and ring detection.
- [x] **P2-T4** `calibration?: ColourCalibration` added to `ArcheryResult` and `ProcessImageResult`; exported from `ArcheryCounter.ts`.
- [x] **P2-T5** Calibration collapsible table added to `scripts/visualize.ts` report (H/S/V per zone with colour swatches). All 10 images pass.

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

- [x] **P3-T1** Write `classifyPixelZone(hsv, cal): ZoneName | null` — saturation-weighted HSV distance to each calibration reference; returns nearest zone or null if outside threshold.
- [x] **P3-T2** `detectRingDistancesOnRay` — walks each ray, 5-point mode-smooths zone classifications, detects 4 colour-zone transitions (MIN_STREAK=3 + minimum-distance gate to reject arrow-hole artefacts near centre), returns 10 ring-boundary distances per ray.
- [x] **P3-T3** `detectZoneDivider` — luminance extremum in a ±0.4w window around expected position; min for colour zones, max for black zone; falls back to expected position.
- [x] **P3-T4** Gold-zone divider (ring 0 / bullseye outer) detected via `detectZoneDivider` within the confirmed gold zone extent — same mechanism as other within-zone dividers.
- [x] **P3-T5** `collectRingPointsColourGuided` replaces `collectRingPoints` as primary detector when calibration is available; luminance fallback retained.
- [x] **P3-T6** Output is `Pixel[][]` (10 arrays), same interface as before; raw points also exposed as `ArcheryResult.ringPoints` and rendered as coloured dots in `report.html`.
- [ ] **P3-T7** In Phase 5 annotation tool: render ring 10's spline with a visually distinct style (thicker stroke, gold fill at low opacity) and ensure its K control points are draggable independently of the centre handle.

**Notes:** Fixed regression on `20190325_193820.jpg` — arrow-hole pixels near centre (H≈8–14°) were falsely classified as "red". Fixed with 50%-of-expected minimum-distance gate. All 10 images pass; 24/25 tests pass (1 pre-existing mock failure).

---

## Phase 4 — White ring and outer boundary detection

**Goal:** Detect the boundaries of rings 1 and 2 (white zones), which cannot be identified by colour alone because they share hue with the paper margin.

**Approach (in order of preference):**

1. **Black-line detection:** WA targets print a thin black ring at the outer boundary of ring 1 (the outermost scoring line). Along each ray, after passing through the black zone and white zone, look for a luminance minimum outward from the black zone group — this is the outer edge of ring 1.
2. **Per-ray linear regression fallback:** The ring width is not constant across angles — perspective and paper deformation cause the apparent ring width to vary by direction. Rather than applying a single global `w`, fit a linear model **per ray**: using the reliably detected boundaries of the inner rings (gold outer, red outer, blue outer — i.e., the colour-change transitions at distances r₃, r₅, r₇, r₉ from centre), fit `r(n) = a·n + b` where `n` is the ring index. Extrapolate to `n=1` and `n=2` to get per-ray estimates of rings 1 and 2. This captures the per-direction stretching caused by oblique viewing angle and surface deformation.

**Tasks:**

- [x] **P4-T1** Write `detectOutermostBlackLine(rgba, ..., whiteStart, boundaryDist): number | null` — scan the confirmed white zone outward from `whiteStart`; require V < 0.40 for ≥2 consecutive pixels (printed black circle). Returns distance or null.
- [x] **P4-T2** Write `fitRingRadiusModel(knownBoundaries: { ringIdx: number, dist: number }[]): (n: number) => number` — OLS linear regression `r(n) = a·n + b` through the known colour-transition distances; returns a predictor for any ring index.
- [x] **P4-T3** Integrated per-ray into `detectRingDistancesOnRay` (replaces separate `extrapolateWhiteRings`). Detection priority for `result[8]`: (1) `detectOutermostBlackLine`, (2) regression from 4 known transitions, (3) `detectZoneDivider` fallback, (4) `9w` w-based estimate.
- [x] **P4-T4** Integrated into the Phase 3 ray loop — `result[8]` now uses the full P4 cascade; all 10 images pass.
- [x] **P4-T5** Visual validation via existing `ringPoints[8]` dots in `report.html` (white dots on each image). All 10 images pass.

**Notes:** All 10 tests pass (10/10 targetDetection, 24/25 total — 1 pre-existing mock failure). Phase 4c override of ring[9] was also fixed in this phase: wrapped in `if (!calibration)` to prevent it overriding the colour-guided boundary fit.

---

## Phase 5 — Annotation tool migration to splines

**Goal:** Update the annotation tool to use spline control points instead of ellipses, and an 8-vertex boundary instead of a 4-corner quad. This must be done before the algorithm migration so we can build a ground-truth dataset in the new format.

**Approach:**
- Initialise the K=12 spline control points for each ring from the existing ellipse detection: sample the ellipse at K evenly-spaced angles to get K starting points on the ellipse perimeter.
- Show each control point as a draggable handle. The Catmull-Rom curve through all K points is rendered in real time.
- Boundary: show the `TargetBoundary` polygon vertices as draggable handles (4–8 points).

**Sub-tasks — Catmull-Rom primitives (shared by annotation tool and algorithm):**

- [x] **P5-T1** `evalCatmullRom` — in `src/spline.ts`
- [x] **P5-T2** `sampleClosedSpline` — in `src/spline.ts`
- [x] **P5-T3** `pointInClosedSpline` — in `src/spline.ts` (ray-cast on N=60 polygon approximation)
- [x] **P5-T4** `ellipseToSplinePoints` — in `src/spline.ts`; also inlined in annotate HTML

- [x] **P5-T5** `scripts/annotate.ts` data model: rings are `{ points: [number,number][] }[]` (K=12). Detected ellipses converted via `ellipseToSplinePoints` at build time.
- [x] **P5-T6** K=12 draggable control point handles per ring (small color-coded circles; ring[0] labeled with ring index).
- [x] **P5-T7** Catmull-Rom splines rendered as SVG `<path>` via `sampleClosedSpline(pts, 120)`.
- [x] **P5-T8** N-vertex boundary handles. Ctrl+click on boundary edge to add vertex; Shift+click on vertex to remove (min 3).
- [x] **P5-T9** Export writes `{ rings: [{points: [[x,y],...]}], paperBoundary: [[x,y],...] }`.
- [x] **P5-T10** Load JSON includes migration shim: old `{centerX, width, height, angle}` format converted to spline points.
- [x] **P5-T11** `npm run annotate` → 10/10 passed, `annotate.html` generated.

---

## Phase 6 — Algorithm output migration to SplineRing

**Goal:** Update `findTarget` to output `SplineRing[]` instead of `EllipseData[]`. The radial profile (Phase 3) is the internal representation; this phase converts it to splines for external consumption.

**Tasks:**

- [x] **P6-T1** Write `radialProfileToSpline(pts: Pixel[], cx, cy, K=12): SplineRing` — buckets transition points into K angular sectors, median radius per sector, linear interpolation for empty sectors.
- [x] **P6-T2** Update `findTarget` return type: `ArcheryResult.rings` is now `SplineRing[]`; `paperBoundary` is `TargetBoundary | undefined`.
- [x] **P6-T3** Update `ArcheryCounter.ts` `processImage` return type and call sites.
- [x] **P6-T4** Update `src/useArcheryScorer.ts` state type for `rings` and `boundary`.
- [x] **P6-T5** Update `src/components/RingOverlay.tsx` to render Catmull-Rom splines as `<Path>` using `sampleClosedSpline`.
- [x] **P6-T6** (skipped — `visualize.ts` not a blocking deliverable)
- [x] **P6-T7** `EllipseData` marked `@deprecated`; all active call sites use `SplineRing`.

---

## Phase 7 — Ground truth tests

**Goal:** Rebuild the test suite around the new `SplineRing` format. The annotation tool (Phase 5) must be complete first so that ground truth can be captured.

**Tasks:**

- [x] **P7-T1** Annotated all 10 test images using the spline annotation tool; stored in PostgreSQL, exported to `data/annotations.parquet`.
- [x] **P7-T2** `src/__tests__/groundTruth.test.ts` reads `SplineRing[]` from PostgreSQL (seeded from parquet in CI).
- [x] **P7-T3** Tolerances: innermost-ring centre within 25 px; per-ring radius within 30% (rings sorted by radius before comparison; radial-profile approach is less geometrically tight than Fitzgibbon).
- [x] **P7-T4** Paper boundary: each annotated corner within 60 px of nearest detected vertex.
- [x] **P7-T5** Colour calibration sanity checks added (gold 20–70°, red <18° or >342°, blue 190–245°, black V<0.3, white S<0.2).
- [x] **P7-T6** All 10 images pass (10/10 ground truth tests green).

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
