# Detection Algorithm — Implementation Plan

Based on the failure analysis in `docs/research.md`. Replaces the old C++/OpenCV port
with a three-phase pipeline: **adaptive color detection → radial-profile ring measurement
→ algebraic ellipse fitting**.

---

## New pipeline overview

```
rgba (original, un-pretreated) ─────────────────────────────────────────┐
                                                                         │ (used for radial sampling)
rgba → pretreat() ──────────────────────────────────────────────────┐   │
                                                                     │   │
                  ┌──────────────────────────────────────────────┐  │   │
                  │  PHASE 1 — Adaptive colour detection          │  │   │
                  │                                               │  │   │
                  │  for each colour (yellow, red, blue):         │◄─┘   │
                  │    applyHsvFilter(wide range)                 │      │
                  │    → aggregateBlobs()                         │      │
                  │    if blob too small: return null             │      │
                  │    computeMedianHue(blob) → re-filter ±22°   │      │
                  │    → aggregateBlobs() again                   │      │
                  │    → ColorBlob { centroid, meanRadius }       │      │
                  └──────────────────┬───────────────────────────┘      │
                                     │ blobs: { yellow?, red?, blue? }  │
                                     ▼                                   │
                  ┌──────────────────────────────────────────────┐      │
                  │  PHASE 2 — Centre + scale bootstrap           │      │
                  │                                               │      │
                  │  cx0, cy0 = mean of blob centroids            │      │
                  │  for each blob: w_i = meanRadius / RATIO[c]  │      │
                  │  w0 = median(w_i)                            │      │
                  │  angle0 = from blob with most boundary pts   │      │
                  └──────────────────┬───────────────────────────┘      │
                                     │ { cx0, cy0, w0, angle0 }         │
                                     ▼                                   │
                  ┌──────────────────────────────────────────────┐      │
                  │  PHASE 3 — Radial profile ring measurement    │      │
                  │                                               │      │
                  │  cast N=360 rays from (cx0, cy0)             │◄─────┘
                  │  along each ray: sample pixels, detect        │
                  │    luminance transitions                       │
                  │  for each ring k (0-9):                       │
                  │    expected radius ≈ (k+1) × w0              │
                  │    collect transition pts within ±0.4×w0     │
                  │  → transitionPoints[10]: Pixel[][]           │
                  │                                               │
                  │  for each ring k with ≥ 6 points:            │
                  │    fitEllipseFitzgibbon(transitionPoints[k])  │
                  │  for rings with < 6 points:                   │
                  │    interpolateMissingRings() from neighbours  │
                  └──────────────────┬───────────────────────────┘
                                     │
                               EllipseData[10]
```

### Key changes vs. current implementation

| Current | New |
|---------|-----|
| `rgbToHsvFull` (BGR swap quirk) | `rgbToHsv` (standard RGB→HSV) |
| Fixed narrow HSV thresholds (3–4 hue units) | Wide ranges + per-image adaptive re-centering |
| `findWeakestElement` heuristic | Dropped — no longer needed |
| `linearInterpolateTo10Rings` (least-squares) | `radialProfileToRings` (measurement-based) |
| `fitEllipsePCA` (fragile on partial arcs) | `fitEllipseFitzgibbon` (algebraically constrained) |
| Radial sampling from colour-pretreated image | Radial sampling from **original** image (sharp transitions) |

---

## Phase 1 — Adaptive colour detection

### 1a. Standard HSV conversion

Replace `rgbToHsvFull` with a clean standard conversion. H in degrees 0–360, S and V
in 0–1.

```typescript
function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const r1 = r / 255, g1 = g / 255, b1 = b / 255;
  const max = Math.max(r1, g1, b1);
  const min = Math.min(r1, g1, b1);
  const d   = max - min;
  const v   = max;
  const s   = max === 0 ? 0 : d / max;
  let h = 0;
  if (d > 0) {
    if      (max === r1) { h = 60 * (((g1 - b1) / d) % 6); }
    else if (max === g1) { h = 60 * ((b1 - r1) / d + 2);   }
    else                 { h = 60 * ((r1 - g1) / d + 4);   }
    if (h < 0) h += 360;
  }
  return [h, s, v];
}
```

### 1b. Wide initial HSV ranges

```typescript
// H in degrees (0-360), S/V in 0-1
const COLOUR_RANGES = {
  yellow: { hRanges: [[20, 70]],            sMin: 0.30, vMin: 0.30 },
  red:    { hRanges: [[0, 18], [342, 360]], sMin: 0.30, vMin: 0.20 },  // wraps around 0°
  blue:   { hRanges: [[190, 245]],          sMin: 0.25, vMin: 0.20 },
};
```

Red needs two sub-ranges because hue wraps at 360°/0°.

### 1c. Two-pass adaptive detection

```typescript
interface ColorBlob {
  centroid:   Pixel;       // area-weighted mean position ≈ target centre
  meanRadius: number;      // mean distance of blob pixels from their centroid
  pixels:     number[];    // flat pixel indices into the mask
  boundary:   Pixel[];     // 4-connected boundary points
}

function detectColorBlob(
  pretreated: Uint8Array,
  rgba:       Uint8Array,   // original (for hue measurement)
  width: number, height: number,
  color: 'yellow' | 'red' | 'blue',
): ColorBlob | null {

  // Pass 1 — wide range
  const mask1  = applyHsvFilter(pretreated, width, height, COLOUR_RANGES[color]);
  const blob1  = aggregateBlobs(mask1, width, height);
  const minPx  = (width * height) * 0.001;   // 0.1 % of image area
  if (blob1.pixelCount < minPx) return null;

  // Pass 2 — adaptive re-centering on the actual hue
  const medHue = computeMedianHue(rgba, width, blob1.pixels);
  const narrowRange = {
    hRanges: [[medHue - 22, medHue + 22]],   // ±22° around measured hue
    sMin: COLOUR_RANGES[color].sMin,
    vMin: COLOUR_RANGES[color].vMin,
  };
  const mask2 = applyHsvFilter(pretreated, width, height, narrowRange);
  const blob2 = aggregateBlobs(mask2, width, height);
  if (blob2.pixelCount < minPx) return null;

  const centroid   = computeCentroid(blob2.pixels, width);
  const meanRadius = computeMeanRadius(blob2.pixels, width, centroid);
  return { centroid, meanRadius, pixels: blob2.pixels, boundary: extractBoundary(mask2, width, height) };
}
```

`computeMedianHue` collects the H values of all blob pixels from the **original**
(not pretreated) image, sorts them, and returns the median. This is robust to
outliers from specular highlights or shadows.

Red's hue range wraps: when the adaptive centre lands near 0° or 360°, the two
sub-ranges merge into one [centre−22, centre+22] range using modular arithmetic.

---

## Phase 2 — Centre and scale bootstrap

### Colour zone centroid radii (from WA proportions)

Each scoring ring is 1 ring-width `w` wide. The colour zones span:

| Zone   | Radial span | Area-weighted centroid radius |
|--------|-------------|-------------------------------|
| Yellow | 0 → 2w      | ≈ 1.33 w                      |
| Red    | 2w → 4w     | ≈ 3.11 w                      |
| Blue   | 4w → 6w     | ≈ 5.07 w                      |

(Formula: centroid of annulus r₁→r₂ = 2(r₁²+r₁r₂+r₂²) / 3(r₁+r₂))

The mean radius of a colour blob's pixels from the target centre equals
approximately the zone centroid radius, so `w ≈ meanRadius / RATIO[color]`.

```typescript
const ZONE_RATIOS = { yellow: 1.33, red: 3.11, blue: 5.07 };

interface BootstrapEstimate {
  cx: number; cy: number;   // coarse target centre
  w:  number;               // coarse ring-width in pixels
}

function estimateCenterAndScale(
  blobs: Partial<Record<'yellow'|'red'|'blue', ColorBlob>>,
): BootstrapEstimate {

  const found = Object.entries(blobs).filter(([, b]) => b != null) as
    ['yellow'|'red'|'blue', ColorBlob][];

  // Target centre ≈ mean of blob centroids
  // (all colour blob centroids approximate the target centre because
  //  for a complete ring or disk, the area centroid equals the circle centre)
  const cx = mean(found.map(([, b]) => b.centroid.x));
  const cy = mean(found.map(([, b]) => b.centroid.y));

  // Ring-width estimates from each detected colour
  const wEsts = found.map(([color, blob]) => {
    // meanRadius is radius of blob pixels from *blob centroid* ≈ zone centroid radius
    return blob.meanRadius / ZONE_RATIOS[color];
  });
  const w = median(wEsts);

  return { cx, cy, w };
}
```

**Robustness note**: If only one colour is detected, the centre estimate equals
that blob's centroid (accurate for yellow since it is a solid disk; slightly
less accurate for red/blue which are rings, but sufficient to seed Phase 3).
Radial profiling in Phase 3 is tolerant of a centre error of up to ±0.3w.

---

## Phase 3 — Radial profile ring measurement

### 3a. Ray sampling

Sample pixels along N=360 evenly-spaced rays from the estimated centre, using the
**original** (un-pretreated) RGBA buffer so transitions are sharp.

```typescript
interface RaySample {
  dist: number;     // distance from centre in pixels
  v:    number;     // luminance (V channel of HSV), 0-1
}

function sampleRay(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, theta: number,
  step = 1.0, maxDist?: number,
): RaySample[] {

  const limit = maxDist ?? Math.hypot(width, height) / 2;
  const samples: RaySample[] = [];
  for (let d = step; d <= limit; d += step) {
    const x = cx + d * Math.cos(theta);
    const y = cy + d * Math.sin(theta);
    if (x < 0 || x >= width || y < 0 || y >= height) break;
    const i = (Math.round(y) * width + Math.round(x)) * 4;
    const [, , v] = rgbToHsv(rgba[i], rgba[i + 1], rgba[i + 2]);
    samples.push({ dist: d, v });
  }
  return samples;
}
```

### 3b. Transition detection along a ray

Apply a 1-D Gaussian smoothing (σ ≈ 1.5 samples) to V, then find local gradient
peaks. These are candidate ring-boundary crossings.

```typescript
interface Transition {
  dist:     number;   // distance from centre
  strength: number;   // |ΔV| at this crossing
}

function detectTransitions(
  samples: RaySample[],
  minStrength = 0.07,   // empirical: ~7% luminance change
): Transition[] {

  if (samples.length < 3) return [];

  // Smooth V with σ=1.5 samples (3-tap approximation [0.25, 0.5, 0.25])
  const vs = samples.map(s => s.v);
  const smoothed = vs.map((v, i) =>
    0.25 * (vs[Math.max(0, i-1)] + vs[Math.min(vs.length-1, i+1)]) + 0.5 * v
  );

  // Gradient magnitude between consecutive samples
  const raw: Transition[] = [];
  for (let i = 1; i < smoothed.length; i++) {
    const strength = Math.abs(smoothed[i] - smoothed[i - 1]);
    if (strength >= minStrength) {
      raw.push({ dist: (samples[i - 1].dist + samples[i].dist) / 2, strength });
    }
  }

  // Non-maximum suppression: keep only local maxima within a window of minSep
  return nonMaxSuppression(raw, /* minSep= */ 3 /* pixels */);
}

function nonMaxSuppression(ts: Transition[], minSep: number): Transition[] {
  // Sort by strength descending, greedily keep non-overlapping peaks
  const sorted = [...ts].sort((a, b) => b.strength - a.strength);
  const kept: Transition[] = [];
  for (const t of sorted) {
    if (kept.every(k => Math.abs(k.dist - t.dist) >= minSep)) kept.push(t);
  }
  return kept.sort((a, b) => a.dist - b.dist);   // back to distance order
}
```

### 3c. Match transitions to ring boundaries

Expected outer radius of ring k (0-indexed, 0 = bullseye) = `(k+1) × w0`.

For each ray, attempt to match one transition per ring boundary. Accept a match only
if it falls within ±0.4w0 of the expected radius.

```typescript
// transitionPoints[k] collects (x,y) for all rays that matched ring k's boundary
function collectRingPoints(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, w0: number,
  N = 360,
): Pixel[][] {

  const transitionPoints: Pixel[][] = Array.from({ length: 10 }, () => []);
  const tolerance = w0 * 0.4;
  const maxDist   = w0 * 11;   // slightly beyond outermost ring

  for (let i = 0; i < N; i++) {
    const theta = (i / N) * 2 * Math.PI;
    const samples = sampleRay(rgba, width, height, cx, cy, theta, 1.0, maxDist);
    const transitions = detectTransitions(samples);

    for (let k = 0; k < 10; k++) {
      const expected = (k + 1) * w0;
      // Closest transition to this ring's expected radius
      let best: Transition | null = null;
      for (const t of transitions) {
        if (Math.abs(t.dist - expected) <= tolerance) {
          if (!best || t.strength > best.strength) best = t;
        }
      }
      if (best) {
        transitionPoints[k].push({
          x: cx + best.dist * Math.cos(theta),
          y: cy + best.dist * Math.sin(theta),
        });
      }
    }
  }

  return transitionPoints;
}
```

---

## Phase 4 — Fitzgibbon algebraic ellipse fit

Replaces `fitEllipsePCA`. Based on Halir & Flusser (1998), which reduces the
constrained least-squares problem to a 3×3 eigenvalue system solvable without
external libraries.

### 4a. Scatter matrices

```typescript
// D1[i] = [xi², xi·yi, yi²],  D2[i] = [xi, yi, 1]
// S1 = D1ᵀ D1 (3×3),  S2 = D1ᵀ D2 (3×3),  S3 = D2ᵀ D2 (3×3)
```

### 4b. Reduced eigenvalue system

```typescript
// T   = −S3⁻¹ S2ᵀ            (3×3, requires 3×3 matrix inverse)
// M   = S1 + S2 T             (3×3)
// M'  = C1⁻¹ M               (3×3, C1 = [[0,0,2],[0,-1,0],[2,0,0]])
//
// C1⁻¹ = [[0, 0, 0.5],
//          [0, −1,  0 ],
//          [0.5, 0, 0 ]]
//
// Find eigenvector v of M' with smallest eigenvalue where 4v[0]v[2]−v[1]²>0
// (this is the ellipse constraint 4ac−b²>0)
```

### 4c. Recover conic coefficients

```typescript
// a1 = v  (the chosen eigenvector, length 3)
// a2 = T · a1                (length 3)
// Full conic: [A,B,C,D,E,F] = [a1[0],a1[1],a1[2], a2[0],a2[1],a2[2]]
// Represents: A·x² + B·x·y + C·y² + D·x + E·y + F = 0
```

### 4d. Convert conic to ellipse parameters

```typescript
function conicToRotatedRect(A,B,C,D,E,F): RotatedRect | null {
  // Must be an ellipse (not hyperbola/parabola)
  if (B*B - 4*A*C >= 0) return null;

  const denom = 4*A*C - B*B;          // > 0 for ellipse

  // Centre
  const cx = (B*E - 2*C*D) / denom;
  const cy = (B*D - 2*A*E) / denom;

  // Axis lengths
  const val    = 2*(A*E*E + C*D*D - B*D*E + (B*B - 4*A*C)*F);
  const common = Math.sqrt((A - C)**2 + B*B);
  const a      = -Math.sqrt(val * (A + C + common)) / denom;   // semi-major
  const b      = -Math.sqrt(val * (A + C - common)) / denom;   // semi-minor

  if (!isFinite(a) || !isFinite(b) || a <= 0 || b <= 0) return null;

  // Tilt angle (radians → degrees)
  const angle = 0.5 * Math.atan2(B, A - C) * (180 / Math.PI);

  return {
    center: { x: cx, y: cy },
    width:  2 * Math.max(a, b),    // full major axis length
    height: 2 * Math.min(a, b),    // full minor axis length
    angle,
  };
}
```

The 3×3 helpers needed (no external deps):
- `inv3x3(m)` — closed-form 3×3 inverse (determinant + cofactor matrix)
- `matMul3x3(a, b)` — 3×3 matrix multiplication
- `eig3x3(m)` — eigenvalues/vectors of a 3×3 real symmetric matrix via the
  analytical method (3rd-degree characteristic polynomial → trigonometric solution)

### 4e. Fallback for rings with sparse points

If a ring collects fewer than 6 transition points (e.g., target edge is cropped or
ring is fully in shadow), fall back to linear interpolation from the two nearest
successfully fitted neighbours.

```typescript
function interpolateMissingRings(rings: (RotatedRect | null)[]): RotatedRect[] {
  // For each null ring, interpolate from nearest valid neighbours
  // (linear interpolation on cx, cy, width, height, angle)
  // If no valid neighbours on one side, extrapolate from the other side
}
```

---

## Updated `findTarget` structure

```typescript
export function findTarget(rgba, width, height): ArcheryResult {
  try {
    const pretreated = pretreat(rgba, width, height);   // keep existing pretreat

    // Phase 1
    const blobs = {
      yellow: detectColorBlob(pretreated, rgba, width, height, 'yellow'),
      red:    detectColorBlob(pretreated, rgba, width, height, 'red'),
      blue:   detectColorBlob(pretreated, rgba, width, height, 'blue'),
    };
    if (Object.values(blobs).every(b => b === null))
      return { rings: [], success: false, error: 'No colour blobs found' };

    // Phase 2
    const { cx, cy, w } = estimateCenterAndScale(blobs);

    // Phase 3 — sample from original rgba, not pretreated
    const transitionPoints = collectRingPoints(rgba, width, height, cx, cy, w);

    // Phase 4
    const fitted = transitionPoints.map(pts => fitEllipseFitzgibbon(pts));
    const rings  = interpolateMissingRings(fitted);

    return {
      rings: rings.map(r => ({
        centerX: r.center.x, centerY: r.center.y,
        width: r.width, height: r.height, angle: r.angle,
      })),
      success: true,
    };
  } catch (e) {
    return { rings: [], success: false, error: String(e) };
  }
}
```

Functions **removed**: `rgbToHsvFull`, `applyColorFilter`, `cleanupContourPoints`,
`fitEllipsePCA`, `findWeakestElement`, `linearInterpolateTo10Rings`.

Functions **kept**: `pretreat`, `gaussianKernel`, `gaussianBlurChannel`,
`morphChannel`, `aggregateBlobs`, `extractBoundary`.

---

## Test suite changes

The new assertions that make failures visible (see research.md §7):

```typescript
// Rings must grow strictly outward
for (let i = 0; i < result.rings.length - 1; i++) {
  expect(result.rings[i].width).toBeLessThan(result.rings[i + 1].width);
}

// Bullseye must be a meaningful size (> 1% of image short side)
expect(result.rings[0].width).toBeGreaterThan(Math.min(width, height) * 0.01);

// No two consecutive rings may be identical (catches degenerate regression)
for (let i = 0; i < result.rings.length - 1; i++) {
  expect(result.rings[i].width).not.toBeCloseTo(result.rings[i + 1].width, 0);
}

// Aspect ratios must be plausible (< 4:1 — no degenerate near-flat ellipses)
for (const ring of result.rings) {
  expect(ring.width / ring.height).toBeLessThan(4);
}
```

---

## Step-by-step todo list

### Phase 1 — HSV fix and adaptive colour detection

- [x] **1.1** Replace `rgbToHsvFull` with `rgbToHsv(r,g,b): [H°, S, V]`
  (H 0–360°, S/V 0–1, no BGR swap). Delete the old function entirely.

- [x] **1.2** Define `COLOUR_RANGES` with wide initial hue windows and S/V floors
  (yellow 20–70°, red 0–18° + 342–360°, blue 190–245°).

- [x] **1.3** Implement `applyHsvFilter(rgba, width, height, range): Uint8Array`
  that accepts an array of `[hMin, hMax]` sub-ranges and handles the red
  hue-wraparound correctly.

- [x] **1.4** Implement `computeMedianHue(rgba, width, pixelIndices): number`
  that reads H from the *original* (not pretreated) pixel buffer for each index
  in a blob and returns the median.

- [x] **1.5** Implement `detectColorBlob(pretreated, rgba, width, height, color)`
  with the two-pass (wide → adaptive narrow) flow described above. Returns
  `ColorBlob | null`.

- [x] **1.6** Smoke-test: run all 10 images through Phase 1 only. Log how many
  colours are detected per image. Verify the 3 previously-failing images
  (193217, 202607, 204137) now detect at least 2 colours each.

### Phase 2 — Centre and scale bootstrap

- [x] **2.1** Implement `computeCentroid(pixelIndices, width): Pixel`.

- [x] **2.2** Implement `computeMeanRadius(pixelIndices, width, centroid): number`.

- [x] **2.3** Implement `estimateCenterAndScale(blobs): BootstrapEstimate`
  using `ZONE_RATIOS = { yellow: 1.33, red: 3.11, blue: 5.07 }`.

- [x] **2.4** Smoke-test: log `cx0, cy0, w0` for all 10 images. Verify `w0` is in a
  plausible range (e.g., 20–150 px for 1200-px images). Visually overlay centre
  + expected ring circles on the report to confirm.

### Phase 3 — Radial profile ring measurement

- [x] **3.1** Implement `sampleRay(rgba, width, height, cx, cy, theta, step, maxDist):
  RaySample[]`. Use nearest-neighbour pixel lookup (no bilinear needed).

- [x] **3.2** Implement `detectTransitions(samples, minStrength): Transition[]`
  with 3-tap smoothing and non-maximum suppression.

- [x] **3.3** Implement `nonMaxSuppression(transitions, minSep): Transition[]`.

- [x] **3.4** Implement `collectRingPoints(rgba, width, height, cx, cy, w0, N=360):
  Pixel[][]`. Log how many points are collected per ring per image during testing.

- [x] **3.5** Tune `minStrength` (transition threshold) and `tolerance` (match window
  ± fraction of w0) on the 10 test images. Target: ≥ 60 points per ring per image
  on the better images, ≥ 20 on the harder ones. Use `scripts/diag.ts` to
  inspect.

### Phase 4 — Fitzgibbon ellipse fit

- [x] **4.1** Implement 3×3 linear algebra helpers:
  - `matMul3x3(a, b): number[][]`
  - `matVecMul3(m, v): number[]`
  - `inv3x3(m): number[][]` (closed-form via cofactors + determinant)
  - `eigenvalues3x3` + `nullVec3x3` for general real 3×3 matrices
    (characteristic polynomial → trigonometric/Cardano roots → cross-product null space).

- [x] **4.2** Implement `fitEllipseFitzgibbon(points: Pixel[]): RotatedRect | null`
  using the Halir-Flusser formulation: build S1/S2/S3, compute T, form M′,
  find the constrained eigenvector, recover conic coefficients.

- [x] **4.3** Implement `conicToRotatedRect(A,B,C,D,E,F): RotatedRect | null`
  with the closed-form centre / axis / angle formulas.

- [x] **4.4** Unit-test `fitEllipseFitzgibbon` with synthetic points on a known
  ellipse (e.g., semi-axes 100×60 at 30°). Verify the returned ellipse matches
  to < 1 px error.

- [x] **4.5** Implement `interpolateMissingRings(rings: (RotatedRect|null)[]):
  RotatedRect[]` — linearly interpolate any null entries from their nearest
  valid neighbours (or extrapolate if only one side has data).

### Wiring + cleanup

- [x] **5.1** Rewrite `findTarget` to call the new phases in sequence. Keep
  `pretreat` call unchanged.

- [x] **5.2** Delete dead code: `rgbToHsvFull`, `applyColorFilter`,
  `cleanupContourPoints`, `fitEllipsePCA`, `findWeakestElement`,
  `linearInterpolateTo10Rings`. Also delete `FILTER_YELLOW / FILTER_RED /
  FILTER_BLUE` constants.

- [x] **5.3** Update test assertions in `targetDetection.test.ts` with the
  stronger checks listed in the "Test suite changes" section above. Run tests;
  all 10 images must pass the new assertions.

- [x] **5.4** Run `npm run visualize` and inspect `report.html`. Verify ellipses
  are visually correct (concentric, covering the coloured rings) on all 10 images,
  paying special attention to 202607, 204137, 193217 (previously all-identical),
  193820, 195129, 195801 (previously degenerate bullseye).

- [x] **5.5** Update `CLAUDE.md` to remove the HSV BGR-swap note and replace with
  the new standard-HSV / radial-profile description.

---

## Implementation order rationale

Phases 1 → 2 → 3 → 4 must be completed in order (each feeds the next), but within
each phase the steps can be developed and tested incrementally. Phase 1 alone is
likely to fix the 3 colour-miss failures (2a); Phases 3–4 are needed to fix the 3
degenerate-bullseye failures (2b). The test suite update (5.3) should be done
*before* finalising tuning so that the new assertions drive the threshold choices.

---

*Last updated: 2026-03-13*
