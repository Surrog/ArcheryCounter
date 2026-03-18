// New archery target detection pipeline.
// See docs/research.md and docs/plan.md for design notes.

import type { SplineRing } from './spline';
import { sampleClosedSpline } from './spline';

export interface Pixel { x: number; y: number }


/** @deprecated Use SplineRing from './spline' instead. Kept for backward compatibility. */
export interface EllipseData {
  centerX: number; centerY: number;
  width: number; height: number;
  angle: number;
}

/**
 * Target paper boundary as an ordered polygon (4–8 vertices, clockwise in image
 * coordinates). Replaces the old fixed 4-point tuple, supporting curvature-detected
 * mid-edge vertices for bowed or folded paper edges.
 */
export interface TargetBoundary {
  points: [number, number][];
}

/** Per-image HSV colour references (illuminant-corrected), one median per zone. */
export interface ColourCalibration {
  gold:  [number, number, number]; // H 0–360, S 0–1, V 0–1
  red:   [number, number, number];
  blue:  [number, number, number];
  black: [number, number, number];
  white: [number, number, number];
}

export interface RayDebugEntry {
  theta: number;              // ray angle in radians
  boundary: number;           // boundary distance (smoothed)
  distances: (number | null)[]; // detected distances per ring [0..9]
}

export interface ArcheryResult {
  rings: SplineRing[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  /** Raw per-ray transition points, indexed by ring (0=innermost). */
  ringPoints?: Pixel[][];
  /** Per-ray debug data, populated only when DEBUG_RAYS is set. */
  rayDebug?: RayDebugEntry[];
  success: boolean;
  error?: string;
}

// ---------------------------------------------------------------------------
// Standard RGB → HSV conversion (H 0–360°, S/V 0–1)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Color filter — multi-range with hue wraparound support
// ---------------------------------------------------------------------------

interface HsvRange {
  hRanges: [number, number][];
  sMin: number;
  vMin: number;
}

const COLOUR_RANGES: Record<'yellow' | 'red' | 'blue', HsvRange> = {
  yellow: { hRanges: [[20, 70]],            sMin: 0.30, vMin: 0.30 },
  red:    { hRanges: [[0, 18], [342, 360]], sMin: 0.30, vMin: 0.20 },
  blue:   { hRanges: [[190, 245]],          sMin: 0.25, vMin: 0.20 },
};

function hueInRange(h: number, hMin: number, hMax: number): boolean {
  const lo = ((hMin % 360) + 360) % 360;
  const hi = ((hMax % 360) + 360) % 360;
  if (lo <= hi) return h >= lo && h <= hi;
  return h >= lo || h <= hi;
}

function applyHsvFilter(
  rgba: Uint8Array, width: number, height: number,
  range: HsvRange,
): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const [h, s, v] = rgbToHsv(r, g, b);
    if (s < range.sMin || v < range.vMin) continue;
    for (const [hMin, hMax] of range.hRanges) {
      if (hueInRange(h, hMin, hMax)) { mask[i] = 255; break; }
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Pretreatment — Gaussian blur + erode + dilate (unchanged from original)
// ---------------------------------------------------------------------------

function gaussianKernel(size: number, sigma: number): Float64Array {
  const k = new Float64Array(size);
  const half = (size - 1) / 2;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - half;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  return k.map(v => v / sum);
}

function gaussianBlurChannel(
  src: Uint8Array, width: number, height: number,
  kernel: Float64Array,
): Uint8Array {
  const half = (kernel.length - 1) / 2;
  const tmp = new Uint8Array(src.length);
  const dst = new Uint8Array(src.length);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kernel.length; k++) {
        const sx = Math.min(Math.max(x - half + k, 0), width - 1);
        acc += src[y * width + sx] * kernel[k];
      }
      tmp[y * width + x] = Math.round(acc);
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = 0; k < kernel.length; k++) {
        const sy = Math.min(Math.max(y - half + k, 0), height - 1);
        acc += tmp[sy * width + x] * kernel[k];
      }
      dst[y * width + x] = Math.round(acc);
    }
  }
  return dst;
}

function morphChannel(
  src: Uint8Array, width: number, height: number, mode: 'erode' | 'dilate',
): Uint8Array {
  const combine = mode === 'erode' ? Math.min : Math.max;
  const n = src.length;
  const tmp = new Uint8Array(n);
  const dst = new Uint8Array(n);

  // Horizontal pass (1×3)
  for (let y = 0; y < height; y++) {
    const row = y * width;
    tmp[row] = combine(src[row], src[row + 1]);
    for (let x = 1; x < width - 1; x++) {
      tmp[row + x] = combine(combine(src[row + x - 1], src[row + x]), src[row + x + 1]);
    }
    tmp[row + width - 1] = combine(src[row + width - 2], src[row + width - 1]);
  }

  // Vertical pass (3×1)
  for (let x = 0; x < width; x++) {
    dst[x] = combine(tmp[x], tmp[width + x]);
    for (let y = 1; y < height - 1; y++) {
      dst[y * width + x] = combine(combine(tmp[(y - 1) * width + x], tmp[y * width + x]), tmp[(y + 1) * width + x]);
    }
    dst[(height - 1) * width + x] = combine(tmp[(height - 2) * width + x], tmp[(height - 1) * width + x]);
  }

  return dst;
}

const GAUSS_KERNEL = gaussianKernel(15, 1.5);

function pretreat(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const n = width * height;
  let r = new Uint8Array(n), g = new Uint8Array(n), b = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = rgba[i * 4]; g[i] = rgba[i * 4 + 1]; b[i] = rgba[i * 4 + 2];
  }
  function process(ch: Uint8Array): Uint8Array {
    let c = gaussianBlurChannel(ch, width, height, GAUSS_KERNEL);
    c = morphChannel(c, width, height, 'erode');
    for (let i = 0; i < 3; i++) c = morphChannel(c, width, height, 'dilate');
    return c;
  }
  r = process(r); g = process(g); b = process(b);
  const out = new Uint8Array(rgba);
  for (let i = 0; i < n; i++) {
    out[i * 4] = r[i]; out[i * 4 + 1] = g[i]; out[i * 4 + 2] = b[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Blob detection — BFS flood fill with aggregation; returns mask + pixel list
// ---------------------------------------------------------------------------

interface BlobResult {
  mask: Uint8Array;
  pixels: number[];          // all aggregated pixels (largest + nearby components)
  largestPixels: number[];   // single largest connected component only
  pixelCount: number;
}

function aggregateBlobs(
  mask: Uint8Array, width: number, height: number,
  mergeThresholdFactor = 2.5,
): BlobResult {
  const labels = new Int32Array(mask.length).fill(-1);
  const componentPixels: number[][] = [];
  let label = 0;

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || labels[start] !== -1) continue;
    const pixels: number[] = [start];
    labels[start] = label;
    let head = 0;
    while (head < pixels.length) {
      const idx = pixels[head++];
      const px = idx % width, py = (idx / width) | 0;
      for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]] as const) {
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const ni = ny * width + nx;
        if (mask[ni] && labels[ni] === -1) { labels[ni] = label; pixels.push(ni); }
      }
    }
    componentPixels.push(pixels);
    label++;
  }

  if (componentPixels.length === 0) {
    return { mask: new Uint8Array(mask.length), pixels: [], largestPixels: [], pixelCount: 0 };
  }

  // Prefer the component whose centroid is closest to the image centre over
  // the globally largest one.  Background regions (hay bale, floor) tend to be
  // large and near the image edge; target colour zones are near the centre.
  // Fall back to globally largest if no component meets the minimum size.
  const minAnchorPx = Math.max(6, Math.floor(mask.length * 0.001));
  const imgCx = width / 2, imgCy = height / 2;
  let anchorIdx = -1;
  let anchorDist = Infinity;
  let largestIdx = 0;
  for (let i = 0; i < componentPixels.length; i++) {
    if (componentPixels[i].length > componentPixels[largestIdx].length) largestIdx = i;
    if (componentPixels[i].length < minAnchorPx) continue;
    let sx = 0, sy = 0;
    for (const idx of componentPixels[i]) { sx += idx % width; sy += (idx / width) | 0; }
    const d = Math.hypot(sx / componentPixels[i].length - imgCx,
                         sy / componentPixels[i].length - imgCy);
    if (d < anchorDist) { anchorDist = d; anchorIdx = i; }
  }
  if (anchorIdx === -1) anchorIdx = largestIdx;

  const largestPixels = componentPixels[anchorIdx];
  let sumX = 0, sumY = 0;
  for (const idx of largestPixels) { sumX += idx % width; sumY += (idx / width) | 0; }
  const cx = sumX / largestPixels.length, cy = sumY / largestPixels.length;

  let totalDist = 0;
  for (const idx of largestPixels) {
    totalDist += Math.hypot((idx % width) - cx, ((idx / width) | 0) - cy);
  }
  // Cap merge distance: never sweep in components more than 40% of the short
  // image side away, regardless of how spread-out the anchor blob is.
  const maxMergeDist = 0.4 * Math.min(width, height);
  const threshold = Math.min(mergeThresholdFactor * (totalDist / largestPixels.length),
                             maxMergeDist);

  const outMask = new Uint8Array(mask.length);
  const allPixels: number[] = [];

  for (let c = 0; c < componentPixels.length; c++) {
    const pixels = componentPixels[c];
    if (c === anchorIdx) {
      for (const idx of pixels) { outMask[idx] = 255; allPixels.push(idx); }
      continue;
    }
    if (pixels.length <= 6) continue;
    let csx = 0, csy = 0;
    for (const idx of pixels) { csx += idx % width; csy += (idx / width) | 0; }
    if (Math.hypot(csx / pixels.length - cx, csy / pixels.length - cy) <= threshold) {
      for (const idx of pixels) { outMask[idx] = 255; allPixels.push(idx); }
    }
  }

  return { mask: outMask, pixels: allPixels, largestPixels, pixelCount: allPixels.length };
}

// ---------------------------------------------------------------------------
// Boundary extraction
// ---------------------------------------------------------------------------

function extractBoundary(blob: Uint8Array, width: number, height: number): Pixel[] {
  const pts: Pixel[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!blob[i]) continue;
      if (
        (x > 0           && !blob[i - 1]) ||
        (x < width - 1   && !blob[i + 1]) ||
        (y > 0           && !blob[i - width]) ||
        (y < height - 1  && !blob[i + width])
      ) pts.push({ x, y });
    }
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Phase 1 — Adaptive colour detection
// ---------------------------------------------------------------------------

function computeMedianHue(rgba: Uint8Array, _width: number, pixelIndices: number[]): number {
  const hues: number[] = [];
  for (const idx of pixelIndices) {
    const r = rgba[idx * 4], g = rgba[idx * 4 + 1], b = rgba[idx * 4 + 2];
    const [h, s, v] = rgbToHsv(r, g, b);
    if (s > 0.1 && v > 0.1) hues.push(h);
  }
  if (hues.length === 0) return 0;
  hues.sort((a, b) => a - b);
  return hues[Math.floor(hues.length / 2)];
}

interface ColorBlob {
  centroid: Pixel;
  meanRadius: number;
  pixels: number[];
  boundary: Pixel[];
}

function computeCentroid(pixelIndices: number[], width: number): Pixel {
  let sumX = 0, sumY = 0;
  for (const idx of pixelIndices) {
    sumX += idx % width;
    sumY += (idx / width) | 0;
  }
  return { x: sumX / pixelIndices.length, y: sumY / pixelIndices.length };
}

/**
 * Median distance from `centroid` to all pixels. Robust to parasitic blobs
 * (hay bale, clothing) that contaminate the aggregated colour mask: they must
 * comprise > 50 % of pixels before the median is affected, vs. any fraction
 * for the mean.
 */
function computeMedianRadius(pixelIndices: number[], width: number, centroid: Pixel): number {
  const dists = pixelIndices.map(
    idx => Math.hypot((idx % width) - centroid.x, ((idx / width) | 0) - centroid.y),
  ).sort((a, b) => a - b);
  const mid = Math.floor(dists.length / 2);
  return dists.length % 2 === 0 ? (dists[mid - 1] + dists[mid]) / 2 : dists[mid];
}

function detectColorBlob(
  pretreated: Uint8Array, rgba: Uint8Array,
  width: number, height: number,
  color: 'yellow' | 'red' | 'blue',
): ColorBlob | null {
  const minPx = width * height * 0.001;

  // Pass 1 — wide initial range on pretreated image
  const mask1 = applyHsvFilter(pretreated, width, height, COLOUR_RANGES[color]);
  const blob1 = aggregateBlobs(mask1, width, height);
  if (blob1.pixelCount < minPx) return null;

  // Pass 2 — adaptive re-centering based on actual hue in original image
  const medHue = computeMedianHue(rgba, width, blob1.pixels);
  const lo = medHue - 22, hi = medHue + 22;

  let hRanges: [number, number][];
  if (color === 'red') {
    // Red can wrap around 0°/360°
    if (lo < 0) {
      hRanges = [[lo + 360, 360], [0, hi]];
    } else if (hi > 360) {
      hRanges = [[lo, 360], [0, hi - 360]];
    } else {
      hRanges = [[lo, hi]];
    }
  } else {
    // Clamp non-wrapping colours to [0, 360]
    hRanges = [[Math.max(0, lo), Math.min(360, hi)]];
  }

  const narrowRange: HsvRange = {
    hRanges,
    sMin: COLOUR_RANGES[color].sMin,
    vMin: COLOUR_RANGES[color].vMin,
  };

  const mask2 = applyHsvFilter(pretreated, width, height, narrowRange);
  // Yellow is a solid disk — use tighter aggregation to exclude nearby hay-bale
  // or clothing pixels that sit just outside the target's yellow zone.
  const mergeThresholdFactor = 2.5;
  const blob2 = aggregateBlobs(mask2, width, height, mergeThresholdFactor);
  if (blob2.pixelCount < minPx) return null;

  const centroid   = computeCentroid(blob2.pixels, width);
  const meanRadius = computeMedianRadius(blob2.pixels, width, centroid);
  return { centroid, meanRadius, pixels: blob2.pixels, boundary: extractBoundary(blob2.mask, width, height) };
}

// ---------------------------------------------------------------------------
// Phase 2 — Centre and scale bootstrap
// ---------------------------------------------------------------------------

// Median-distance ratios for each WA colour zone (in units of ring-width w).
// Median is used instead of area-weighted mean for robustness to parasitic blobs
// (e.g., hay-bale behind target). Values from F(r)=0.5 of the zone area CDF:
//   yellow disk  0→2w : r = 2w/√2   ≈ 1.414w
//   red annulus  2w→4w: r = √10 w   ≈ 3.162w
//   blue annulus 4w→6w: r = √26 w   ≈ 5.099w
const ZONE_RATIOS: Record<'yellow' | 'red' | 'blue', number> = {
  yellow: 1.414,
  red:    3.162,
  blue:   5.099,
};

function arrayMean(values: number[]): number {
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function arrayMedian(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Convert radial-profile transition points for one ring into a closed SplineRing.
 * Sorts pts by angle around (cx, cy) and uses them directly as Catmull-Rom control
 * points, so the rendered spline passes through every detected transition position.
 * Falls back to a unit-circle placeholder when no points are available.
 */
function radialProfileToSpline(pts: Pixel[], cx: number, cy: number, K = 12): SplineRing {
  if (pts.length === 0) {
    // Degenerate: return unit-circle placeholder.
    return {
      points: Array.from({ length: K }, (_, k) => [
        cx + Math.cos((2 * Math.PI * k) / K),
        cy + Math.sin((2 * Math.PI * k) / K),
      ] as [number, number]),
    };
  }
  // Sort detected points by angle and use them directly as Catmull-Rom control points.
  // The spline will interpolate through each point, so the curve passes through the dots.
  const sorted = [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );
  return { points: sorted.map(p => [p.x, p.y] as [number, number]) };
}

/** Circular (wrap-around) median filter for a 1-D array. */
function medianFilter1D(data: number[], windowRadius: number): number[] {
  const n = data.length;
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    const window: number[] = [];
    for (let k = -windowRadius; k <= windowRadius; k++) {
      window.push(data[(i + k + n) % n]);
    }
    window.sort((a, b) => a - b);
    result[i] = window[Math.floor(window.length / 2)];
  }
  return result;
}

/**
 * Clamp outlier points in a ring's point set using angular-neighbour comparison.
 *
 * Points are sorted by angle, then each point's radius is compared against the
 * median of its K nearest angular neighbours (K each side, circular wrap).
 * Points that deviate more than `maxDeviation` from their local median are
 * snapped to that local median, keeping their original angle.
 * Two passes let snapped points stabilise the neighbourhood for the next pass.
 */
function filterRingOutliers(pts: Pixel[], cx: number, cy: number, maxDeviation: number): Pixel[] {
  if (pts.length < 4) return pts;

  const K = 3; // neighbours on each side
  // Sort by angle once; subsequent passes preserve this order.
  let result = [...pts].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx),
  );

  for (let pass = 0; pass < 2; pass++) {
    const n = result.length;
    const radii = result.map(p => Math.hypot(p.x - cx, p.y - cy));
    result = result.map((p, i) => {
      const window: number[] = [];
      for (let d = -K; d <= K; d++) window.push(radii[(i + d + n) % n]);
      const localMed = arrayMedian(window);
      if (Math.abs(radii[i] - localMed) <= maxDeviation) return p;
      const angle = Math.atan2(p.y - cy, p.x - cx);
      return { x: cx + localMed * Math.cos(angle), y: cy + localMed * Math.sin(angle) };
    });
  }
  return result;
}

interface BootstrapEstimate { cx: number; cy: number; w: number; }

function estimateCenterAndScale(
  blobs: Partial<Record<'yellow' | 'red' | 'blue', ColorBlob>>,
): BootstrapEstimate {
  const found = (['yellow', 'red', 'blue'] as const)
    .filter(k => blobs[k] != null)
    .map(k => [k, blobs[k]!] as ['yellow' | 'red' | 'blue', ColorBlob]);

  const cx = arrayMedian(found.map(([, b]) => b.centroid.x));
  const cy = arrayMedian(found.map(([, b]) => b.centroid.y));
  const wEsts = found.map(([c, b]) => b.meanRadius / ZONE_RATIOS[c]);
  const w = arrayMedian(wEsts);
  return { cx, cy, w };
}


// ---------------------------------------------------------------------------
// Phase 2 — Per-image colour calibration
// ---------------------------------------------------------------------------

type ZoneName = 'gold' | 'red' | 'blue' | 'black' | 'white';

// Sample at the midpoint of each ring within the zone (in units of w).
// Gold: rings 0–1 (radius 0–2w), Red: rings 2–3 (2w–4w), etc.
const ZONE_SAMPLE_RADII: Record<ZoneName, number[]> = {
  gold:  [0.5, 1.5],
  red:   [2.5, 3.5],
  blue:  [4.5, 5.5],
  black: [6.5, 7.5],
  white: [8.5, 9.5],
};

interface RawZoneSamples {
  gold:  [number, number, number][];
  red:   [number, number, number][];
  blue:  [number, number, number][];
  black: [number, number, number][];
  white: [number, number, number][];
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r1 = 0, g1 = 0, b1 = 0;
  if      (h < 60)  { r1 = c; g1 = x; }
  else if (h < 120) { r1 = x; g1 = c; }
  else if (h < 180) { g1 = c; b1 = x; }
  else if (h < 240) { g1 = x; b1 = c; }
  else if (h < 300) { r1 = x; b1 = c; }
  else              { r1 = c; b1 = x; }
  return [r1 + m, g1 + m, b1 + m];
}

/**
 * Circular mean hue + median S/V from a set of HSV samples.
 * Returns null if the sample array is empty.
 */
function summariseHsv(samples: [number, number, number][]): [number, number, number] | null {
  if (samples.length === 0) return null;
  // Circular mean for hue (robust for unimodal zone distributions)
  const sinH = arrayMean(samples.map(([h]) => Math.sin(h * Math.PI / 180)));
  const cosH = arrayMean(samples.map(([h]) => Math.cos(h * Math.PI / 180)));
  let medH = Math.atan2(sinH, cosH) * 180 / Math.PI;
  if (medH < 0) medH += 360;
  const medS = arrayMedian(samples.map(([, s]) => s));
  const medV = arrayMedian(samples.map(([,, v]) => v));
  return [medH, medS, medV];
}

/**
 * Cast 8 evenly-spaced rays from (cx, cy) and sample HSV at the expected
 * radial midpoints of each scoring zone.  Samples beyond the smoothed
 * boundary (boundaryDists, indexed 0..N_BOUNDARY-1 for 360 rays) are skipped.
 */
function sampleZoneColours(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, w: number,
  boundaryDists: number[],
): RawZoneSamples {
  const N_SAMPLE_RAYS = 8;
  const N_BOUNDARY = boundaryDists.length; // typically 360
  const samples: RawZoneSamples = { gold: [], red: [], blue: [], black: [], white: [] };

  for (let i = 0; i < N_SAMPLE_RAYS; i++) {
    const theta = (i / N_SAMPLE_RAYS) * 2 * Math.PI;
    // Map this ray angle to the nearest boundary-dist index
    const boundaryIdx = Math.round((i / N_SAMPLE_RAYS) * N_BOUNDARY) % N_BOUNDARY;
    const maxDist = boundaryDists[boundaryIdx];

    for (const [zoneName, radii] of Object.entries(ZONE_SAMPLE_RADII) as [ZoneName, number[]][]) {
      for (const rw of radii) {
        const d = rw * w;
        if (d >= maxDist) continue; // outside boundary
        const px = Math.round(cx + d * Math.cos(theta));
        const py = Math.round(cy + d * Math.sin(theta));
        if (px < 0 || px >= width || py < 0 || py >= height) continue;
        const pidx = (py * width + px) * 4;
        samples[zoneName].push(rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]));
      }
    }
  }

  return samples;
}

/**
 * Compute per-image colour calibration from zone samples.
 * Applies a von Kries illuminant correction so that the white zone's median
 * RGB maps to (1, 1, 1), then expresses every zone in that corrected space.
 * Returns null if any zone has no samples.
 */
function computeCalibration(samples: RawZoneSamples): ColourCalibration | null {
  const zones: ZoneName[] = ['gold', 'red', 'blue', 'black', 'white'];
  const medians: Partial<Record<ZoneName, [number, number, number]>> = {};

  for (const z of zones) {
    const m = summariseHsv(samples[z]);
    if (!m) return null;
    medians[z] = m;
  }

  // Von Kries: scale R/G/B channels so the white zone median maps to (1,1,1).
  const [wh, ws, wv] = medians.white!;
  const [wr, wg, wb] = hsvToRgb(wh, ws, wv);
  const rScale = wr > 0.01 ? 1 / wr : 1;
  const gScale = wg > 0.01 ? 1 / wg : 1;
  const bScale = wb > 0.01 ? 1 / wb : 1;
  // Normalise so no channel exceeds 1 in the corrected space
  const maxScale = Math.max(rScale, gScale, bScale);
  const rs = rScale / maxScale, gs = gScale / maxScale, bs = bScale / maxScale;

  const correct = ([h, s, v]: [number, number, number]): [number, number, number] => {
    const [r, g, b] = hsvToRgb(h, s, v);
    return rgbToHsv(
      Math.round(Math.min(1, r * rs) * 255),
      Math.round(Math.min(1, g * gs) * 255),
      Math.round(Math.min(1, b * bs) * 255),
    );
  };

  return {
    gold:  correct(medians.gold!),
    red:   correct(medians.red!),
    blue:  correct(medians.blue!),
    black: correct(medians.black!),
    white: correct(medians.white!),
  };
}

// ---------------------------------------------------------------------------
// Phase 3 — Colour-guided ring detection
// ---------------------------------------------------------------------------

/**
 * P3-T1: Classify one pixel (by HSV) to the nearest WA scoring zone using
 * the per-image calibration as references.  Saturation-weighted hue distance
 * means well-saturated zones are primarily discriminated by hue, while
 * black/white (low S or V) are discriminated by S and V.
 * Returns null if the pixel is further from every reference than the threshold.
 */
function classifyPixelZone(
  [h, s, v]: [number, number, number],
  cal: ColourCalibration,
): ZoneName | null {
  let minDist = 1.2; // rejection threshold
  let nearest: ZoneName | null = null;
  for (const [zn, ref] of Object.entries(cal) as [ZoneName, [number, number, number]][]) {
    const [rh, rs, rv] = ref;
    let dh = Math.abs(h - rh);
    if (dh > 180) dh = 360 - dh;
    const avgS = (s + rs) / 2;
    const dist = Math.sqrt(
      ((dh / 180) * (1 + avgS * 2)) ** 2 +
      ((s - rs) * 1.2) ** 2 +
      (v - rv) ** 2,
    );
    if (dist < minDist) { minDist = dist; nearest = zn; }
  }
  return nearest;
}

/**
 * Fit a linear model r(n) = a*n + b through known ring-boundary distances (OLS).
 * Returns a predictor for any ring index n.
 * Used to extrapolate white zone rings [8,9] from the 4 detected colour transitions.
 */
function fitRingRadiusModel(
  knownBoundaries: { ringIdx: number; dist: number }[],
): (n: number) => number {
  const k = knownBoundaries.length;
  if (k === 0) return (n) => n;
  if (k === 1) { const d0 = knownBoundaries[0].dist; return () => d0; }
  const sumN  = knownBoundaries.reduce((s, { ringIdx }) => s + ringIdx, 0);
  const sumD  = knownBoundaries.reduce((s, { dist })    => s + dist, 0);
  const sumNN = knownBoundaries.reduce((s, { ringIdx }) => s + ringIdx * ringIdx, 0);
  const sumND = knownBoundaries.reduce((s, { ringIdx, dist }) => s + ringIdx * dist, 0);
  const denom = k * sumNN - sumN * sumN;
  if (Math.abs(denom) < 1e-9) return () => sumD / k;
  const a = (k * sumND - sumN * sumD) / denom;
  const b = (sumD - a * sumN) / k;
  return (n: number) => a * n + b;
}

/**
 * Walk one ray, classify each pixel by colour zone, detect the 4 colour-zone
 * transitions, then derive all 10 ring-boundary distances.
 *
 * Ring-index → boundary:
 *   [0] ≈1w  gold inner (X-ring)      — interpolated: r1 / 2
 *   [1] ≈2w  gold→red transition      — detected
 *   [2] ≈3w  red inner                — interpolated: (r1 + r3) / 2
 *   [3] ≈4w  red→blue transition      — detected
 *   [4] ≈5w  blue inner               — interpolated: (r3 + r5) / 2
 *   [5] ≈6w  blue→black transition    — detected
 *   [6] ≈7w  black inner              — interpolated: (r5 + r7) / 2
 *   [7] ≈8w  black→white transition   — detected
 *   [8] ≈9w  white inner              — linear regression on [1,3,5,7]
 *   [9] ≈10w white outer              — linear regression on [1,3,5,7]
 *
 * White zone (rings 8–9) is not directly detected (low saturation, similar to
 * background hay/wall), so both boundaries are extrapolated via linear
 * regression through the four confirmed colour transitions, clamped below the
 * paper-boundary distance.
 */
function detectRingDistancesOnRay(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, cosT: number, sinT: number,
  boundaryDist: number, w: number,
  cal: ColourCalibration,
): (number | null)[] {
  const result: (number | null)[] = Array(10).fill(null);
  // result[9] is deduced from regression below — boundaryDist is the paper edge
  // (slightly beyond the outermost scoring line) and is used only as a clamp.

  // Walk from d=2 outward, classify each pixel.
  const zones: (ZoneName | null)[] = [];
  const dArr: number[] = [];
  for (let d = 2; d <= Math.floor(boundaryDist); d++) {
    const px = Math.round(cx + d * cosT);
    const py = Math.round(cy + d * sinT);
    if (px < 0 || px >= width || py < 0 || py >= height) {
      zones.push(null);
    } else {
      const pidx = (py * width + px) * 4;
      zones.push(classifyPixelZone(rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]), cal));
    }
    dArr.push(d);
  }
  if (zones.length === 0) return result;

  // Mode-smooth (window = 5) to remove single-pixel classification noise.
  const L = zones.length;
  const smooth: (ZoneName | null)[] = zones.map((_, i) => {
    const counts = new Map<ZoneName | null, number>();
    for (let k = Math.max(0, i - 2); k <= Math.min(L - 1, i + 2); k++) {
      const z = zones[k];
      counts.set(z, (counts.get(z) ?? 0) + 1);
    }
    let best: ZoneName | null = null, bestC = 0;
    for (const [z, c] of counts) if (c > bestC) { bestC = c; best = z; }
    return best;
  });

  // Detect the 4 colour-zone transitions independently.
  // Requires a streak of MIN_STREAK_PER_TRANSITION consecutive pixels in the
  // new zone AND a minimum distance gate to reject near-centre artefacts.
  //
  // Detect the 4 reliable colour-zone transitions: gold→red, red→blue,
  // blue→black, black→white.  These are committed to result[1,3,5,7].
  // The white zone MIDDLE boundary (result[8]) is NOT directly detected —
  // it is deduced from regression over all reliably-detected rings.
  //
  // Arrow-shaft / reflection rejection (applies to all 4 transitions):
  //   1. MIN_STREAK: must see N consecutive pixels of the new zone.
  //   2. MIN_ZONE_WIDTH: the new zone must span ≥ this many total pixels.
  //      Real zones are ~1w wide; shafts/reflections are typically < 0.3w.
  const SEQUENCE: ZoneName[] = ['gold', 'red', 'blue', 'black', 'white'];
  const TRANSITION_EXPECTED_W = [2, 4, 6, 8];
  const MIN_STREAK     = 10;
  const MIN_ZONE_WIDTH = Math.round(w * 0.4);
  const transitionDist: (number | null)[] = Array(4).fill(null);

  for (let ti = 0; ti < 4; ti++) {
    const fromZone = SEQUENCE[ti], toZone = SEQUENCE[ti + 1];
    const minD = TRANSITION_EXPECTED_W[ti] * w * 0.5;
    let sawFrom = false, streakTo = 0, streakStart = -1;
    for (let i = 0; i < smooth.length; i++) {
      const z = smooth[i];
      if (z === fromZone) { sawFrom = true; streakTo = 0; streakStart = -1; }
      else if (sawFrom && z === toZone) {
        if (streakTo === 0) streakStart = i;
        streakTo++;
        if (streakTo >= MIN_STREAK) {
          const transD = dArr[streakStart];
          if (transD >= minD) {
            let zoneWidth = streakTo;
            for (let j = i + 1; j < smooth.length && smooth[j] === toZone; j++) zoneWidth++;
            if (zoneWidth < MIN_ZONE_WIDTH) { streakTo = 0; streakStart = -1; continue; }
            transitionDist[ti] = transD;
            break;
          }
          streakTo = 0; streakStart = -1;
        }
      } else if (sawFrom && z !== null) {
        streakTo = 0; streakStart = -1;
      }
    }
  }

  // Commit the 4 detected colour transitions to ring indices [1, 3, 5, 7].
  // Intra-zone rings [0, 2, 4, 6] are NOT computed here — they are derived at
  // the spline level in findTarget() by interpolating between adjacent fitted
  // zone-boundary splines.  That avoids propagating per-ray noise before fitting.
  const [r1, r3, r5, r7] = transitionDist;
  result[1] = r1;
  result[3] = r3;
  result[5] = r5;
  result[7] = r7;

  // White zone: not reliably detectable (low saturation, similar to background).
  // Extrapolate rings[8] and rings[9] via linear regression through the 4
  // confirmed colour transitions, clamped below the paper boundary.
  const knownBoundaries = ([
    { ringIdx: 1, dist: r1 },
    { ringIdx: 3, dist: r3 },
    { ringIdx: 5, dist: r5 },
    { ringIdx: 7, dist: r7 },
  ] as { ringIdx: number; dist: number | null }[])
    .filter(({ dist }) => dist !== null) as { ringIdx: number; dist: number }[];

  const r7safe = r7 ?? 8 * w;
  if (knownBoundaries.length >= 2) {
    const predictR = fitRingRadiusModel(knownBoundaries);
    result[8] = Math.max(r7safe,              Math.min(boundaryDist * 0.98, predictR(8)));
    result[9] = Math.max(result[8] as number, Math.min(boundaryDist * 0.99, predictR(9)));
  } else {
    result[8] = 9  * w < boundaryDist ? 9  * w : null;
    result[9] = 10 * w < boundaryDist ? 10 * w : null;
  }

  return result;
}

/**
 * Colour-guided ring detection.
 * For each of N rays, detects all 10 ring-boundary distances via zone
 * classification and within-zone luminance divider detection.
 * Returns raw Pixel[][] arrays and per-ray debug data (rayDebug).
 */
function collectRingPointsColourGuided(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, w: number,
  N: number, smoothedDists: number[],
  cal: ColourCalibration,
): { ringPoints: Pixel[][]; rayDebug: RayDebugEntry[] } {
  const ringPoints: Pixel[][] = Array.from({ length: 10 }, () => []);
  const rayDebug: RayDebugEntry[] = [];
  for (let i = 0; i < N; i++) {
    const theta = (i / N) * 2 * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    const distances = detectRingDistancesOnRay(
      rgba, width, height, cx, cy, cosT, sinT, smoothedDists[i], w, cal,
    );
    rayDebug.push({ theta, boundary: smoothedDists[i], distances });
    for (let k = 0; k < 10; k++) {
      const d = distances[k];
      if (d === null || d <= 0) continue;
      const px = Math.round(cx + d * cosT);
      const py = Math.round(cy + d * sinT);
      if (px >= 0 && px < width && py >= 0 && py < height) {
        ringPoints[k].push({ x: px, y: py });
      }
    }
  }
  return { ringPoints, rayDebug };
}

// ---------------------------------------------------------------------------
// Phase 3b — Radial profile ring measurement (luminance, legacy fallback)
// ---------------------------------------------------------------------------

interface RaySample  { dist: number; v: number; }
interface Transition { dist: number; strength: number; }

function sampleRay(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, theta: number,
  step = 1.0, maxDist?: number,
): RaySample[] {
  const limit = maxDist ?? Math.hypot(width, height) / 2;
  const samples: RaySample[] = [];
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  for (let d = step; d <= limit; d += step) {
    const x = cx + d * cosT, y = cy + d * sinT;
    if (x < 0 || x >= width || y < 0 || y >= height) break;
    const i = (Math.round(y) * width + Math.round(x)) * 4;
    const [, , v] = rgbToHsv(rgba[i], rgba[i + 1], rgba[i + 2]);
    samples.push({ dist: d, v });
  }
  return samples;
}

function nonMaxSuppression(ts: Transition[], minSep: number): Transition[] {
  const sorted = [...ts].sort((a, b) => b.strength - a.strength);
  const kept: Transition[] = [];
  for (const t of sorted) {
    if (kept.every(k => Math.abs(k.dist - t.dist) >= minSep)) kept.push(t);
  }
  return kept.sort((a, b) => a.dist - b.dist);
}

function detectTransitions(samples: RaySample[], minStrength = 0.07): Transition[] {
  if (samples.length < 3) return [];
  const vs = samples.map(s => s.v);
  // 3-tap Gaussian smoothing [0.25, 0.5, 0.25]
  const smoothed = vs.map((v, i) =>
    0.25 * (vs[Math.max(0, i - 1)] + vs[Math.min(vs.length - 1, i + 1)]) + 0.5 * v
  );
  const raw: Transition[] = [];
  for (let i = 1; i < smoothed.length; i++) {
    const strength = Math.abs(smoothed[i] - smoothed[i - 1]);
    if (strength >= minStrength) {
      raw.push({ dist: (samples[i - 1].dist + samples[i].dist) / 2, strength });
    }
  }
  return nonMaxSuppression(raw, 3);
}

function collectRingPoints(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, w0: number, N = 360,
  toleranceFactor = 0.4,
  boundaryDists?: number[],
): Pixel[][] {
  const transitionPoints: Pixel[][] = Array.from({ length: 10 }, () => []);
  const tolerance = w0 * toleranceFactor;

  for (let i = 0; i < N; i++) {
    const theta = (i / N) * 2 * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    // Cap search distance at the detected target-paper boundary for this ray.
    const maxDist = boundaryDists ? boundaryDists[i] : w0 * 11.5;
    const samples = sampleRay(rgba, width, height, cx, cy, theta, 1.0, maxDist);
    const transitions = detectTransitions(samples);

    for (let k = 0; k < 10; k++) {
      const expected = (k + 1) * w0;
      let best: Transition | null = null;
      for (const t of transitions) {
        if (Math.abs(t.dist - expected) <= tolerance && (!best || t.strength > best.strength)) {
          best = t;
        }
      }
      if (best) {
        transitionPoints[k].push({ x: cx + best.dist * cosT, y: cy + best.dist * sinT });
      }
    }
  }

  return transitionPoints;
}

// ---------------------------------------------------------------------------
// Quadrilateral fitting — rectangular paper boundary
// ---------------------------------------------------------------------------


/**
 * Fits a perspective-projected rectangle to a set of boundary points.
 *
 * Uses four fixed axis-aligned diagonal projections (x+y, x−y) to locate the
 * four corners.  For any convex quadrilateral — including a perspective-
 * projected rectangle at any orientation — the four corners are exactly the
 * points that are most extreme in these four directions:
 *
 *   TL → min(x + y)   (upper-left  in image coords, y-down)
 *   TR → max(x − y)   (upper-right)
 *   BR → max(x + y)   (lower-right)
 *   BL → min(x − y)   (lower-left)
 *
 * No PCA rotation is needed: rotating the coordinate frame by the PCA angle
 * would work for a perfectly uniform point distribution but is biased when
 * boundary points are denser on some sides (e.g. when the image crops the
 * top of the target), causing the PCA angle to deviate and picking the wrong
 * corners.  The unrotated diagonal projections are parameter-free and robust.
 *
 * Returns [topLeft, topRight, bottomRight, bottomLeft] in image coordinates,
 * or null if there are fewer than 4 points.
 */
function fitQuadrilateral(pts: Pixel[]): [Pixel, Pixel, Pixel, Pixel] | null {
  if (pts.length < 4) return null;

  let tlS =  Infinity, trS = -Infinity;
  let brS = -Infinity, blS =  Infinity;
  let tl = pts[0], tr = pts[0], br = pts[0], bl = pts[0];

  for (const p of pts) {
    const pp = p.x + p.y;   // sum  → selects TL (min) and BR (max)
    const pm = p.x - p.y;   // diff → selects TR (max) and BL (min)
    if (pp < tlS) { tlS = pp; tl = p; }
    if (pm > trS) { trS = pm; tr = p; }
    if (pp > brS) { brS = pp; br = p; }
    if (pm < blS) { blS = pm; bl = p; }
  }

  return [tl, tr, br, bl];
}

/**
 * Fits a convex `TargetBoundary` polygon to a set of boundary scan points.
 *
 * Uses gift-wrapping (Jarvis march) to compute the convex hull of the scan
 * points, then simplifies it to at most `maxVertices` vertices by
 * repeatedly removing the vertex whose removal causes the smallest triangle
 * area (i.e., the most collinear vertex).  Always produces a convex polygon.
 *
 * A convex polygon is correct here because the target paper boundary is a
 * perspective-projected rectangle — always convex — and `pointInPolygon`
 * is simpler and faster for convex polygons.
 */
function fitBoundaryPolygon(pts: Pixel[], maxVertices = 8): TargetBoundary {
  if (pts.length < 3) return { points: [] };

  // Signed-area cross product in image coordinates (y-down).
  // Positive value → clockwise turn; negative → counter-clockwise.
  const cross2d = (ox: number, oy: number, ax: number, ay: number, bx: number, by: number): number =>
    (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);

  // --- Gift-wrapping convex hull (CW orientation in image coords / CCW in math) ---
  // Start from the topmost point (min y), then leftmost as tie-break.
  let startIdx = 0;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].y < pts[startIdx].y ||
        (pts[i].y === pts[startIdx].y && pts[i].x < pts[startIdx].x)) {
      startIdx = i;
    }
  }

  const hull: Pixel[] = [];
  let currIdx = startIdx;
  do {
    hull.push(pts[currIdx]);
    let nextIdx = (currIdx + 1) % pts.length;
    for (let i = 0; i < pts.length; i++) {
      const c = cross2d(
        pts[currIdx].x, pts[currIdx].y,
        pts[nextIdx].x, pts[nextIdx].y,
        pts[i].x,       pts[i].y,
      );
      // Negative cross → pts[i] is more CCW than pts[nextIdx] in image coords
      // (more to the right when walking CW around the hull from the top).
      if (c < 0) {
        nextIdx = i;
      } else if (c === 0) {
        // Collinear: keep the farther candidate
        const d1 = (pts[nextIdx].x - pts[currIdx].x) ** 2 + (pts[nextIdx].y - pts[currIdx].y) ** 2;
        const d2 = (pts[i].x       - pts[currIdx].x) ** 2 + (pts[i].y       - pts[currIdx].y) ** 2;
        if (d2 > d1) nextIdx = i;
      }
    }
    currIdx = nextIdx;
  } while (currIdx !== startIdx && hull.length <= pts.length);

  if (hull.length < 3) return { points: hull.map(p => [p.x, p.y]) };

  // --- Simplify: repeatedly remove the vertex with smallest triangle area ---
  while (hull.length > maxVertices) {
    let minArea = Infinity;
    let minIdx = 0;
    const n = hull.length;
    for (let i = 0; i < n; i++) {
      const prev = hull[(i + n - 1) % n];
      const next = hull[(i + 1) % n];
      const curr = hull[i];
      const area = Math.abs(cross2d(prev.x, prev.y, curr.x, curr.y, next.x, next.y)) / 2;
      if (area < minArea) { minArea = area; minIdx = i; }
    }
    hull.splice(minIdx, 1);
  }

  return { points: hull.map(p => [p.x, p.y]) };
}

/**
 * Ray-cast point-in-polygon test for a `TargetBoundary` polygon.
 * Returns true if `pt` lies strictly inside the polygon.
 */
export function pointInPolygon(pt: Pixel, poly: TargetBoundary): boolean {
  const { points } = poly;
  const n = points.length;
  if (n < 3) return false;
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = points[i];
    const [xj, yj] = points[j];
    if (((yi > pt.y) !== (yj > pt.y)) &&
        (pt.x < (xj - xi) * (pt.y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

// ---------------------------------------------------------------------------
// Phase 2b — Target paper boundary scan
// ---------------------------------------------------------------------------

interface BoundaryScan {
  /** Per-ray distance from (cx,cy) to the last non-background pixel. */
  dists: number[];
  /** Last non-background pixel per ray — used for ring[9] ellipse fitting. */
  points: Pixel[];
}

/**
 * Scans N rays from (cx, cy) starting at `startRadius`, walking outward until
 * hitting hay-bale background (H∈[15°,65°], S>0.25) or the image edge.
 *
 * Starting past the colour zones (≥4w from centre) avoids confusion with the
 * yellow and red zones which share a similar hue range to hay bale.  The black
 * zone (very dark, low S) is not hay-bale and is traversed transparently.
 *
 * Returns the boundary distance for each ray; rays that reach the image edge
 * without crossing hay bale record the image-edge distance as their limit.
 */
function scanTargetBoundary(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number,
  startRadius: number,
  N = 360,
): BoundaryScan {
  const dists: number[] = new Array(N).fill(0);
  const points: Pixel[] = [];

  for (let i = 0; i < N; i++) {
    const theta = (i / N) * 2 * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    let lastValidD = startRadius;

    for (let d = Math.max(1, Math.round(startRadius)); ; d++) {
      const x = cx + d * cosT, y = cy + d * sinT;
      if (x < 0 || x >= width || y < 0 || y >= height) {
        dists[i] = lastValidD;
        break;
      }
      const pidx = (Math.round(y) * width + Math.round(x)) * 4;
      const [h, s, v] = rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]);
      // Hay bale: brownish-yellow, moderately saturated, NOT very dark.
      // The V > 0.15 floor is critical: the black scoring zone has H ≈ 15–65°
      // and S > 0.25 (faint warm tint on near-black paper), so without a
      // brightness floor it triggers as hay and stops the scan early inside
      // the target.  Real hay straw always has V > 0.25 in practice.
      const isBackground = s > 0.25 && h >= 15 && h <= 65 && v > 0.20;
      if (isBackground) {
        dists[i] = lastValidD;
        break;
      }
      lastValidD = d;
    }

    points.push({
      x: Math.round(cx + lastValidD * cosT),
      y: Math.round(cy + lastValidD * sinT),
    });
  }

  return { dists, points };
}

// ---------------------------------------------------------------------------
// Main findTarget function
// ---------------------------------------------------------------------------

export function findTarget(
  rgba: Uint8Array, width: number, height: number,
): ArcheryResult {
  try {
    const pretreated = pretreat(rgba, width, height);

    // Phase 1 — adaptive colour detection
    const blobs = {
      yellow: detectColorBlob(pretreated, rgba, width, height, 'yellow'),
      red:    detectColorBlob(pretreated, rgba, width, height, 'red'),
      blue:   detectColorBlob(pretreated, rgba, width, height, 'blue'),
    };

    if (Object.values(blobs).every(b => b === null)) {
      return { rings: [], success: false, error: 'No colour blobs found' };
    }

    // Phase 2 — coarse centre + ring-width estimate
    const { cx, cy, w } = estimateCenterAndScale(blobs);

    if (w <= 0 || !isFinite(cx) || !isFinite(cy)) {
      return { rings: [], success: false, error: 'Invalid bootstrap estimate' };
    }

    // Phase 2b — Detect target paper boundary.
    // Start scanning from 4×w (past yellow+red zones) so hay-bale hue range
    // is not confused with the yellow scoring zone.  Per-ray distances cap all
    // subsequent ring searches, preventing ellipses from extending beyond the
    // rectangular white target border.
    // Boundary scan uses 360 rays for precise polygon-corner fitting.
    // Ring detection uses 32 rays (reduced cost; each ring still gets ample coverage).
    const N_BOUNDARY = 360;
    const N_RINGS    = 32;
    const boundary = scanTargetBoundary(
      rgba, width, height, cx, cy, Math.max(w * 4, 20), N_BOUNDARY,
    );
    // Smooth the per-ray distances with a circular median filter (±10 rays)
    // to eliminate single-angle outliers caused by hay-straw spikes or
    // borderline-threshold pixels at isolated angles.
    const smoothedDists = medianFilter1D(boundary.dists, 10);
    const smoothedPoints = smoothedDists.map((d, i) => {
      const theta = (i / N_BOUNDARY) * 2 * Math.PI;
      return { x: Math.round(cx + d * Math.cos(theta)), y: Math.round(cy + d * Math.sin(theta)) };
    });
    if (process.env.DEBUG_RINGS) {
      const medD = arrayMedian(smoothedDists);
      console.error(`[debug] boundary median dist=${medD.toFixed(1)} min=${Math.min(...smoothedDists).toFixed(1)} max=${Math.max(...smoothedDists).toFixed(1)}`);
    }

    // Phase 2 — per-image colour calibration.
    // sampleZoneColours uses N_BOUNDARY rays for the same smoothedDists array.
    const zoneSamples = sampleZoneColours(rgba, width, height, cx, cy, w, smoothedDists);
    const calibration = computeCalibration(zoneSamples) ?? undefined;

    // Phase 3 — colour-guided ring detection (falls back to luminance if no calibration).
    // Subsample smoothedDists to N_RINGS angles by picking every other ray.
    const smoothedDistsRings = Array.from({ length: N_RINGS }, (_, i) =>
      smoothedDists[Math.round(i * N_BOUNDARY / N_RINGS) % N_BOUNDARY],
    );
    const { ringPoints: rawTransitionPoints, rayDebug } = calibration
      ? collectRingPointsColourGuided(rgba, width, height, cx, cy, w, N_RINGS, smoothedDistsRings, calibration)
      : { ringPoints: collectRingPoints(rgba, width, height, cx, cy, w, N_RINGS, 0.4, smoothedDistsRings), rayDebug: [] };

    // Filter outlier points: any point whose radius deviates by more than 0.15w from
    // its angular-local median is snapped to that median radius.  This corrects
    // stray detections (arrow holes, single-ray noise) without discarding points.
    const transitionPoints = rawTransitionPoints.map(
      pts => filterRingOutliers(pts, cx, cy, w * 0.15),
    );

    const maxBoundaryR = Math.max(...smoothedDists);

    // Build splines for the 5 directly-detected rings [1,3,5,7,9] plus regression-
    // derived white boundaries [8,9].  Intra-zone rings [0,2,4,6] are filled below
    // by spline-level interpolation — this avoids accumulating per-ray detection
    // noise into regions that have no colour-zone boundary to anchor them.
    const rings: SplineRing[] = transitionPoints.map(pts => radialProfileToSpline(pts, cx, cy));

    // Interpolate intra-zone ring splines from adjacent detected zone boundaries.
    // Each WA zone is divided exactly in half, so t=0.5 is geometrically correct.
    // Ring[0]: between the target centre and ring[1] (the gold/red boundary spline).
    // Resample a spline to exactly K uniformly-distributed points for arithmetic operations.
    const LERP_K = 12;
    const resample = (ring: SplineRing): [number, number][] => {
      const all = sampleClosedSpline(ring.points, LERP_K * ring.points.length);
      const step = all.length / LERP_K;
      return Array.from({ length: LERP_K }, (_, i) => all[Math.floor(i * step)] as [number, number]);
    };
    const centerRing: SplineRing = { points: Array.from({ length: LERP_K }, () => [cx, cy] as [number, number]) };
    const lerpSpline = (a: SplineRing, b: SplineRing): SplineRing => {
      const aPts = resample(a);
      const bPts = resample(b);
      return { points: aPts.map((pa, i) => [(pa[0] + bPts[i][0]) / 2, (pa[1] + bPts[i][1]) / 2] as [number, number]) };
    };
    rings[0] = lerpSpline(centerRing, rings[1]);  // gold inner  (X-ring)
    rings[2] = lerpSpline(rings[1],   rings[3]);  // red inner
    rings[4] = lerpSpline(rings[3],   rings[5]);  // blue inner
    rings[6] = lerpSpline(rings[5],   rings[7]);  // black inner
    rings[8] = lerpSpline(rings[7],   rings[9]);  // white inner

    // Fit a boundary polygon (4–8 vertices) to the full set of boundary points.
    // The extra vertices capture bowed or folded paper edges that a fixed
    // 4-corner quad cannot represent.
    const paperBoundary = fitBoundaryPolygon(smoothedPoints);

    // ringPoints only exposes the 4 directly-detected colour-transition rings
    // (indices 1,3,5,7) plus the regression-derived white rings (8,9).
    // Intra-zone rings (0,2,4,6) have no raw detection points.
    const detectedRingPoints = transitionPoints.map((pts, i) =>
      [0, 2, 4, 6].includes(i) ? [] : pts,
    );

    return {
      rings,
      paperBoundary,
      calibration,
      ringPoints: detectedRingPoints,
      rayDebug: rayDebug.length > 0 ? rayDebug : undefined,
      success: true,
    };
  } catch (e) {
    return { rings: [], success: false, error: String(e) };
  }
}
