// New archery target detection pipeline.
// See docs/research.md and docs/plan.md for design notes.

export interface Pixel { x: number; y: number }

interface RotatedRect {
  center: Pixel;
  width: number;
  height: number;
  angle: number; // degrees
}

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

export interface ArcheryResult {
  rings: EllipseData[];
  paperBoundary?: TargetBoundary;
  calibration?: ColourCalibration;
  /** Raw per-ray transition points before ellipse fitting, indexed by ring (0=innermost). */
  ringPoints?: Pixel[][];
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
  const dst = new Uint8Array(src.length);
  const combine = mode === 'erode' ? Math.min : Math.max;
  const init = mode === 'erode' ? 255 : 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let val = init;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = Math.min(Math.max(x + dx, 0), width - 1);
          const ny = Math.min(Math.max(y + dy, 0), height - 1);
          val = combine(val, src[ny * width + nx]);
        }
      }
      dst[y * width + x] = val;
    }
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
 * Remove outlier points from a ring's point set.
 *
 * Stage 1 — radial (two-pass): discard points whose distance from (cx,cy)
 * deviates more than `maxDeviation` from the set's median radius.  Catches
 * rays that latched onto a completely wrong ring boundary.
 *
 * Stage 2 — angular neighbor: the points are assumed to be ordered by angle
 * (one per ray).  A point is rejected if its radius differs from the median
 * of its ±NEIGHBOR_WIN circular neighbors by more than `maxDeviation`.  This
 * catches single-ray spikes that survive the radial pass because they happen
 * to cluster near the overall median, without penalising the genuine radial
 * variation of an oblique ellipse.
 */
function filterRingOutliers(pts: Pixel[], cx: number, cy: number, maxDeviation: number): Pixel[] {
  if (pts.length < 6) return pts;

  // Stage 1: two-pass radial filter.
  let result = pts;
  for (let pass = 0; pass < 2; pass++) {
    if (result.length < 6) break;
    const radii = result.map(p => Math.hypot(p.x - cx, p.y - cy));
    const medR = arrayMedian(radii);
    result = result.filter((_, i) => Math.abs(radii[i] - medR) <= maxDeviation);
  }
  if (result.length < 6) return result;

  // Stage 2: angular neighbor filter.
  const NEIGHBOR_WIN = 5; // ±5 rays = ±5° for 360-ray layout
  const radii = result.map(p => Math.hypot(p.x - cx, p.y - cy));
  const n = result.length;
  return result.filter((_, i) => {
    const neighbors: number[] = [];
    for (let k = -NEIGHBOR_WIN; k <= NEIGHBOR_WIN; k++) {
      if (k === 0) continue;
      neighbors.push(radii[(i + k + n) % n]);
    }
    const localMed = arrayMedian(neighbors);
    return Math.abs(radii[i] - localMed) <= maxDeviation;
  });
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
 * P4-T1: Scan the white zone (from `whiteStart` to `boundaryDist`) along one ray
 * for the printed black ring that marks the outer boundary of ring 1.
 * Requires V < DARK_THRESHOLD for MIN_STREAK consecutive pixels.
 * Returns the distance of the first such dark pixel, or null if not found.
 */
function detectOutermostBlackLine(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, cosT: number, sinT: number,
  whiteStart: number, boundaryDist: number,
): number | null {
  // Scan INWARD from the paper boundary so we find the outermost dark feature
  // in the white zone — which is the printed black scoring circle.  Scanning
  // outward risks stopping at a smudge or shadow near the inner white boundary.
  const DARK_THRESHOLD = 0.40;
  const MIN_STREAK = 2;
  let streak = 0, lastDark = -1;
  for (let d = Math.floor(boundaryDist) - 1; d >= Math.ceil(whiteStart); d--) {
    const px = Math.round(cx + d * cosT);
    const py = Math.round(cy + d * sinT);
    if (px < 0 || px >= width || py < 0 || py >= height) continue;
    const pidx = (py * width + px) * 4;
    const [,, v] = rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]);
    if (v < DARK_THRESHOLD) {
      if (streak === 0) lastDark = d;
      streak++;
      if (streak >= MIN_STREAK) return lastDark;
    } else {
      streak = 0;
    }
  }
  return null;
}

/**
 * P4-T2: Fit a linear model r(n) = a*n + b through the known ring-boundary
 * distances (OLS). Returns a predictor function for any ring index n.
 * Used to extrapolate white ring positions when direct detection fails.
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
 * P3-T3: Find the luminance extremum along one ray segment [dStart, dEnd]
 * within a ±searchHalf window around `dExpected`.
 * `findMin=true` for colour zones (divider is a dark printed line);
 * `findMin=false` for the black zone (divider is a bright printed line).
 * Falls back to `dExpected` if nothing is found in range.
 */
function detectZoneDivider(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, cosT: number, sinT: number,
  dStart: number, dEnd: number, dExpected: number,
  searchHalf: number, findMin: boolean,
): number {
  const lo = Math.max(dStart + 1, dExpected - searchHalf);
  const hi = Math.min(dEnd - 1, dExpected + searchHalf);
  if (hi <= lo) return dExpected;
  let extremeV = findMin ? Infinity : -Infinity;
  let extremeD = dExpected;
  for (let d = lo; d <= hi; d++) {
    const px = Math.round(cx + d * cosT);
    const py = Math.round(cy + d * sinT);
    if (px < 0 || px >= width || py < 0 || py >= height) continue;
    const pidx = (py * width + px) * 4;
    const [,, v] = rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]);
    if (findMin ? v < extremeV : v > extremeV) { extremeV = v; extremeD = d; }
  }
  return extremeD;
}

/**
 * P3-T2/T4: Walk one ray, classify each pixel by zone, detect 4 colour-zone
 * transitions and 5 within-zone dividers.  Returns 10 ring-boundary distances.
 *
 * Ring-index → boundary:
 *   [0] ≈1w  gold divider (bullseye outer / X-ring line)  [luminance min]
 *   [1] ≈2w  gold→red colour transition
 *   [2] ≈3w  red divider                                   [luminance min]
 *   [3] ≈4w  red→blue colour transition
 *   [4] ≈5w  blue divider                                  [luminance min]
 *   [5] ≈6w  blue→black colour transition
 *   [6] ≈7w  black divider                                 [luminance max]
 *   [7] ≈8w  black→white colour transition
 *   [8] ≈9w  white divider                                 [luminance min]
 *   [9] ≈10w paper boundary (from smoothedDists)
 */
function detectRingDistancesOnRay(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, cosT: number, sinT: number,
  boundaryDist: number, w: number,
  cal: ColourCalibration,
): (number | null)[] {
  const result: (number | null)[] = Array(10).fill(null);
  result[9] = boundaryDist;

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
  // Requires a streak of MIN_STREAK consecutive pixels in the new zone AND a
  // minimum distance gate to reject near-centre artefacts (arrow holes, wear)
  // that can briefly match the next zone's colour.
  // Expected transitions: gold→red ≈2w, red→blue ≈4w, blue→black ≈6w, black→white ≈8w.
  // Gate: must be at least 0.7×expected distance from centre.
  const SEQUENCE: ZoneName[] = ['gold', 'red', 'blue', 'black', 'white'];
  const MIN_STREAK = 3;
  const TRANSITION_EXPECTED_W = [2, 4, 6, 8]; // ×w multiples for each transition
  const transitionDist: (number | null)[] = Array(4).fill(null);

  for (let ti = 0; ti < 4; ti++) {
    const fromZone = SEQUENCE[ti], toZone = SEQUENCE[ti + 1];
    const minD = TRANSITION_EXPECTED_W[ti] * w * 0.5; // min distance gate (50% of expected)
    let sawFrom = false, streakTo = 0;
    for (let i = 0; i < smooth.length; i++) {
      const z = smooth[i];
      if (z === fromZone) { sawFrom = true; streakTo = 0; }
      else if (sawFrom && z === toZone) {
        streakTo++;
        if (streakTo >= MIN_STREAK) {
          const transD = dArr[i - MIN_STREAK + 1];
          if (transD >= minD) {
            transitionDist[ti] = transD;
            break;
          }
          // Below minimum distance — keep scanning (reset streak, wait for the real transition)
          streakTo = 0;
        }
      } else if (sawFrom && z !== null) {
        streakTo = 0; // reset on recognised non-transition zone; ignore null (noise)
      }
    }
  }

  // Map transitions to ring indices: [1, 3, 5, 7]
  result[1] = transitionDist[0];
  result[3] = transitionDist[1];
  result[5] = transitionDist[2];
  result[7] = transitionDist[3];

  // Find the first confirmed gold pixel to anchor the gold zone start.
  let goldStart = 2;
  for (let i = 0; i < smooth.length; i++) {
    if (smooth[i] === 'gold') { goldStart = dArr[i]; break; }
  }

  // Zone extents derived from transition distances.
  const zoneExtent: ([number, number] | null)[] = [
    transitionDist[0] != null ? [goldStart, transitionDist[0]] : null,
    transitionDist[0] != null && transitionDist[1] != null ? [transitionDist[0], transitionDist[1]] : null,
    transitionDist[1] != null && transitionDist[2] != null ? [transitionDist[1], transitionDist[2]] : null,
    transitionDist[2] != null && transitionDist[3] != null ? [transitionDist[2], transitionDist[3]] : null,
    transitionDist[3] != null ? [transitionDist[3], boundaryDist] : null,
  ];

  // Within-zone dividers: [ringIdx, seqIdx, expectedW×multiple, findMin]
  // Note: ring[8] (white divider) is handled separately below via P4 detection.
  const DIVIDER_CFG: [number, number, number, boolean][] = [
    [0, 0, 1, true ],  // gold divider  ≈1w  (bullseye outer)
    [2, 1, 3, true ],  // red divider   ≈3w
    [4, 2, 5, true ],  // blue divider  ≈5w
    [6, 3, 7, false],  // black divider ≈7w  (white printed line → max)
  ];
  const searchHalf = 0.4 * w;

  for (const [ringIdx, si, mult, findMin] of DIVIDER_CFG) {
    const extent = zoneExtent[si];
    const expectedD = mult * w;
    if (!extent) {
      if (expectedD < boundaryDist) result[ringIdx] = expectedD;
      continue;
    }
    const [dStart, dEnd] = extent;
    const clampedExp = Math.max(dStart + 1, Math.min(dEnd - 1, expectedD));
    result[ringIdx] = detectZoneDivider(
      rgba, width, height, cx, cy, cosT, sinT,
      dStart, dEnd, clampedExp, searchHalf, findMin,
    );
  }

  // P4-T3: White ring divider (result[8]) — the printed black circle at the
  // outer edge of ring 1.  Detection priority:
  //   1. detectOutermostBlackLine: scan the confirmed white zone for a dark streak
  //   2. fitRingRadiusModel: linear regression from 4 known colour transitions
  //   3. detectZoneDivider fallback (luminance min within white zone)
  //   4. w-based estimate (9w), capped at boundaryDist
  {
    const whiteExtent = zoneExtent[4]; // [transitionDist[3], boundaryDist]
    const expectedD = 9 * w;

    // Build regression model from reliably-detected colour-zone transitions +
    // the paper boundary as a hard anchor at ring index 9.  Anchoring at the
    // actual boundary turns the extrapolation into an interpolation, preventing
    // the regression from overshooting on angled/distorted shots.
    const knownBoundaries = ([
      { ringIdx: 1, dist: transitionDist[0] },
      { ringIdx: 3, dist: transitionDist[1] },
      { ringIdx: 5, dist: transitionDist[2] },
      { ringIdx: 7, dist: transitionDist[3] },
      { ringIdx: 9, dist: boundaryDist },      // anchor at paper boundary
    ] as { ringIdx: number; dist: number | null }[])
      .filter(({ dist }) => dist !== null) as { ringIdx: number; dist: number }[];
    const predictR = knownBoundaries.length >= 2
      ? fitRingRadiusModel(knownBoundaries)
      : null;

    let detected: number | null = null;

    // 1. Black-line scan within confirmed white zone.
    if (whiteExtent) {
      detected = detectOutermostBlackLine(
        rgba, width, height, cx, cy, cosT, sinT,
        whiteExtent[0], whiteExtent[1],
      );
    }

    // 2. Linear regression estimate (only if the regression point is inside boundary).
    if (detected === null && predictR !== null) {
      const regEst = predictR(8);
      if (regEst > 0 && regEst < boundaryDist) detected = regEst;
    }

    // 3. detectZoneDivider fallback within confirmed white zone.
    if (detected === null && whiteExtent) {
      const [dStart, dEnd] = whiteExtent;
      const clampedExp = Math.max(dStart + 1, Math.min(dEnd - 1, expectedD));
      detected = detectZoneDivider(
        rgba, width, height, cx, cy, cosT, sinT,
        dStart, dEnd, clampedExp, searchHalf, true,
      );
    }

    // 4. w-based fallback.
    if (detected === null && expectedD < boundaryDist) detected = expectedD;

    result[8] = detected;
  }

  return result;
}

/**
 * P3-T5: Colour-guided replacement for collectRingPoints.
 * For each of N rays detects all 10 ring-boundary distances via zone
 * classification + within-zone luminance divider detection, then returns
 * Pixel[][] arrays for fitEllipseFitzgibbon (same interface as before).
 */
function collectRingPointsColourGuided(
  rgba: Uint8Array, width: number, height: number,
  cx: number, cy: number, w: number,
  N: number, smoothedDists: number[],
  cal: ColourCalibration,
): Pixel[][] {
  const ringPoints: Pixel[][] = Array.from({ length: 10 }, () => []);
  for (let i = 0; i < N; i++) {
    const theta = (i / N) * 2 * Math.PI;
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    const distances = detectRingDistancesOnRay(
      rgba, width, height, cx, cy, cosT, sinT, smoothedDists[i], w, cal,
    );
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
  return ringPoints;
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
// Phase 4 — Fitzgibbon algebraic ellipse fit (Halir & Flusser 1998)
// ---------------------------------------------------------------------------

function matMul3x3(a: number[][], b: number[][]): number[][] {
  const c = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++)
        c[i][j] += a[i][k] * b[k][j];
  return c;
}

function matVecMul3(m: number[][], v: number[]): number[] {
  return [
    m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
    m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
    m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
  ];
}

function inv3x3(m: number[][]): number[][] | null {
  const [[a,b,c],[d,e,f],[g,h,k]] = m;
  const det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
  if (Math.abs(det) < 1e-14) return null;
  const inv = 1 / det;
  return [
    [ (e*k - f*h)*inv, -(b*k - c*h)*inv,  (b*f - c*e)*inv ],
    [-(d*k - f*g)*inv,  (a*k - c*g)*inv, -(a*f - c*d)*inv ],
    [ (d*h - e*g)*inv, -(a*h - b*g)*inv,  (a*e - b*d)*inv ],
  ];
}

/**
 * Eigenvalues of a real 3×3 matrix via the characteristic polynomial.
 * Returns all three real roots (may include repeated roots).
 * Falls back to a single real root + two imaginary via Cardano when discriminant < 0.
 */
function eigenvalues3x3(m: number[][]): number[] {
  const a00=m[0][0], a01=m[0][1], a02=m[0][2];
  const a10=m[1][0], a11=m[1][1], a12=m[1][2];
  const a20=m[2][0], a21=m[2][1], a22=m[2][2];

  const T = a00 + a11 + a22;
  const P = (a00*a11 - a01*a10) + (a11*a22 - a12*a21) + (a00*a22 - a02*a20);
  const D = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);

  // Depressed cubic t³ + p·t + q = 0, where λ = t + T/3
  const p = P - T*T/3;
  const q = -2*T*T*T/27 + T*P/3 - D;

  const disc = -(4*p*p*p + 27*q*q);
  const offset = T / 3;

  if (p === 0 && q === 0) return [offset, offset, offset];

  if (disc >= 0 && p < 0) {
    // Three real roots — trigonometric method
    const alpha = Math.sqrt(-p / 3);
    const arg   = Math.max(-1, Math.min(1, -q / (2 * alpha * alpha * alpha)));
    const phi   = Math.acos(arg) / 3;
    const m2    = 2 * alpha;
    return [
      offset + m2 * Math.cos(phi),
      offset + m2 * Math.cos(phi + 2 * Math.PI / 3),
      offset + m2 * Math.cos(phi + 4 * Math.PI / 3),
    ];
  }

  // One real root — Cardano's formula
  const rad  = Math.sqrt(Math.abs(q*q/4 + p*p*p/27));
  const u    = Math.cbrt(-q/2 + rad);
  const v    = Math.cbrt(-q/2 - rad);
  return [offset + u + v];
}

/**
 * Null-space vector of a 3×3 matrix (assumed near-singular for eigenvalue λ).
 * Uses cross product of the two highest-norm rows.
 */
function nullVec3x3(m: number[][], lambda: number): number[] {
  const A = [
    [m[0][0] - lambda, m[0][1],           m[0][2]          ],
    [m[1][0],          m[1][1] - lambda,  m[1][2]          ],
    [m[2][0],          m[2][1],           m[2][2] - lambda ],
  ];
  const norms = A.map(r => Math.hypot(r[0], r[1], r[2]));
  const sorted = [0,1,2].sort((i,j) => norms[j] - norms[i]);
  const r0 = A[sorted[0]], r1 = A[sorted[1]];
  const v = [
    r0[1]*r1[2] - r0[2]*r1[1],
    r0[2]*r1[0] - r0[0]*r1[2],
    r0[0]*r1[1] - r0[1]*r1[0],
  ];
  const norm = Math.hypot(v[0], v[1], v[2]);
  return norm < 1e-14 ? [1, 0, 0] : [v[0]/norm, v[1]/norm, v[2]/norm];
}

function conicToRotatedRect(
  A: number, B: number, C: number, D: number, E: number, F: number,
): RotatedRect | null {
  if (B*B - 4*A*C >= 0) return null; // not an ellipse

  const denom = 4*A*C - B*B; // > 0 for ellipse

  const cx = (B*E - 2*C*D) / denom;
  const cy = (B*D - 2*A*E) / denom;

  const val    = 2 * (A*E*E + C*D*D - B*D*E + (B*B - 4*A*C)*F);
  const common = Math.sqrt((A - C)**2 + B*B);

  const a2 = val * (A + C + common);
  const b2 = val * (A + C - common);
  if (a2 <= 0 || b2 <= 0) return null;

  const a = Math.sqrt(a2) / Math.abs(denom);
  const b = Math.sqrt(b2) / Math.abs(denom);
  if (!isFinite(a) || !isFinite(b)) return null;

  const angle = 0.5 * Math.atan2(B, A - C) * (180 / Math.PI);

  return {
    center: { x: cx, y: cy },
    width:  2 * Math.max(a, b),
    height: 2 * Math.min(a, b),
    angle,
  };
}

function fitEllipseFitzgibbon(points: Pixel[]): RotatedRect | null {
  if (points.length < 6) return null;

  const n = points.length;

  // Normalize for numerical stability
  let mx = 0, my = 0;
  for (const p of points) { mx += p.x; my += p.y; }
  mx /= n; my /= n;

  let scale = 0;
  for (const p of points) scale += Math.hypot(p.x - mx, p.y - my);
  scale /= n;
  if (scale < 1e-10) return null;

  const invScale = 1 / scale;
  const xs = points.map(p => (p.x - mx) * invScale);
  const ys = points.map(p => (p.y - my) * invScale);

  // Scatter matrices S1 (3×3), S2 (3×3), S3 (3×3)
  const S1: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  const S2: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  const S3: number[][] = [[0,0,0],[0,0,0],[0,0,0]];

  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    const d1 = [x*x, x*y, y*y];
    const d2 = [x, y, 1];
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        S1[r][c] += d1[r] * d1[c];
        S2[r][c] += d1[r] * d2[c];
        S3[r][c] += d2[r] * d2[c];
      }
    }
  }

  // T = -S3⁻¹ S2ᵀ
  const S3inv = inv3x3(S3);
  if (!S3inv) return null;

  const S2T = [[S2[0][0], S2[1][0], S2[2][0]],
               [S2[0][1], S2[1][1], S2[2][1]],
               [S2[0][2], S2[1][2], S2[2][2]]];
  const T = matMul3x3(S3inv, S2T).map(row => row.map(v => -v));

  // M = S1 + S2 * T
  const S2T_prod = matMul3x3(S2, T);
  const M: number[][] = S1.map((row, r) => row.map((v, c) => v + S2T_prod[r][c]));

  // M' = C1⁻¹ M, where C1⁻¹ = [[0,0,0.5],[0,-1,0],[0.5,0,0]]
  const C1inv: number[][] = [[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]];
  const Mprime = matMul3x3(C1inv, M);

  // Find eigenvector satisfying ellipse constraint 4ac - b² > 0
  const eigVals = eigenvalues3x3(Mprime);
  let bestVec: number[] | null = null;
  let bestVal = Infinity;

  for (const lambda of eigVals) {
    const v = nullVec3x3(Mprime, lambda);
    const cond = 4 * v[0] * v[2] - v[1] * v[1];
    if (cond > 0 && lambda < bestVal) {
      bestVal = lambda;
      bestVec = v;
    }
  }
  if (!bestVec) return null;

  // a2 = T * a1
  const a2 = matVecMul3(T, bestVec);

  // Conic coefficients in normalized space: [An, Bn, Cn, Dn, En, Fn]
  const [An, Bn, Cn] = bestVec;
  const [Dn, En, Fn] = a2;

  // Denormalize: xn = (x - mx)/s, yn = (y - my)/s
  // A·x² + B·xy + C·y² + D·x + E·y + F = 0 in original coords
  const s = scale, s2 = s * s;
  const A = An / s2;
  const B = Bn / s2;
  const C = Cn / s2;
  const D = (-2*An*mx - Bn*my) / s2 + Dn / s;
  const E = (-2*Cn*my - Bn*mx) / s2 + En / s;
  const F = An*mx*mx/s2 + Bn*mx*my/s2 + Cn*my*my/s2 - Dn*mx/s - En*my/s + Fn;

  return conicToRotatedRect(A, B, C, D, E, F);
}

// ---------------------------------------------------------------------------
// Phase 4e — Interpolate/extrapolate rings with too few sample points
// ---------------------------------------------------------------------------

function interpolateMissingRings(rings: (RotatedRect | null)[]): RotatedRect[] {
  const n = rings.length;
  const result: (RotatedRect | null)[] = [...rings];

  function lerp(a: RotatedRect, b: RotatedRect, t: number): RotatedRect {
    return {
      center: {
        x: a.center.x + t * (b.center.x - a.center.x),
        y: a.center.y + t * (b.center.y - a.center.y),
      },
      width:  Math.max(1, a.width  + t * (b.width  - a.width)),
      height: Math.max(1, a.height + t * (b.height - a.height)),
      angle:  a.angle + t * (b.angle - a.angle),
    };
  }

  // Pass 1: interpolate nulls that have valid neighbours on both sides
  for (let i = 0; i < n; i++) {
    if (result[i]) continue;
    let prevIdx = -1, nextIdx = -1;
    for (let j = i - 1; j >= 0; j--) { if (result[j]) { prevIdx = j; break; } }
    for (let j = i + 1; j < n;  j++) { if (result[j]) { nextIdx = j; break; } }
    if (prevIdx >= 0 && nextIdx >= 0) {
      const t = (i - prevIdx) / (nextIdx - prevIdx);
      result[i] = lerp(result[prevIdx]!, result[nextIdx]!, t);
    }
  }

  // Pass 2: extrapolate leftward nulls (no valid left neighbour)
  for (let i = 0; i < n; i++) {
    if (result[i]) continue;
    // Find two valid rings to the right to extrapolate inward
    let r1 = -1, r2 = -1;
    for (let j = i + 1; j < n; j++) {
      if (result[j]) { if (r1 < 0) r1 = j; else { r2 = j; break; } }
    }
    if (r1 >= 0 && r2 >= 0) {
      const t = (i - r1) / (r2 - r1);
      result[i] = lerp(result[r1]!, result[r2]!, t);
    } else if (r1 >= 0) {
      result[i] = { ...result[r1]!, width: Math.max(1, result[r1]!.width * 0.7) };
    }
  }

  // Pass 3: extrapolate rightward nulls (no valid right neighbour)
  for (let i = n - 1; i >= 0; i--) {
    if (result[i]) continue;
    let l1 = -1, l2 = -1;
    for (let j = i - 1; j >= 0; j--) {
      if (result[j]) { if (l1 < 0) l1 = j; else { l2 = j; break; } }
    }
    if (l1 >= 0 && l2 >= 0) {
      const t = (i - l1) / (l2 - l1);
      result[i] = lerp(result[l1]!, result[l2]!, t);
    } else if (l1 >= 0) {
      result[i] = { ...result[l1]!, width: result[l1]!.width * 1.3 };
    }
  }

  // Final fallback: any remaining nulls get a degenerate default
  const reference = result.find(r => r) ??
    { center: { x: 0, y: 0 }, width: 10, height: 8, angle: 0 };
  return result.map((r, i) => r ?? { ...reference, width: reference.width * (i + 1) });
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
    const N_RAYS = 360;
    const boundary = scanTargetBoundary(
      rgba, width, height, cx, cy, Math.max(w * 4, 20), N_RAYS,
    );
    // Smooth the per-ray distances with a circular median filter (±10 rays)
    // to eliminate single-angle outliers caused by hay-straw spikes or
    // borderline-threshold pixels at isolated angles.
    const smoothedDists = medianFilter1D(boundary.dists, 10);
    const smoothedPoints = smoothedDists.map((d, i) => {
      const theta = (i / N_RAYS) * 2 * Math.PI;
      return { x: Math.round(cx + d * Math.cos(theta)), y: Math.round(cy + d * Math.sin(theta)) };
    });
    if (process.env.DEBUG_RINGS) {
      const medD = arrayMedian(smoothedDists);
      console.error(`[debug] boundary median dist=${medD.toFixed(1)} min=${Math.min(...smoothedDists).toFixed(1)} max=${Math.max(...smoothedDists).toFixed(1)}`);
    }

    // Phase 2 — per-image colour calibration.
    const zoneSamples = sampleZoneColours(rgba, width, height, cx, cy, w, smoothedDists);
    const calibration = computeCalibration(zoneSamples) ?? undefined;

    // Phase 3 — colour-guided ring detection (falls back to luminance if no calibration).
    const rawTransitionPoints = calibration
      ? collectRingPointsColourGuided(rgba, width, height, cx, cy, w, N_RAYS, smoothedDists, calibration)
      : collectRingPoints(rgba, width, height, cx, cy, w, N_RAYS, 0.4, smoothedDists);

    // Erode outlier points: discard any point whose radius from (cx,cy) deviates
    // more than 1.5w from its ring's median radius.  Stray detections (arrow holes,
    // zone-divider noise on single rays) otherwise pull the Fitzgibbon fit off-axis.
    const transitionPoints = rawTransitionPoints.map(
      pts => filterRingOutliers(pts, cx, cy, w * 0.25),
    );

    // Precompute max boundary radius for fit rejection in Phase 4.
    const maxBoundaryR = Math.max(...smoothedDists);

    // Phase 4 — Fitzgibbon algebraic ellipse fit per ring.
    // Rings of a circular target are concentric in image projection — force the
    // bootstrap centre (cx, cy) on every fit so downstream interpolation and the
    // centre-deviation test work correctly.
    const fitted = transitionPoints.map((pts, k) => {
      const r = fitEllipseFitzgibbon(pts);
      if (!r) return null;
      // Reject degenerate fits and those wildly off from the bootstrap expectation.
      // Expected full diameter of ring k: 2*(k+1)*w
      const expectedW = 2 * (k + 1) * w;
      if (r.width <= 0 || r.height <= 0) return null;
      if (r.width / r.height > 6) return null;
      if (r.width < expectedW * 0.25 || r.width > expectedW * 3.0) return null;
      // Hard cap: no ring may extend beyond the detected paper boundary.
      if (r.width / 2 > maxBoundaryR * 1.05) return null;
      return { ...r, center: { x: cx, y: cy } };
    });

    // Phase 4b — Refine w from the successfully fitted rings, then retry any null
    // rings with the refined scale.  The bootstrap w0 can be off by 20-35% when
    // colour blobs are contaminated; the fitted ring radii give a better estimate.
    const wSamples = fitted
      .map((r, k) => (r ? r.width / (2 * (k + 1)) : null))
      .filter((v): v is number => v !== null);
    if (wSamples.length >= 3) {
      const wRefined = arrayMedian(wSamples);
      const relErr = Math.abs(wRefined - w) / Math.max(w, wRefined);
      if (process.env.DEBUG_RINGS) console.error(`[debug] w=${w.toFixed(2)} wRefined=${wRefined.toFixed(2)} relErr=${relErr.toFixed(3)} nulls=${fitted.filter(r=>!r).length}`);
      if (relErr > 0.08) {
        // Re-search only the null rings using the refined scale + wider tolerance.
        const nullIndices = fitted
          .map((r, k) => (r === null ? k : null))
          .filter((k): k is number => k !== null);
        if (nullIndices.length > 0) {
          const refined = calibration
            ? collectRingPointsColourGuided(rgba, width, height, cx, cy, wRefined, N_RAYS, smoothedDists, calibration)
            : collectRingPoints(rgba, width, height, cx, cy, wRefined, N_RAYS, 0.65, smoothedDists);
          for (const k of nullIndices) {
            if (refined[k].length < 6) { if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] only ${refined[k].length} pts`); continue; }
            const r = fitEllipseFitzgibbon(refined[k]);
            if (!r) { if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] fitz failed`); continue; }
            const expectedW = 2 * (k + 1) * wRefined;
            if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] r.w=${r.width.toFixed(1)} expectedW=${expectedW.toFixed(1)} ar=${(r.width/r.height).toFixed(2)}`);
            if (r.width <= 0 || r.height <= 0) continue;
            if (r.width / r.height > 6) continue;
            if (r.width < expectedW * 0.25 || r.width > expectedW * 3.0) continue;
            if (r.width / 2 > maxBoundaryR * 1.05) continue;
            fitted[k] = { ...r, center: { x: cx, y: cy } };
          }
        }
      }
    }

    // Phase 4c — Ring[9] (outermost white ring) boundary fit.
    // Only used when colour calibration is unavailable.  When calibration IS
    // available, detectRingDistancesOnRay already sets the last transition
    // distance to the actual paper boundary per ray, so the colour-guided
    // ring[9] from transitionPoints[9] is already the correct boundary-fit.
    // Running Phase 4c on top would override that with unfiltered boundary
    // scan points, producing an ellipse that routinely exceeds the inradius.
    if (!calibration) {
      const ring8r = fitted[8] ? fitted[8].width / 2 : 0;
      // Keep boundary points that lie outside ring[8] (or at least 10% of
      // the median boundary distance from centre if ring[8] is unknown).
      const minInner = ring8r > 0
        ? ring8r * 1.02
        : arrayMedian(smoothedDists) * 0.10;
      const bPts = smoothedPoints.filter(
        p => Math.hypot(p.x - cx, p.y - cy) > minInner,
      );
      if (process.env.DEBUG_RINGS) {
        console.error(`[debug] ring[9] boundary pts=${bPts.length} ring8r=${ring8r.toFixed(1)}`);
      }
      if (bPts.length >= 6) {
        const bFit = fitEllipseFitzgibbon(bPts);
        if (process.env.DEBUG_RINGS && bFit) {
          console.error(`[debug] ring[9] boundary fit: w=${bFit.width.toFixed(1)} h=${bFit.height.toFixed(1)} ar=${(bFit.width/bFit.height).toFixed(2)}`);
        }
        if (bFit && bFit.width / bFit.height < 6) {
          const currentW = fitted[9]?.width ?? 0;
          // Prefer boundary fit if it is larger (i.e., closer to the paper edge)
          // and at least 5% wider than ring[8].
          if (bFit.width > currentW && bFit.width > (ring8r * 2) * 1.05) {
            fitted[9] = { ...bFit, center: { x: cx, y: cy } };
          }
        }
      }
    }

    // Interpolate/extrapolate nulls, then sort by width ascending.
    // Physical requirement: outer rings must always be wider than inner rings.
    // Enforce a minimum gap so interpolation artifacts don't produce equal-width
    // adjacent rings (which would be physically impossible on a real target).
    const sortedRings = interpolateMissingRings(fitted)
      .sort((a, b) => a.width - b.width);
    for (let i = 1; i < sortedRings.length; i++) {
      if (sortedRings[i].width < sortedRings[i - 1].width + 2) {
        sortedRings[i] = { ...sortedRings[i], width: sortedRings[i - 1].width + 2 };
      }
      if (sortedRings[i].height < sortedRings[i - 1].height + 2) {
        sortedRings[i] = { ...sortedRings[i], height: sortedRings[i - 1].height + 2 };
      }
    }
    const rings = sortedRings;

    // Fit a boundary polygon (4–8 vertices) to the full set of boundary points.
    // The extra vertices capture bowed or folded paper edges that a fixed
    // 4-corner quad cannot represent.
    const paperBoundary = fitBoundaryPolygon(smoothedPoints);

    return {
      rings: rings.map(r => ({
        centerX: r.center.x, centerY: r.center.y,
        width: r.width, height: r.height, angle: r.angle,
      })),
      paperBoundary,
      calibration,
      ringPoints: transitionPoints,
      success: true,
    };
  } catch (e) {
    return { rings: [], success: false, error: String(e) };
  }
}
