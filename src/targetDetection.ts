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

export interface ArcheryResult {
  rings: EllipseData[];
  /**
   * The 4 corners of the target paper boundary as a quadrilateral in image pixels.
   * Ordered: top-left, top-right, bottom-right, bottom-left (in the paper's
   * principal-axis frame — visually a perspective-projected rectangle).
   */
  paperBoundary?: [Pixel, Pixel, Pixel, Pixel];
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
// Phase 3 — Radial profile ring measurement
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
      const [h, s] = rgbToHsv(rgba[pidx], rgba[pidx + 1], rgba[pidx + 2]);
      // Hay bale: brownish-yellow, saturated.  Intentionally exclude the V<0.15
      // criterion so the black scoring zone (dark, not hay bale) is passed through.
      const isBackground = s > 0.25 && h >= 15 && h <= 65;
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
    if (process.env.DEBUG_RINGS) {
      const medD = arrayMedian(boundary.dists);
      console.error(`[debug] boundary median dist=${medD.toFixed(1)} min=${Math.min(...boundary.dists).toFixed(1)} max=${Math.max(...boundary.dists).toFixed(1)}`);
    }

    // Phase 3 — radial-profile ring measurement from original image,
    // capped at the target paper boundary for each ray direction.
    const transitionPoints = collectRingPoints(
      rgba, width, height, cx, cy, w, N_RAYS, 0.4, boundary.dists,
    );

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
          const refined = collectRingPoints(
            rgba, width, height, cx, cy, wRefined, N_RAYS, 0.65, boundary.dists,
          );
          for (const k of nullIndices) {
            if (refined[k].length < 6) { if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] only ${refined[k].length} pts`); continue; }
            const r = fitEllipseFitzgibbon(refined[k]);
            if (!r) { if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] fitz failed`); continue; }
            const expectedW = 2 * (k + 1) * wRefined;
            if (process.env.DEBUG_RINGS) console.error(`[debug] ring[${k}] r.w=${r.width.toFixed(1)} expectedW=${expectedW.toFixed(1)} ar=${(r.width/r.height).toFixed(2)}`);
            if (r.width <= 0 || r.height <= 0) continue;
            if (r.width / r.height > 6) continue;
            if (r.width < expectedW * 0.25 || r.width > expectedW * 3.0) continue;
            fitted[k] = { ...r, center: { x: cx, y: cy } };
          }
        }
      }
    }

    // Phase 4c — Ring[9] (outermost white ring) boundary fit.
    // On a WA target the white zone extends to the paper edge, so the
    // paper boundary IS the outermost ring boundary.  Always attempt a
    // Fitzgibbon fit on the pre-computed boundary points; prefer the
    // boundary-derived ellipse over the radial-profile fit when it is
    // larger (closer to the true paper edge) and well-conditioned.
    {
      const ring8r = fitted[8] ? fitted[8].width / 2 : 0;
      // Keep boundary points that lie outside ring[8] (or at least 10% of
      // the median boundary distance from centre if ring[8] is unknown).
      const minInner = ring8r > 0
        ? ring8r * 1.02
        : arrayMedian(boundary.dists) * 0.10;
      const bPts = boundary.points.filter(
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

    // Fit a quadrilateral to the full set of boundary points so the caller
    // can display the detected target paper edge as a 4-sided polygon.
    const paperBoundary = fitQuadrilateral(boundary.points) ?? undefined;

    return {
      rings: rings.map(r => ({
        centerX: r.center.x, centerY: r.center.y,
        width: r.width, height: r.height, angle: r.angle,
      })),
      paperBoundary,
      success: true,
    };
  } catch (e) {
    return { rings: [], success: false, error: String(e) };
  }
}
