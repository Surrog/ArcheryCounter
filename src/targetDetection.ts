// TypeScript port of the C++ OpenCV archery target detection pipeline.
// See docs/research.md and docs/plan.md for design notes.

interface Pixel { x: number; y: number }

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
  success: boolean;
  error?: string;
}

// ---------------------------------------------------------------------------
// HSV conversion — BGR-as-RGB convention
// ---------------------------------------------------------------------------

/** RGB → HSV_FULL (all 0-255). Swap r↔b to match OpenCV BGR-as-RGB quirk. */
function rgbToHsvFull(r: number, g: number, b: number): [number, number, number] {
  // Replicate COLOR_RGB2HSV_FULL applied to a BGR-loaded image:
  // what OpenCV sees as "R" is our B channel, and vice-versa.
  const rr = b, gg = g, bb = r;

  const max = Math.max(rr, gg, bb);
  const min = Math.min(rr, gg, bb);
  const delta = max - min;

  const v = max;
  const s = max === 0 ? 0 : Math.round((delta * 255) / max);

  let h = 0;
  if (delta > 0) {
    if (max === rr) {
      h = 42.5 * ((gg - bb) / delta);
      if (h < 0) h += 255;
    } else if (max === gg) {
      h = 42.5 * ((bb - rr) / delta) + 85;
    } else {
      h = 42.5 * ((rr - gg) / delta) + 170;
    }
  }
  return [Math.round(h) & 0xff, s, v];
}

// ---------------------------------------------------------------------------
// Color filter — binary mask
// ---------------------------------------------------------------------------

function applyColorFilter(
  rgba: Uint8Array, width: number, height: number,
  hMin: number, hMax: number,
  sMin: number, sMax: number,
  vMin: number, vMax: number,
): Uint8Array {
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const [h, s, v] = rgbToHsvFull(r, g, b);
    if (h >= hMin && h <= hMax && s >= sMin && s <= sMax && v >= vMin && v <= vMax)
      mask[i] = 255;
  }
  return mask;
}

// Thresholds from research.md (H, S, V ranges in HSV_FULL 0-255):
const FILTER_YELLOW = [136, 140, 64, 255, 0, 255] as const;
const FILTER_RED    = [168, 171, 64, 255, 0, 255] as const;
const FILTER_BLUE   = [ 24,  32, 64, 255, 0, 255] as const;

// ---------------------------------------------------------------------------
// Pretreatment — Gaussian blur + erode + dilate
// ---------------------------------------------------------------------------

/** Build a separable Gaussian kernel of given size and σ. */
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

/**
 * Separable 1-D Gaussian convolution on a single uint8 channel.
 * Applied horizontally then vertically.
 */
function gaussianBlurChannel(
  src: Uint8Array, width: number, height: number,
  kernel: Float64Array,
): Uint8Array {
  const half = (kernel.length - 1) / 2;
  const tmp = new Uint8Array(src.length);
  const dst = new Uint8Array(src.length);

  // Horizontal pass
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

  // Vertical pass
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

/**
 * Morphological erode/dilate for a single uint8 channel (3×3 rectangular kernel).
 * mode 'erode' = replace with min of neighborhood; 'dilate' = max.
 * Uses 3×3 box (all 8 neighbors + center) to match cv::Mat() default kernel.
 */
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

// Computed once at module load time
const GAUSS_KERNEL = gaussianKernel(15, 1.5);

/**
 * pretreatment(): blur + erode×1 + dilate×3 on the RGBA image's color channels.
 * Returns a new RGBA buffer; A channel is copied unchanged.
 */
function pretreat(rgba: Uint8Array, width: number, height: number): Uint8Array {
  const n = width * height;

  // Extract R, G, B channels
  let r = new Uint8Array(n), g = new Uint8Array(n), b = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = rgba[i * 4]; g[i] = rgba[i * 4 + 1]; b[i] = rgba[i * 4 + 2];
  }

  // Process each channel identically
  function process(ch: Uint8Array): Uint8Array {
    let c = gaussianBlurChannel(ch, width, height, GAUSS_KERNEL);
    c = morphChannel(c, width, height, 'erode');
    for (let i = 0; i < 3; i++) c = morphChannel(c, width, height, 'dilate');
    return c;
  }
  r = process(r); g = process(g); b = process(b);

  // Repack into RGBA
  const out = new Uint8Array(rgba); // copy alpha
  for (let i = 0; i < n; i++) {
    out[i * 4] = r[i]; out[i * 4 + 1] = g[i]; out[i * 4 + 2] = b[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Largest blob (BFS flood fill)
// ---------------------------------------------------------------------------

function findLargestBlob(mask: Uint8Array, width: number, height: number): Uint8Array {
  const labels = new Int32Array(mask.length).fill(-1);
  const sizes: number[] = [];
  let label = 0;

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || labels[start] !== -1) continue;

    const queue: number[] = [start];
    labels[start] = label;
    let head = 0;
    while (head < queue.length) {
      const idx = queue[head++];
      const px = idx % width, py = (idx / width) | 0;
      for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nx = px + dx, ny = py + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const ni = ny * width + nx;
        if (mask[ni] && labels[ni] === -1) { labels[ni] = label; queue.push(ni); }
      }
    }
    sizes.push(queue.length);
    label++;
  }

  if (sizes.length === 0) return new Uint8Array(mask.length);

  const best = sizes.indexOf(Math.max(...sizes));
  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) if (labels[i] === best) out[i] = 255;
  return out;
}

// ---------------------------------------------------------------------------
// Blob aggregation (mirrors C++ aggregate_contour)
// ---------------------------------------------------------------------------

/**
 * Label all connected components (BFS), find largest component, compute its
 * centroid and mean_distance. Merge any component (> 6 pixels) whose centroid
 * is within 2.5 × mean_distance of the largest component centroid.
 */
function aggregateBlobs(
  mask: Uint8Array, width: number, height: number,
): Uint8Array {
  // Label all connected components via BFS
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

  if (componentPixels.length === 0) return new Uint8Array(mask.length);

  // Find the largest component
  let largestIdx = 0;
  for (let i = 1; i < componentPixels.length; i++) {
    if (componentPixels[i].length > componentPixels[largestIdx].length) largestIdx = i;
  }

  const largestPixels = componentPixels[largestIdx];

  // Compute centroid of largest component
  let sumX = 0, sumY = 0;
  for (const idx of largestPixels) {
    sumX += idx % width;
    sumY += (idx / width) | 0;
  }
  const cx = sumX / largestPixels.length;
  const cy = sumY / largestPixels.length;

  // Compute mean distance of all pixels in largest component from centroid
  let totalDist = 0;
  for (const idx of largestPixels) {
    const px = idx % width;
    const py = (idx / width) | 0;
    totalDist += Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
  }
  const meanDist = totalDist / largestPixels.length;

  // Build merged mask: include largest + any component with > 6 pixels whose centroid
  // is within 2.5 × meanDist of the largest centroid
  const threshold = 2.5 * meanDist;
  const out = new Uint8Array(mask.length);

  for (let c = 0; c < componentPixels.length; c++) {
    const pixels = componentPixels[c];
    if (c === largestIdx) {
      // Always include the largest
      for (const idx of pixels) out[idx] = 255;
      continue;
    }
    if (pixels.length <= 6) continue;

    // Compute centroid of this component
    let csx = 0, csy = 0;
    for (const idx of pixels) {
      csx += idx % width;
      csy += (idx / width) | 0;
    }
    const compCx = csx / pixels.length;
    const compCy = csy / pixels.length;

    // Check if centroid is within 2.5 * meanDist
    const dist = Math.sqrt((compCx - cx) ** 2 + (compCy - cy) ** 2);
    if (dist <= threshold) {
      for (const idx of pixels) out[idx] = 255;
    }
  }

  return out;
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
// PCA ellipse fit
// ---------------------------------------------------------------------------

function fitEllipsePCA(points: Pixel[]): RotatedRect | null {
  if (points.length < 6) return null;

  const n = points.length;
  let mx = 0, my = 0;
  for (const p of points) { mx += p.x; my += p.y; }
  mx /= n; my /= n;

  let cxx = 0, cxy = 0, cyy = 0;
  for (const p of points) {
    const dx = p.x - mx, dy = p.y - my;
    cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
  }
  cxx /= n; cxy /= n; cyy /= n;

  // Eigenvalues of [[cxx, cxy], [cxy, cyy]]
  const mid = (cxx + cyy) / 2;
  const disc = Math.sqrt(Math.max(0, ((cxx - cyy) / 2) ** 2 + cxy * cxy));
  const lambda1 = mid + disc; // larger  → major axis
  const lambda2 = mid - disc; // smaller → minor axis

  // Semi-axis a = sqrt(2λ), full axis (diameter) = 2a = 2√(2λ)
  const width  = 2 * Math.sqrt(2 * Math.max(0, lambda1));
  const height = 2 * Math.sqrt(2 * Math.max(0, lambda2));

  // Angle of major axis in degrees
  const angle = Math.atan2(lambda1 - cxx, cxy) * (180 / Math.PI);

  return { center: { x: mx, y: my }, width, height, angle };
}

// ---------------------------------------------------------------------------
// Contour cleanup (mirrors cleanup_center_points)
// ---------------------------------------------------------------------------

function cleanupContourPoints(points: Pixel[]): Pixel[] {
  const MAX_ITER = 8;
  const STEP = 0.03;
  const LOWER_START = 0.48 - STEP * MAX_ITER; // 0.24
  const UPPER_START = 0.58 + STEP * MAX_ITER; // 0.82

  let pts = [...points];
  for (let iter = 0; iter < 12 && pts.length > 5; iter++) {
    const ellipse = fitEllipsePCA(pts);
    if (!ellipse) break;

    const t = Math.min(iter, MAX_ITER);
    const lo = ellipse.width * (LOWER_START + STEP * t);
    const hi = ellipse.width * (UPPER_START - STEP * t);

    const newPts = pts.filter(p => {
      const d = Math.hypot(p.x - ellipse.center.x, p.y - ellipse.center.y);
      return d >= lo && d <= hi;
    });

    // Early stop: if no points were removed, the set is stable
    if (newPts.length === pts.length) break;

    pts = newPts;
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Weakest element detection (direct C++ port)
// ---------------------------------------------------------------------------

function findWeakestElement(ellipses: RotatedRect[]): number {
  // Immediately eliminate any failed detection (zero rect)
  for (let i = 0; i < ellipses.length; i++) {
    if (ellipses[i].width === 0 && ellipses[i].height === 0) return i;
  }

  const bad = [0, 0, 0];

  // Criterion 1: distance from centroid of all 3 centers
  const cx = (ellipses[0].center.x + ellipses[1].center.x + ellipses[2].center.x) / 3;
  const cy = (ellipses[0].center.y + ellipses[1].center.y + ellipses[2].center.y) / 3;
  const dists = ellipses.map(e => Math.hypot(e.center.x - cx, e.center.y - cy));
  const worstDist = dists.indexOf(Math.max(...dists));
  if (dists[worstDist] > 20) bad[worstDist]++;

  // Criterion 2: angle deviation from mean (always adds 1 to worst)
  const meanAngle = (ellipses[0].angle + ellipses[1].angle + ellipses[2].angle) / 3;
  const angleDiffs = ellipses.map(e => Math.abs(e.angle - meanAngle));
  bad[angleDiffs.indexOf(Math.max(...angleDiffs))]++;

  // Criterion 3: aspect ratio deviation from mean
  const ratios = ellipses.map(e => e.width / Math.max(e.height, 1e-6));
  const meanRatio = (ratios[0] + ratios[1] + ratios[2]) / 3;
  const ratioDiffs = ratios.map(r => Math.abs(r - meanRatio));
  const worstRatio = ratioDiffs.indexOf(Math.max(...ratioDiffs));
  if (ratioDiffs[worstRatio] > 0.01) bad[worstRatio]++;

  const maxBad = Math.max(...bad);
  if (maxBad > 1) return bad.indexOf(maxBad);
  return 3; // no clear weakest → use all 3
}

// ---------------------------------------------------------------------------
// Linear interpolation to 10 rings (direct C++ port)
// ---------------------------------------------------------------------------

function linearInterpolateTo10Rings(
  ellipses: RotatedRect[], ignoredIdx: number,
): RotatedRect[] {
  const valid = ellipses
    .map((e, i) => ({ e, x: i }))
    .filter((_, i) => i !== ignoredIdx)
    // Also skip zero-size ellipses (failed detections)
    .filter(({ e }) => e.width > 0 && e.height > 0);
  const n = valid.length;

  // Handle degenerate case: no valid ellipses at all
  if (n === 0) {
    return Array.from({ length: 10 }, (_, i) => ({
      center: { x: 0, y: 0 }, width: 1, height: 1, angle: 0,
    }));
  }

  const xSum  = valid.reduce((s, { x }) => s + x, 0);
  const xxSum = valid.reduce((s, { x }) => s + x * x, 0);
  const xSq   = n * xxSum - xSum * xSum; // denominator for regression

  function regress(values: number[]): (x: number) => number {
    const ySum  = values.reduce((s, v) => s + v, 0);
    const xySum = valid.reduce((s, { x }, i) => s + x * values[i], 0);
    if (xSq === 0) return () => ySum / n; // degenerate: all same x
    const coef     = (n * xySum - xSum * ySum) / xSq;
    const constant = (ySum - coef * xSum) / n;
    return (x: number) => coef * x + constant;
  }

  const cxFn = regress(valid.map(({ e }) => e.center.x));
  const cyFn = regress(valid.map(({ e }) => e.center.y));
  const wFn  = regress(valid.map(({ e }) => e.width));
  const hFn  = regress(valid.map(({ e }) => e.height));
  const aFn  = regress(valid.map(({ e }) => e.angle));

  return Array.from({ length: 10 }, (_, i) => {
    const x = i * 0.5 - 0.5;
    return {
      center: { x: cxFn(x), y: cyFn(x) },
      width:  Math.max(1, wFn(x)),
      height: Math.max(1, hFn(x)),
      angle:  aFn(x),
    };
  });
}

// ---------------------------------------------------------------------------
// Main findTarget function
// ---------------------------------------------------------------------------

export function findTarget(
  rgba: Uint8Array, width: number, height: number,
): ArcheryResult {
  try {
    const pretreated = pretreat(rgba, width, height);

    function detect(hMin: number, hMax: number, sMin: number, sMax: number, vMin: number, vMax: number): RotatedRect {
      const mask = applyColorFilter(pretreated, width, height,
        hMin, hMax, sMin, sMax, vMin, vMax);
      const blob = aggregateBlobs(mask, width, height);
      const boundary = extractBoundary(blob, width, height);
      const cleaned  = cleanupContourPoints(boundary);
      return fitEllipsePCA(cleaned) ?? { center: { x: 0, y: 0 }, width: 0, height: 0, angle: 0 };
    }

    // Order matches C++: yellow=0, red=1, blue=2
    const ellipses: RotatedRect[] = [
      detect(...FILTER_YELLOW),
      detect(...FILTER_RED),
      detect(...FILTER_BLUE),
    ];

    const weakest = findWeakestElement(ellipses);
    const rings   = linearInterpolateTo10Rings(ellipses, weakest);

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
