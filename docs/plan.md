# TypeScript Migration Plan

Port the C++ OpenCV target-detection pipeline to pure TypeScript with no native
dependencies for the algorithm. Faithful reimplementation based on `docs/research.md`.

---

## Scope

### Files created
| File | Role |
|------|------|
| `src/targetDetection.ts` | Core algorithm (~500 lines) |
| `src/imageLoader.ts` | Acquire RGBA pixel buffer (RN + Node.js) |
| `src/ArcheryCounter.ts` | Public API; replaces `NativeArcheryCounter.ts` |
| `src/__tests__/targetDetection.test.ts` | Integration tests using `images/*.jpg` |

### Files deleted
| Path | Reason |
|------|--------|
| `native/` (except `native/Images/`) | C++ source no longer needed |
| `cpp/` | Platform-independent C++ wrapper |
| `ios/ArcheryCounterBridge/` | Obj-C++ native module bridge |
| `android/app/src/main/jni/` | JNI bridge and CMakeLists |
| `ArcheryCounterNative.podspec` | CocoaPods spec for the native module |
| `android/app/src/test/…/ArcheryCounterModuleTest.java` | Tests the deleted JNI |

### Files moved
`native/Images/` → `images/` (root level, kept for algorithm tests)

### Files updated
| File | Change |
|------|--------|
| `src/NativeArcheryCounter.ts` | Replaced by `src/ArcheryCounter.ts` |
| `src/useArcheryScorer.ts` | Use new TS module; add `includeBase64: true` |
| `src/__mocks__/NativeArcheryCounter.ts` | Update import path |
| `android/app/build.gradle` | Remove `externalNativeBuild`, JNI, OpenCV |
| `ios/Podfile` | Remove `ArcheryCounterNative` pod |
| `package.json` | Add `jpeg-js`; add `jimp` + `@types/jimp` to devDeps |
| `.github/workflows/ci.yml` | Remove C++ Linux/Windows jobs; keep Android/iOS jobs but strip OpenCV steps |
| `CLAUDE.md` | Fix ring-ordering note (index 0 = innermost, not outermost) |

---

## New packages

```jsonc
// dependencies (shipped in the app)
"jpeg-js": "^0.4.4"   // pure-JS JPEG decoder, works in Hermes

// devDependencies (Node.js/Jest only)
"jimp": "^0.22.12"    // image loading for integration tests
"@types/jimp": "*"
```

---

## Architecture

### Image loading (two paths)

```
React Native app                         Jest integration tests
─────────────────                        ─────────────────────
launchImageLibrary({includeBase64:true}) Jimp.read(filePath)
        │                                        │
        ▼                                        ▼
base64 string                            img.bitmap.data (RGBA Buffer)
        │                                        │
jpeg.decode(base64ToBytes(b64))          Uint8Array(img.bitmap.data.buffer)
        │                                        │
        └──────────────┬──────────────────────────┘
                       ▼
           { rgba: Uint8Array, width, height }
                       │
                       ▼
               findTarget(rgba, width, height)
                       │
                       ▼
                 EllipseData[10]
```

### Algorithm pipeline (mirrors C++)

```
rgba (RGBA, 4 bytes/pixel)
    │
    ▼  pretreat()
Gaussian blur 15×15 σ=1.5  ← per R,G,B channel independently
erode 3×3 × 1              ← per channel
dilate 3×3 × 3             ← per channel
    │
    ▼  (3 parallel calls)
filterYellow / filterRed / filterBlue
  rgbToHsvFull(b, g, r)    ← note: R↔B swap to replicate BGR-as-RGB quirk
  inRange(H,S,V)
  → binary mask (Uint8Array, 0 or 255)
    │
    ▼  detectEllipseFromMask()
findLargestBlob (BFS flood fill)
aggregateNearbyBlobs (merge blobs within 2.5 × mean_distance)
extractBoundary (4-connected)
cleanupContourPoints (iterative outlier removal)
fitEllipsePCA (2×2 covariance → closed-form eigenvalues)
  → RotatedRect
    │
    ▼  (3 RotatedRects: yellow, red, blue)
findWeakestElement()       ← score on center dist / angle / aspect ratio
    │
    ▼
linearInterpolateTo10Rings()  ← least-squares regression; x = i*0.5 − 0.5
    │
    ▼
EllipseData[10]  (index 0 = innermost/bullseye, index 9 = outermost)
```

---

## Implementation: `src/targetDetection.ts`

### 1. Types

```typescript
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
```

---

### 2. HSV conversion — BGR-as-RGB convention

The C++ code calls `COLOR_RGB2HSV_FULL` on a BGR image, effectively swapping R↔B.
We replicate this by passing `(b, g, r)` to our HSV function (the caller feeds RGB).

```typescript
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
```

---

### 3. Color filter — binary mask

```typescript
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
```

---

### 4. Pretreatment — Gaussian blur + erode + dilate

Operates on the full color image (R, G, B channels independently; A ignored).

```typescript
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
 * Morphological erode/dilate for a single uint8 channel (3×3 cross kernel).
 * mode 'erode' = replace with min of neighborhood; 'dilate' = max.
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

const GAUSS_KERNEL = gaussianKernel(15, 1.5); // computed once

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
```

---

### 5. Largest blob (BFS flood fill)

```typescript
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
```

---

### 6. Blob aggregation (mirrors C++ `aggregate_contour`)

Finds other blobs whose centroid lies within `2.5 × mean_distance` of the largest blob's
centroid. Merges their pixels into one mask.

```typescript
function aggregateBlobs(
  mask: Uint8Array, width: number, height: number,
): Uint8Array {
  // Label all connected components
  // For each component, compute centroid
  // Largest component → reference centroid, mean_distance of its pixels
  // Merge any component whose centroid is within 2.5 × mean_distance
  // Return merged mask
  // (implementation uses the same BFS approach as findLargestBlob)
}
```

---

### 7. Boundary extraction

```typescript
function extractBoundary(blob: Uint8Array, width: number, height: number): Pixel[] {
  const pts: Pixel[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!blob[i]) continue;
      if (
        (x > 0         && !blob[i - 1]) ||
        (x < width - 1 && !blob[i + 1]) ||
        (y > 0         && !blob[i - width]) ||
        (y < height - 1 && !blob[i + width])
      ) pts.push({ x, y });
    }
  }
  return pts;
}
```

---

### 8. Contour cleanup (mirrors `cleanup_center_points`)

Iteratively removes points outside an annular band around the fitted ellipse center.
The band starts wide ([24%, 82%] of ellipse width) and converges to [48%, 58%].

```typescript
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

    pts = pts.filter(p => {
      const d = Math.hypot(p.x - ellipse.center.x, p.y - ellipse.center.y);
      return d >= lo && d <= hi;
    });
  }
  return pts;
}
```

---

### 9. PCA ellipse fit

For points uniformly distributed on an ellipse boundary, the eigenvalues of the 2×2
covariance matrix satisfy `λ = (semi-axis)² / 2`.  The 2×2 eigenvalue problem has a
closed-form solution — no external linear-algebra library needed.

```typescript
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
```

---

### 10. Weakest element detection (direct C++ port)

```typescript
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
```

---

### 11. Linear interpolation to 10 rings (direct C++ port)

x-values: yellow=0, red=1, blue=2. Output rings evaluate at `x = i * 0.5 − 0.5`.

```typescript
function linearInterpolateTo10Rings(
  ellipses: RotatedRect[], ignoredIdx: number,
): RotatedRect[] {
  const valid = ellipses
    .map((e, i) => ({ e, x: i }))
    .filter((_, i) => i !== ignoredIdx);
  const n = valid.length;

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
      width:  wFn(x),
      height: hFn(x),
      angle:  aFn(x),
    };
  });
}
```

---

### 12. Main `findTarget` function

```typescript
export function findTarget(
  rgba: Uint8Array, width: number, height: number,
): ArcheryResult {
  try {
    const pretreated = pretreat(rgba, width, height);

    function detect(hMin: number, hMax: number, sMin: number): RotatedRect {
      const mask = applyColorFilter(pretreated, width, height,
        hMin, hMax, sMin, 255, 0, 255);
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
```

---

## Implementation: `src/imageLoader.ts`

Abstracts image decoding. Returns a normalized `{ rgba, width, height }` object.

```typescript
export interface ImageBuffer {
  rgba: Uint8Array;
  width: number;
  height: number;
}

/**
 * Decode a base64 JPEG string to an RGBA pixel buffer.
 * Used in the React Native app (jpeg-js, works in Hermes).
 */
export function decodeBase64Jpeg(base64: string): ImageBuffer {
  // atob is available in React Native / Hermes
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const jpeg = require('jpeg-js') as typeof import('jpeg-js');
  const { data, width, height } = jpeg.decode(bytes, { useTrichromaticQuantization: false });
  return { rgba: new Uint8Array(data.buffer), width, height };
}

/**
 * Load a JPEG from the filesystem for Node.js / Jest tests.
 * NOT imported in the React Native bundle (jimp is a devDependency).
 */
export async function loadImageNode(filePath: string): Promise<ImageBuffer> {
  const Jimp = (await import('jimp')).default;
  const img  = await Jimp.read(filePath);
  return {
    rgba: new Uint8Array(img.bitmap.data.buffer),
    width: img.bitmap.width,
    height: img.bitmap.height,
  };
}
```

---

## Implementation: `src/ArcheryCounter.ts`

Replaces `NativeArcheryCounter.ts`. No `NativeModules` — calls TS directly.

```typescript
import { findTarget } from './targetDetection';
import { decodeBase64Jpeg } from './imageLoader';
import type { EllipseData } from './targetDetection';

export type { EllipseData as RingEllipse };

const ArcheryCounter = {
  async processImage(imageUri: string, base64: string): Promise<EllipseData[]> {
    const { rgba, width, height } = decodeBase64Jpeg(base64);
    const result = findTarget(rgba, width, height);
    if (!result.success) throw new Error(result.error ?? 'Detection failed');
    return result.rings;
  },
};

export default ArcheryCounter;
```

---

## Implementation: `src/useArcheryScorer.ts` changes

```typescript
// Before:
const rings = await ArcheryCounter.processImage(uri);

// After (adds includeBase64: true to the picker call):
const pickerResult = await launchImageLibrary({
  mediaType: 'photo',
  includeBase64: true,   // ← new
  maxWidth: 1200,        // ← resize to limit processing time
  maxHeight: 1200,
});
// …
const rings = await ArcheryCounter.processImage(uri, asset.base64!);
```

---

## Implementation: `src/__tests__/targetDetection.test.ts`

Uses `jimp` (devDep, Node.js only) to load the test images from `images/`.

```typescript
import * as path from 'path';
import * as fs from 'fs';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR = path.resolve(__dirname, '../../../images');

const jpgFiles = fs.readdirSync(IMAGES_DIR)
  .filter(f => f.endsWith('.jpg'))
  .map(f => path.join(IMAGES_DIR, f));

describe('findTarget', () => {
  test.each(jpgFiles)('%s — detects 10 concentric rings', async (imgPath) => {
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);
    expect(result.rings).toHaveLength(10);

    for (const ring of result.rings) {
      expect(ring.width).toBeGreaterThan(0);
      expect(ring.height).toBeGreaterThan(0);
      expect(ring.centerX).toBeGreaterThanOrEqual(0);
      expect(ring.centerX).toBeLessThanOrEqual(width);
      expect(ring.centerY).toBeGreaterThanOrEqual(0);
      expect(ring.centerY).toBeLessThanOrEqual(height);
    }

    // All centers within 100 px of ring[0] center
    const { centerX: ox, centerY: oy } = result.rings[0];
    for (const ring of result.rings) {
      const d = Math.hypot(ring.centerX - ox, ring.centerY - oy);
      expect(d).toBeLessThanOrEqual(100);
    }
  }, 30_000); // each image may take a few seconds in JS
});
```

---

## CI changes

### Removed jobs
- `linux` (C++ desktop build with scan-build)
- `windows` (C++ desktop build)

### Android job — changes only
Remove:
- OpenCV SDK download/cache step
- `OPENCV_ANDROID_SDK` env var references
- `assembleDebug` step (the JNI/CMake build; no longer needed)

Keep:
- `testDebugUnitTest` step (but `ArcheryCounterModuleTest.java` is deleted, so this runs
  whatever Android unit tests remain — currently none, but the step is harmless)

### iOS job — changes only
Remove:
- `pod install` is kept but ArcheryCounterNative is gone so it will be faster
- The `xcodebuild` step remains; it now builds without the C++ native module

---

## Performance notes

The Gaussian blur (15×15 kernel on a full-resolution image) is the bottleneck.
A 12 MP image (4000×3000) requires ~4000×3000×15×2 ≈ 360 M multiply-adds per channel.
In JavaScript this will be slow (~5–15 s). Two mitigations:

1. **`maxWidth: 1200, maxHeight: 1200`** in the image picker → reduces to ~1.4 M pixels.
   This is sufficient for ring detection accuracy.
2. **WASM future path**: the same TypeScript can be compiled to WASM with AssemblyScript
   or Emscripten if performance becomes unacceptable.

The test timeout is set to 30 s per image to accommodate full-resolution test images.

---

## Implementation order

1. Move `native/Images/` → `images/`
2. Create `src/targetDetection.ts` (all functions above)
3. Create `src/imageLoader.ts`
4. Create `src/__tests__/targetDetection.test.ts`
5. Run `npm test` — iterate on thresholds / algorithm until all images pass
6. Create `src/ArcheryCounter.ts`, update `useArcheryScorer.ts`
7. Delete C++ files, update `build.gradle`, `Podfile`, CI
8. Update `CLAUDE.md` (ring ordering fix)
