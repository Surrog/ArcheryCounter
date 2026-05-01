/**
 * Neural-network paper boundary detector (Ph-14).
 *
 * Runs a ResNet50+UNet segmentation model (ONNX) on a grayscale letterboxed
 * 512×512 image and returns one polygon per detected paper target.
 *
 * Post-processing pipeline:
 *   1. Sigmoid → threshold at 0.5 → binary mask
 *   2. Connected components (flood-fill), discard regions < MIN_AREA_PX
 *   3. Moore-neighbor contour trace per component
 *   4. Douglas-Peucker simplification to ≤ MAX_VERTICES vertices (simplify-js)
 *   5. Map vertices from model-input coords back to original image coords
 *
 * Fallback: if no region's mean sigmoid confidence exceeds CONFIDENCE_THRESHOLD,
 * returns null — the caller should fall back to the ray-scan boundary.
 *
 * Environment:
 *   React Native app  → onnxruntime-react-native
 *   Node.js scripts   → onnxruntime-node
 */

import simplify from 'simplify-js';
import { connectedComponents } from './connectedComponents';
import { traceContour } from './contourTrace';
import { getSession, OnnxSession, releaseSession } from './ortComponents';

// ── constants (must match training) ──────────────────────────────────────────

const INPUT_SIZE          = 512;
const SIGMOID_THRESHOLD   = 0.5;
const CONFIDENCE_THRESHOLD = 0.6;   // mean confidence below this → fallback
const MIN_AREA_PX         = Math.round(INPUT_SIZE * INPUT_SIZE * 0.01); // 1% ≈ 2621 px
const MAX_VERTICES        = 10;
const SIMPLIFY_TOLERANCE  = 3.0;    // Douglas-Peucker tolerance in model-input px

const IMAGENET_GRAY_MEAN = 0.449;
const IMAGENET_GRAY_STD  = 0.226;

// ── letterbox (grayscale) ─────────────────────────────────────────────────────

interface LetterboxResult {
  data:  Float32Array;   // (1, INPUT_SIZE, INPUT_SIZE) NCHW
  scale: number;
  padX:  number;
  padY:  number;
}

/**
 * Letterbox-resize an RGBA buffer to INPUT_SIZE×INPUT_SIZE, convert to
 * normalised grayscale Float32Array in NCHW order (1 channel).
 */
export function letterboxGray(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
): LetterboxResult {
  if (srcW <= 0 || srcH <= 0) throw new Error(`Invalid image dimensions: ${srcW}×${srcH}`);
  const scale = INPUT_SIZE / Math.max(srcW, srcH);
  const newW  = Math.round(srcW * scale);
  const newH  = Math.round(srcH * scale);
  const padX  = Math.floor((INPUT_SIZE - newW) / 2);
  const padY  = Math.floor((INPUT_SIZE - newH) / 2);

  const data = new Float32Array(INPUT_SIZE * INPUT_SIZE);

  for (let dy = 0; dy < INPUT_SIZE; dy++) {
    for (let dx = 0; dx < INPUT_SIZE; dx++) {
      const sx = (dx - padX) / scale;
      const sy = (dy - padY) / scale;
      let gray = 0;
      if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(x0 + 1, srcW - 1), y1 = Math.min(y0 + 1, srcH - 1);
        const fx = sx - x0, fy = sy - y0;
        const idx = (r: number, c: number) => (r * srcW + c) * 4;
        const lum = (r: number, g: number, b: number) => 0.299 * r + 0.587 * g + 0.114 * b;
        const v00 = lum(rgba[idx(y0, x0)], rgba[idx(y0, x0) + 1], rgba[idx(y0, x0) + 2]);
        const v01 = lum(rgba[idx(y0, x1)], rgba[idx(y0, x1) + 1], rgba[idx(y0, x1) + 2]);
        const v10 = lum(rgba[idx(y1, x0)], rgba[idx(y1, x0) + 1], rgba[idx(y1, x0) + 2]);
        const v11 = lum(rgba[idx(y1, x1)], rgba[idx(y1, x1) + 1], rgba[idx(y1, x1) + 2]);
        gray = (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                v10 * (1 - fx) * fy        + v11 * fx * fy);
        gray /= 255;
      }
      data[dy * INPUT_SIZE + dx] = (gray - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD;
    }
  }

  return { data, scale, padX, padY };
}

// ── post-processing ───────────────────────────────────────────────────────────

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Convert logit map → polygons in original image coordinates.
 * Returns null if confidence is too low (caller should fall back to ray-scan).
 */
export function extractPolygons(
  logits: Float32Array,
  scale: number,
  padX: number,
  padY: number,
): [number, number][][] | null {
  const n = INPUT_SIZE * INPUT_SIZE;

  // Convert logits → probabilities; check overall confidence
  const probs = new Float32Array(n);
  for (let i = 0; i < n; i++) probs[i] = sigmoid(logits[i]);

  // Binary mask
  const binaryMask = new Uint8Array(n);
  let foregroundCount = 0;
  let foregroundConfSum = 0;
  for (let i = 0; i < n; i++) {
    if (probs[i] >= SIGMOID_THRESHOLD) {
      binaryMask[i] = 1;
      foregroundCount++;
      foregroundConfSum += probs[i];
    }
  }

  if (foregroundCount === 0) return null;

  const meanConf = foregroundConfSum / foregroundCount;
  if (meanConf < CONFIDENCE_THRESHOLD) return null;

  // Connected components
  const components = connectedComponents(binaryMask, INPUT_SIZE, INPUT_SIZE);
  const largeComponents = components.filter(c => c.length >= MIN_AREA_PX);
  if (largeComponents.length === 0) return null;

  const polygons: [number, number][][] = [];

  for (const component of largeComponents) {
    // Trace contour in model-input coordinates
    const contour = traceContour(binaryMask, INPUT_SIZE, INPUT_SIZE, component);
    if (contour.length < 3) continue;

    // Simplify with Douglas-Peucker (simplify-js uses {x, y} objects)
    const pts = contour.map(([x, y]) => ({ x, y }));
    const simplified = simplify(pts, SIMPLIFY_TOLERANCE, true);

    // Enforce max vertices by increasing tolerance until under limit
    let result = simplified;
    let tol = SIMPLIFY_TOLERANCE;
    while (result.length > MAX_VERTICES && tol < 50) {
      tol *= 1.5;
      result = simplify(pts, tol, true);
    }

    if (result.length < 3) continue;

    // Map from model-input coordinates back to original image coordinates
    const poly: [number, number][] = result.map(({ x, y }) => [
      (x - padX) / scale,
      (y - padY) / scale,
    ]);

    polygons.push(poly);
  }

  return polygons.length > 0 ? polygons : null;
}

// ── public API ────────────────────────────────────────────────────────────────

let sessionCache: OnnxSession | null = null;

/**
 * Detect paper boundary polygons in an image using the segmentation NN.
 *
 * @param rgba       Raw RGBA pixel buffer of the original image.
 * @param width      Image width in pixels.
 * @param height     Image height in pixels.
 * @param modelPath  Path to boundary_detector.onnx.
 * @returns Array of polygons (one per target) in original image coordinates,
 *          or null if confidence is too low (caller should fall back to ray-scan).
 */
export async function detectBoundaries(
  rgba: Uint8Array,
  width: number,
  height: number,
  modelPath: string,
): Promise<[number, number][][] | null> {
  const { data, scale, padX, padY } = letterboxGray(rgba, width, height);

  if (!sessionCache || sessionCache.currentModelPath !== modelPath) {
    if (sessionCache && !sessionCache.isReleased) releaseSession(sessionCache);
    sessionCache = await getSession(modelPath);
  }

  const tensor = new sessionCache.ort.Tensor('float32', data, [1, 1, INPUT_SIZE, INPUT_SIZE]);
  const results = await sessionCache.session.run({ image: tensor });
  if (!results['logits']) {
    throw new Error(
      'Model output "logits" not found. Check the model was exported with output name "logits". ' +
      `Available outputs: ${Object.keys(results).join(', ')}`,
    );
  }
  const logits = results['logits'].data as Float32Array;

  return extractPolygons(logits, scale, padX, padY);
}

export function releaseBoundarySession(): void {
  if (sessionCache && !sessionCache.isReleased) {
    releaseSession(sessionCache);
    sessionCache = null;
  }
}
