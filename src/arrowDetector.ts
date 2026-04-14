/**
 * Neural-network arrow tip detector (P11-T2b).
 *
 * Wraps the exported ONNX model (arrow_detector_fp32.onnx) with the same
 * letterbox pre-processing used during training and the standard heatmap-NMS
 * post-processing to return tip positions in original image coordinates.
 *
 * The ONNX session is created lazily and cached for the lifetime of the process.
 * Call `releaseSession()` to free it explicitly (e.g. on app background).
 *
 * Environment:
 *   - React Native app  → onnxruntime-react-native  (bundled as a static asset)
 *   - Node.js scripts   → onnxruntime-node           (loaded from the filesystem)
 */

import type { ScoredArrow } from './scoring';

// ── constants (must match training) ──────────────────────────────────────────

const INPUT_SIZE    = 640;
const HEATMAP_SIZE  = 160;
const RATIO         = INPUT_SIZE / HEATMAP_SIZE;   // 4.0
const THRESHOLD     = 0.35;
const NMS_KERNEL    = 5;

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

// ── letterbox ────────────────────────────────────────────────────────────────

interface LetterboxResult {
  data:  Float32Array;   // (3, INPUT_SIZE, INPUT_SIZE) NCHW — RGB
  scale: number;
  padX:  number;
  padY:  number;
}

/**
 * Letterbox-resize an RGBA pixel buffer to INPUT_SIZE×INPUT_SIZE.
 * Returns a normalised Float32Array in NCHW order (3 channels, RGB).
 */
export function letterboxRgba(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
): LetterboxResult {
  if (srcW <= 0 || srcH <= 0) throw new Error(`Invalid image dimensions: ${srcW}×${srcH}`);
  const scale  = INPUT_SIZE / Math.max(srcW, srcH);
  const newW   = Math.round(srcW * scale);
  const newH   = Math.round(srcH * scale);
  const padX   = Math.floor((INPUT_SIZE - newW) / 2);
  const padY   = Math.floor((INPUT_SIZE - newH) / 2);

  // Float32 buffer: 3 channels × INPUT_SIZE × INPUT_SIZE
  const nCh   = 3;
  const data  = new Float32Array(nCh * INPUT_SIZE * INPUT_SIZE);

  // Bilinear resize + ImageNet normalisation (RGB channels 0-2)
  for (let dy = 0; dy < INPUT_SIZE; dy++) {
    for (let dx = 0; dx < INPUT_SIZE; dx++) {
      // Map display pixel → source pixel
      const sx = (dx - padX) / scale;
      const sy = (dy - padY) / scale;

      let r = 0, g = 0, b = 0;
      if (sx >= 0 && sx < srcW && sy >= 0 && sy < srcH) {
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const x1 = Math.min(x0 + 1, srcW - 1);
        const y1 = Math.min(y0 + 1, srcH - 1);
        const wx = sx - x0, wy = sy - y0;

        const idx00 = (y0 * srcW + x0) * 4;
        const idx10 = (y0 * srcW + x1) * 4;
        const idx01 = (y1 * srcW + x0) * 4;
        const idx11 = (y1 * srcW + x1) * 4;

        r = (1-wx)*(1-wy)*rgba[idx00]   + wx*(1-wy)*rgba[idx10] +
            (1-wx)*wy    *rgba[idx01]   + wx*wy    *rgba[idx11];
        g = (1-wx)*(1-wy)*rgba[idx00+1] + wx*(1-wy)*rgba[idx10+1] +
            (1-wx)*wy    *rgba[idx01+1] + wx*wy    *rgba[idx11+1];
        b = (1-wx)*(1-wy)*rgba[idx00+2] + wx*(1-wy)*rgba[idx10+2] +
            (1-wx)*wy    *rgba[idx01+2] + wx*wy    *rgba[idx11+2];
      }

      const pixIdx = dy * INPUT_SIZE + dx;
      data[0 * INPUT_SIZE * INPUT_SIZE + pixIdx] = (r / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      data[1 * INPUT_SIZE * INPUT_SIZE + pixIdx] = (g / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      data[2 * INPUT_SIZE * INPUT_SIZE + pixIdx] = (b / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }

  return { data, scale, padX, padY };
}

// ── heatmap NMS ───────────────────────────────────────────────────────────────

/**
 * Max-pool NMS on a flat HW heatmap.  Returns a new array where non-local-max
 * values are zeroed out.
 */
function heatmapNMS(hm: Float32Array, H: number, W: number, k: number = NMS_KERNEL): Float32Array {
  const pad  = Math.floor(k / 2);
  const out  = new Float32Array(H * W);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = hm[y * W + x];
      let isMax = true;
      outer: for (let ky = -pad; ky <= pad; ky++) {
        for (let kx = -pad; kx <= pad; kx++) {
          const ny = y + ky, nx = x + kx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          if (hm[ny * W + nx] > v) { isMax = false; break outer; }
        }
      }
      out[y * W + x] = isMax ? v : 0;
    }
  }
  return out;
}

// ── ONNX session (lazy, cached) ───────────────────────────────────────────────

let _session: any = null;
let _ort: any = null;

async function getSession(modelPath: string): Promise<{ session: any; ort: any }> {
  if (_session) return { session: _session, ort: _ort };

  // Prefer onnxruntime-react-native in RN; fall back to onnxruntime-node in Node.js
  let ort: any;
  try {
    ort = require('onnxruntime-react-native');
  } catch {
    try {
      ort = require('onnxruntime-node');
    } catch {
      throw new Error(
        'ONNX runtime not available. Install onnxruntime-node (Node.js) or onnxruntime-react-native (React Native).',
      );
    }
  }

  _session = await ort.InferenceSession.create(modelPath);
  _ort = ort;
  return { session: _session, ort: _ort };
}

export function releaseSession(): void {
  if (_session) {
    try { _session.release?.(); } catch {}
    _session = null;
    _ort = null;
  }
}

// ── main inference function ───────────────────────────────────────────────────

/**
 * Detect arrow tips and scores in `rgba` using the exported ONNX model.
 *
 * @param rgba         Raw RGBA pixel buffer (Uint8Array, row-major)
 * @param width        Source image width  (pixels)
 * @param height       Source image height (pixels)
 * @param modelPath    Filesystem path to the ONNX model
 * @param threshold    Minimum heatmap confidence to accept a peak
 * @param maxDetections Maximum number of tips to return
 */
export async function detectArrowsNN(
  rgba: Uint8Array,
  width: number,
  height: number,
  modelPath: string,
  threshold: number = THRESHOLD,
  maxDetections: number = 20,
): Promise<ScoredArrow[]> {
  const { data, scale, padX, padY } = letterboxRgba(rgba, width, height);

  const { session, ort } = await getSession(modelPath);

  const inputTensor = new ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results = await session.run({ image: inputTensor });

  if (!results['tip_hm']) {
    throw new Error(
      'Model output "tip_hm" not found. Check the model was exported with output name "tip_hm". ' +
      `Available outputs: ${Object.keys(results).join(', ')}`,
    );
  }
  if (!results['score_map']) {
    throw new Error(
      'Model output "score_map" not found. Check the model was exported with output name "score_map". ' +
      `Available outputs: ${Object.keys(results).join(', ')}`,
    );
  }

  const tipHmRaw  = results['tip_hm'].data   as Float32Array;   // (1,1,HEATMAP_SIZE,HEATMAP_SIZE)
  const scoreRaw  = results['score_map'].data as Float32Array;   // (1,11,HEATMAP_SIZE,HEATMAP_SIZE)

  const tipHm = heatmapNMS(tipHmRaw, HEATMAP_SIZE, HEATMAP_SIZE);

  type Peak = { hx: number; hy: number; confidence: number };
  const peaks: Peak[] = [];
  for (let hy = 0; hy < HEATMAP_SIZE; hy++) {
    for (let hx = 0; hx < HEATMAP_SIZE; hx++) {
      const confidence = tipHm[hy * HEATMAP_SIZE + hx];
      if (confidence > threshold) peaks.push({ hx, hy, confidence });
    }
  }
  peaks.sort((a, b) => b.confidence - a.confidence);

  const arrows: ScoredArrow[] = [];
  for (const { hx, hy } of peaks.slice(0, maxDetections)) {
    const lbx   = (hx + 0.5) * RATIO;
    const lby   = (hy + 0.5) * RATIO;
    const origX = (lbx - padX) / scale;
    const origY = (lby - padY) / scale;

    // Read score from score_map at this heatmap location: argmax over 11 classes
    let score = 0, bestLogit = -Infinity;
    for (let c = 0; c < 11; c++) {
      const logit = scoreRaw[c * HEATMAP_SIZE * HEATMAP_SIZE + hy * HEATMAP_SIZE + hx];
      if (logit > bestLogit) { bestLogit = logit; score = c; }
    }

    arrows.push({ tip: [Math.round(origX), Math.round(origY)], score });
  }

  return arrows;
}
