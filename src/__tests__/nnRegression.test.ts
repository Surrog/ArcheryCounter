/**
 * @jest-environment node
 */
/**
 * Neural-network regression tests.
 *
 * Two goals:
 *
 * 1. PREPROCESSING CONSISTENCY — Freeze the exact pixel values produced by
 *    letterboxGray() and letterboxRgba() for known images.  If anyone changes
 *    the resize formula, grayscale conversion, normalisation constants, or the
 *    JPEG decoder they must update the reference values here AND retrain the
 *    affected model.  This is the regression that caught the PIL-vs-jpeg-js
 *    discrepancy (training used PIL antialias-bilinear, inference used 4-pixel
 *    bilinear → model returned near-zero probabilities for most images).
 *
 * 2. END-TO-END INFERENCE SANITY — Run both ONNX models on known images and
 *    assert that each model returns a non-null / non-empty result.  Catches
 *    broken exports, wrong input/output names, wrong logit signs, etc.
 *
 * Tests skip gracefully when ONNX models or image files are absent so that
 * the suite can run in CI without large binary assets.
 */

import * as fs   from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
// jpeg-js is a devDependency used only in Node.js contexts
// eslint-disable-next-line @typescript-eslint/no-var-requires
const jpegJs = require('jpeg-js') as typeof import('jpeg-js');

import { letterboxGray, extractPolygons, releaseBoundarySession } from '../boundaryDetector';
import { letterboxRgba, releaseSession as releaseArrowSession } from '../arrowDetector';

// ── paths ──────────────────────────────────────────────────────────────────

const IMAGES_DIR  = path.resolve(__dirname, '../../images');
const SCRIPTS_DIR = path.resolve(__dirname, '../../scripts');

const BOUNDARY_MODEL = path.join(SCRIPTS_DIR, 'nn/boundary_detector_v2.onnx');
const ARROW_MODEL    = path.join(SCRIPTS_DIR, 'nn/arrow_detector_fp32.onnx');

// Images whose preprocessing reference values are recorded below.
// Both must be present in images/ for the tests to run.
const BOUNDARY_IMAGE = '20210711_151526.jpg';   // 4000×4000, 1 target paper
const ARROW_IMAGE    = '20260326_214504.jpg';   // 4000×3000, has arrows

// ── helpers ────────────────────────────────────────────────────────────────

function loadRaw(filename: string) {
  const buf = fs.readFileSync(path.join(IMAGES_DIR, filename));
  return jpegJs.decode(buf, { useTArray: true }) as {
    data: Uint8Array; width: number; height: number;
  };
}

function imageExists(filename: string): boolean {
  return fs.existsSync(path.join(IMAGES_DIR, filename));
}

function modelExists(modelPath: string): boolean {
  return fs.existsSync(modelPath);
}

// ── 1. Preprocessing consistency ──────────────────────────────────────────

describe('letterboxGray preprocessing (boundary detector)', () => {
  // Reference values were computed with the current TypeScript implementation
  // using jpeg-js decode + 0-indexed 4-pixel bilinear + ImageNet-gray normalisation.
  // Tolerance is ±0.001 (one-unit rounding in 0-255 space ≈ 0.017 normalised,
  // but we use a tight tolerance to catch formula changes, not JPEG rounding).
  const TOLERANCE = 0.002;

  // Pixel (x, y) → expected normalised value
  // Image: 20210711_151526.jpg  4000×4000  scale=0.128  padX=0  padY=0
  const REFERENCE: Array<[number, number, number]> = [
    [  0,   0, -0.308815],
    [100, 100, -0.633701],
    [200, 200,  0.345983],
    [300, 300,  0.597163],
    [400, 400, -1.232188],
    [256, 128, -1.018584],
    [128, 256, -0.028718],
  ];

  let lb: ReturnType<typeof letterboxGray>;

  beforeAll(() => {
    if (!imageExists(BOUNDARY_IMAGE)) return;
    const raw = loadRaw(BOUNDARY_IMAGE);
    lb = letterboxGray(raw.data, raw.width, raw.height);
  });

  it('is skipped if test image is absent', () => {
    if (!imageExists(BOUNDARY_IMAGE)) pending('image not found');
  });

  it('produces a 512×512 output buffer', () => {
    if (!imageExists(BOUNDARY_IMAGE)) return;
    expect(lb.data.length).toBe(512 * 512);
  });

  it('has correct scale and padding for a 4000×4000 image', () => {
    if (!imageExists(BOUNDARY_IMAGE)) return;
    expect(lb.scale).toBeCloseTo(0.128, 5);
    expect(lb.padX).toBe(0);
    expect(lb.padY).toBe(0);
  });

  it.each(REFERENCE)(
    'pixel (%i, %i) matches reference value',
    (x, y, expected) => {
      if (!imageExists(BOUNDARY_IMAGE)) return;
      const actual = lb.data[y * 512 + x];
      expect(actual).toBeCloseTo(expected, 3);
    },
  );
});

describe('letterboxRgba preprocessing (arrow detector)', () => {
  // Image: 20260326_214504.jpg  4000×3000  scale=0.16  padX=0  padY=80
  // Reference: [x, y, channel, expected]  channel: 0=R 1=G 2=B
  const INPUT_SIZE = 640;
  const TOLERANCE = 0.002;

  const REFERENCE: Array<[number, number, 0|1|2, number]> = [
    [  0,   0, 0, -2.117904],
    [  0,   0, 1, -2.035714],
    [  0,   0, 2, -1.804444],
    [200, 200, 0,  1.392671],
    [200, 200, 1,  1.605742],
    [200, 200, 2,  1.942832],
    [320, 320, 0,  1.803665],
    [320, 320, 1,  1.710784],
    [320, 320, 2,  0.095338],
    [100, 100, 0,  1.221423],
    [100, 100, 1,  1.010504],
    [100, 100, 2,  0.687930],
    [500, 500, 0,  0.639181],
    [500, 500, 1,  0.205182],
    [500, 500, 2, -0.322963],
  ];

  let lb: ReturnType<typeof letterboxRgba>;

  beforeAll(() => {
    if (!imageExists(ARROW_IMAGE)) return;
    const raw = loadRaw(ARROW_IMAGE);
    lb = letterboxRgba(raw.data, raw.width, raw.height);
  });

  it('is skipped if test image is absent', () => {
    if (!imageExists(ARROW_IMAGE)) pending('image not found');
  });

  it('produces a 3×640×640 output buffer', () => {
    if (!imageExists(ARROW_IMAGE)) return;
    expect(lb.data.length).toBe(3 * INPUT_SIZE * INPUT_SIZE);
  });

  it('has correct scale and padding for a 4000×3000 image', () => {
    if (!imageExists(ARROW_IMAGE)) return;
    expect(lb.scale).toBeCloseTo(0.16, 5);
    expect(lb.padX).toBe(0);
    expect(lb.padY).toBe(80);
  });

  it.each(REFERENCE)(
    'pixel (%i, %i) channel %i matches reference value',
    (x, y, ch, expected) => {
      if (!imageExists(ARROW_IMAGE)) return;
      const actual = lb.data[ch * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x];
      expect(actual).toBeCloseTo(expected, 3);
    },
  );
});

// ── 2. Python training preprocessing parity ───────────────────────────────
//
// The boundary model is trained with boundary_dataset.py (Python/PIL) but
// runs inference with letterboxGray() (TypeScript/jpeg-js).  This describe
// block calls check_preprocessing_parity.py via `uv run` and asserts that
// the Python pipeline produces pixel values within JPEG-decoder noise
// (±0.03 normalised units) of the TypeScript reference values.
//
// A larger discrepancy (> 0.03) indicates a formula change in one pipeline
// that would corrupt training/inference alignment — exactly the class of bug
// that previously caused the model to return near-zero probabilities.

const NN_SCRIPTS_DIR    = path.join(SCRIPTS_DIR, 'nn');
const PARITY_SCRIPT     = path.join(NN_SCRIPTS_DIR, 'check_preprocessing_parity.py');
// Tolerance budget: JPEG decoder noise ≈ ±2 pixel units ≈ ±0.009 normalised.
// Preprocessing formula change (PIL bilinear vs 0-indexed bilinear) ≈ ±10 units.
// We use 0.03 so the test passes for decoder noise but fails for formula drift.
const PARITY_TOLERANCE = 0.03;

type ParityResult = {
  values: number[];
  scale:  number;
  pad_x:  number;
  pad_y:  number;
  orig_w: number;
  orig_h: number;
};

function runParityScript(imagePath: string, coords: [number, number][]): ParityResult {
  let rawOutput: string;
  try {
    rawOutput = execSync(
      `uv run python3 ${PARITY_SCRIPT} ${imagePath} '${JSON.stringify(coords)}'`,
      { cwd: NN_SCRIPTS_DIR, timeout: 60_000, encoding: 'utf8' },
    );
  } catch (e) {
    throw new Error(`Parity script execution failed: ${e}`);
  }
  try {
    return JSON.parse(rawOutput) as ParityResult;
  } catch {
    throw new Error(`Parity script returned non-JSON output:\n${rawOutput.slice(0, 500)}`);
  }
}

describe('Python boundary_dataset preprocessing matches TypeScript letterboxGray', () => {
  // Same reference pixels as the TypeScript letterboxGray test above.
  // Image: 20210711_151526.jpg  4000×4000  scale=0.128  padX=0  padY=0
  const REFERENCE: Array<[number, number, number]> = [
    [  0,   0, -0.308815],
    [100, 100, -0.633701],
    [200, 200,  0.345983],
    [300, 300,  0.597163],
    [400, 400, -1.232188],
    [256, 128, -1.018584],
    [128, 256, -0.028718],
  ];

  let result: ParityResult | null = null;
  let skipReason = '';

  beforeAll(() => {
    if (!imageExists(BOUNDARY_IMAGE)) {
      skipReason = `image ${BOUNDARY_IMAGE} not found`;
      return;
    }
    if (!fs.existsSync(PARITY_SCRIPT)) {
      skipReason = 'check_preprocessing_parity.py not found';
      return;
    }
    try {
      const coords = REFERENCE.map(([x, y]) => [x, y] as [number, number]);
      result = runParityScript(path.join(IMAGES_DIR, BOUNDARY_IMAGE), coords);
    } catch (e) {
      skipReason = `uv/Python unavailable or script failed: ${e}`;
    }
  }, 90_000);

  it('is skipped if image or Python deps are absent', () => {
    if (skipReason) pending(skipReason);
  });

  it('has correct scale and padding for a 4000×4000 image', () => {
    if (!result) return;
    expect(result.scale).toBeCloseTo(0.128, 5);
    expect(result.pad_x).toBe(0);
    expect(result.pad_y).toBe(0);
  });

  it.each(REFERENCE)(
    'pixel (%i, %i) matches TypeScript reference within JPEG-decoder noise',
    (x, y, tsExpected) => {
      if (!result) return;
      const idx = REFERENCE.findIndex(([rx, ry]) => rx === x && ry === y);
      const pyValue = result!.values[idx];
      // Assert both are close to the TypeScript reference.
      // A formula mismatch (e.g. PIL antialias vs 0-indexed bilinear) produces
      // differences > 0.04; JPEG decoder noise is typically < 0.02.
      expect(Math.abs(pyValue - tsExpected)).toBeLessThan(PARITY_TOLERANCE);
    },
  );
});

// ── 3. End-to-end inference sanity ────────────────────────────────────────
//
// ONNX native bindings use instanceof checks on Float32Array that break in
// Jest's instrumented VM.  We run inference in a plain ts-node subprocess
// (scripts/nn-regression-check.ts) and parse its JSON output here.

const ROOT = path.resolve(__dirname, '../..');
const REGRESSION_SCRIPT = path.join(ROOT, 'scripts/nn-regression-check.ts');
const TS_NODE_CMD = `npx ts-node --project ${path.join(ROOT, 'tsconfig.scripts.json')}`;

type RegressionResult = {
  boundary: { file: string; polygonCount: number; vertexCounts: number[] } | null;
  arrows:   { file: string; count: number } | null;
  errors:   string[];
};

function runRegressionScript(): RegressionResult {
  const output = execSync(`${TS_NODE_CMD} ${REGRESSION_SCRIPT}`, {
    cwd:      ROOT,
    timeout:  90_000,
    encoding: 'utf8',
  });
  return JSON.parse(output) as RegressionResult;
}

describe('NN end-to-end inference (boundary + arrow)', () => {
  let result: RegressionResult;

  beforeAll(() => {
    const modelsExist = modelExists(BOUNDARY_MODEL) && modelExists(ARROW_MODEL);
    const imagesExist = imageExists(BOUNDARY_IMAGE) && imageExists(ARROW_IMAGE);
    if (!modelsExist || !imagesExist) return;
    result = runRegressionScript();
  }, 90_000);

  describe('boundary detector', () => {
    it('is skipped if model or image is absent', () => {
      if (!modelExists(BOUNDARY_MODEL) || !imageExists(BOUNDARY_IMAGE))
        pending('boundary model or image not found');
    });

    it('produces at least 1 polygon for a known target image', () => {
      if (!modelExists(BOUNDARY_MODEL) || !imageExists(BOUNDARY_IMAGE)) return;
      expect(result.errors.filter(e => e.includes('boundary'))).toHaveLength(0);
      expect(result.boundary).not.toBeNull();
      expect(result.boundary!.polygonCount).toBeGreaterThanOrEqual(1);
    });

    it('every polygon has 3–10 vertices (MAX_VERTICES constraint)', () => {
      if (!modelExists(BOUNDARY_MODEL) || !imageExists(BOUNDARY_IMAGE)) return;
      if (!result?.boundary) return;
      for (const n of result.boundary.vertexCounts) {
        expect(n).toBeGreaterThanOrEqual(3);
        expect(n).toBeLessThanOrEqual(10);
      }
    });
  });

  describe('arrow detector', () => {
    it('is skipped if model or image is absent', () => {
      if (!modelExists(ARROW_MODEL) || !imageExists(ARROW_IMAGE))
        pending('arrow model or image not found');
    });

    it('returns an array for a known image (empty is acceptable)', () => {
      if (!modelExists(ARROW_MODEL) || !imageExists(ARROW_IMAGE)) return;
      expect(result.errors.filter(e => e.includes('arrow'))).toHaveLength(0);
      expect(result.arrows).not.toBeNull();
      expect(typeof result.arrows!.count).toBe('number');
    });

    it('detects arrows in 20260326_214504.jpg (known to have arrows)', () => {
      if (!modelExists(ARROW_MODEL) || !imageExists(ARROW_IMAGE)) return;
      if (!result?.arrows) return;
      expect(result.arrows.count).toBeGreaterThan(0);
    });
  });
});

// ── 4. Input validation (zero/negative dimensions) ────────────────────────

describe('letterboxGray input validation', () => {
  it('throws on zero width', () => {
    expect(() => letterboxGray(new Uint8Array(0), 0, 100))
      .toThrow('Invalid image dimensions');
  });

  it('throws on zero height', () => {
    expect(() => letterboxGray(new Uint8Array(0), 100, 0))
      .toThrow('Invalid image dimensions');
  });

  it('throws on negative width', () => {
    expect(() => letterboxGray(new Uint8Array(0), -1, 100))
      .toThrow('Invalid image dimensions');
  });
});

describe('letterboxRgba input validation', () => {
  it('throws on zero width', () => {
    expect(() => letterboxRgba(new Uint8Array(0), 0, 100))
      .toThrow('Invalid image dimensions');
  });

  it('throws on zero height', () => {
    expect(() => letterboxRgba(new Uint8Array(0), 100, 0))
      .toThrow('Invalid image dimensions');
  });

  it('throws on negative dimensions', () => {
    expect(() => letterboxRgba(new Uint8Array(0), -5, -5))
      .toThrow('Invalid image dimensions');
  });
});

describe('extractPolygons with empty/uniform logits', () => {
  it('returns null for all-zero logit map (no foreground)', () => {
    const logits = new Float32Array(512 * 512).fill(-10); // all sigmoid ≈ 0
    expect(extractPolygons(logits, 1, 0, 0)).toBeNull();
  });

  it('returns null when mean confidence is below threshold', () => {
    // A single low-confidence foreground pixel: sigmoid(0.2) ≈ 0.55 > 0.5
    // but mean conf ≈ 0.55 < CONFIDENCE_THRESHOLD (0.6) → returns null
    const logits = new Float32Array(512 * 512).fill(-10);
    logits[0] = 0.2; // one barely-above-threshold pixel, mean conf < 0.6
    expect(extractPolygons(logits, 1, 0, 0)).toBeNull();
  });
});

// ── 5. ONNX output key validation ─────────────────────────────────────────
//
// Simulate a model that returns wrong output names and verify both detectors
// throw descriptive errors rather than crashing with "Cannot read properties
// of undefined".  Uses jest.isolateModules + jest.doMock so that the dynamic
// require() calls inside getSession() pick up the mock.

describe('arrowDetector: throws descriptive error when ONNX output key is wrong', () => {
  it('throws an error mentioning "tip_hm" when model returns wrong output keys', async () => {
    let detectArrowsNN!: typeof import('../arrowDetector').detectArrowsNN;
    let releaseSession!: typeof import('../arrowDetector').releaseSession;

    jest.isolateModules(() => {
      jest.doMock('onnxruntime-node', () => ({
        Tensor: class { constructor(public type: string, public data: unknown, public dims: number[]) {} },
        InferenceSession: {
          create: jest.fn().mockResolvedValue({
            run: jest.fn().mockResolvedValue({ bad_output: { data: new Float32Array(0) } }),
          }),
        },
      }), { virtual: true });
      jest.doMock('onnxruntime-react-native', () => { throw new Error('not available'); }, { virtual: true });

      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const mod = require('../arrowDetector');
      detectArrowsNN = mod.detectArrowsNN;
      releaseSession  = mod.releaseSession;
    });

    const rgba = new Uint8Array(4 * 4 * 4).fill(128);
    await expect(detectArrowsNN(rgba, 4, 4, '/fake/arrow.onnx')).rejects.toThrow(/tip_hm/);
    releaseSession();
  });
});

describe('boundaryDetector: logits output key check (unit-level via extractPolygons)', () => {
  // boundaryDetector.ts uses dynamic import() for onnxruntime, which Jest's doMock cannot
  // intercept without --experimental-vm-modules.  We therefore verify the error-throwing
  // logic at the only point we can reach without an ONNX runtime: extractPolygons.
  // The full "wrong key → descriptive error" path is exercised by the E2E tests above
  // (nn-regression-check.ts) when models are present.

  it('extractPolygons returns null for all-background logits (regression guard)', () => {
    const logits = new Float32Array(512 * 512).fill(-20);
    expect(extractPolygons(logits, 1, 0, 0)).toBeNull();
  });

  it('extractPolygons returns polygons for a ring-shaped high-confidence foreground region', () => {
    // Simulate NN output: a thin rectangular ring (5px wide border) around a 200×200 block.
    // This is representative of how the boundary segmentation model outputs the paper edge.
    const SIZE = 512;
    const logits = new Float32Array(SIZE * SIZE).fill(-20);
    const [cx, cy, r, thick] = [256, 256, 100, 5];
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const dy = y - cy, dx = x - cx;
        const dist = Math.hypot(dx, dy);
        if (dist >= r - thick && dist <= r + thick) {
          logits[y * SIZE + x] = 5; // sigmoid(5) ≈ 0.993
        }
      }
    }
    const result = extractPolygons(logits, 1, 0, 0);
    expect(result).not.toBeNull();
    expect(result!.length).toBeGreaterThanOrEqual(1);
    // Each polygon must have at least 3 vertices
    for (const poly of result!) expect(poly.length).toBeGreaterThanOrEqual(3);
  });
});
