/**
 * NN inference regression check.
 *
 * Runs detectBoundaries and detectArrowsNN on known test images and prints
 * a JSON result to stdout.  Called by nnRegression.test.ts via execSync so
 * that the ONNX native bindings run in a plain Node.js process (not Jest's
 * instrumented VM which breaks instanceof checks on Float32Array).
 *
 * Exit 0 on success, 1 on failure.
 *
 * Usage:
 *   npx ts-node --project tsconfig.scripts.json scripts/nn-regression-check.ts
 */

import * as fs   from 'fs';
import * as path from 'path';
import * as jpegJs from 'jpeg-js';

import { detectBoundaries, releaseBoundarySession } from '../src/boundaryDetector';
import { detectArrowsNN, releaseSession }            from '../src/arrowDetector';

const IMAGES_DIR     = path.resolve(__dirname, '../images');
const SCRIPTS_DIR    = path.resolve(__dirname);

const BOUNDARY_MODEL = path.join(SCRIPTS_DIR, 'nn/boundary_detector_v2.onnx');
const ARROW_MODEL    = path.join(SCRIPTS_DIR, 'nn/arrow_detector_fp32.onnx');

// Two images that must produce non-trivial output
const BOUNDARY_IMAGE = '20210711_151526.jpg';   // 1 paper target, no EXIF rotation
const ARROW_IMAGE    = '20260326_214504.jpg';   // multiple arrows

type Result = {
  boundary: { file: string; polygonCount: number; vertexCounts: number[] } | null;
  arrows:   { file: string; count: number }                                | null;
  errors:   string[];
};

async function run(): Promise<Result> {
  const errors: string[] = [];
  let boundary: Result['boundary'] = null;
  let arrows:   Result['arrows']   = null;

  // ── boundary ──────────────────────────────────────────────────────────────
  const bPath = path.join(IMAGES_DIR, BOUNDARY_IMAGE);
  if (!fs.existsSync(bPath)) {
    errors.push(`boundary image not found: ${BOUNDARY_IMAGE}`);
  } else if (!fs.existsSync(BOUNDARY_MODEL)) {
    errors.push(`boundary model not found: ${BOUNDARY_MODEL}`);
  } else {
    try {
      const raw = jpegJs.decode(fs.readFileSync(bPath), { useTArray: true });
      const polys = await detectBoundaries(raw.data, raw.width, raw.height, BOUNDARY_MODEL);
      boundary = polys
        ? { file: BOUNDARY_IMAGE, polygonCount: polys.length, vertexCounts: polys.map(p => p.length) }
        : { file: BOUNDARY_IMAGE, polygonCount: 0, vertexCounts: [] };
    } catch (e) {
      errors.push(`boundary inference error: ${e}`);
    }
  }

  // ── arrows ─────────────────────────────────────────────────────────────────
  const aPath = path.join(IMAGES_DIR, ARROW_IMAGE);
  if (!fs.existsSync(aPath)) {
    errors.push(`arrow image not found: ${ARROW_IMAGE}`);
  } else if (!fs.existsSync(ARROW_MODEL)) {
    errors.push(`arrow model not found: ${ARROW_MODEL}`);
  } else {
    try {
      const raw = jpegJs.decode(fs.readFileSync(aPath), { useTArray: true });
      const detected = await detectArrowsNN(raw.data, raw.width, raw.height, ARROW_MODEL);
      arrows = { file: ARROW_IMAGE, count: detected.length };
    } catch (e) {
      errors.push(`arrow inference error: ${e}`);
    }
  }

  releaseBoundarySession();
  releaseSession();

  return { boundary, arrows, errors };
}

run().then(result => {
  console.log(JSON.stringify(result, null, 2));
  process.exit(result.errors.length > 0 ? 1 : 0);
}).catch(e => {
  console.error(e);
  process.exit(1);
});
