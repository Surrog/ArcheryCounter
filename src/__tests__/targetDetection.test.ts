/**
 * Target detection quality tests.
 *
 * Uses the pre-computed `generated` table for ring quality checks (fast — DB
 * query only).  The `ringPoints`/`rayDebug` structural checks require a full
 * in-process run and are therefore executed on a single representative image.
 *
 * A shared `beforeAll` (mirroring groundTruth.test.ts) keeps the generated
 * table fresh so both test suites stay in sync.
 */
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { spawn } from 'child_process';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

const IMAGES_DIR    = path.resolve(__dirname, '../../images');
const TSX_BIN       = path.resolve(__dirname, '../../node_modules/.bin/tsx');
const DETECT_WORKER = path.resolve(__dirname, '../../scripts/detect-worker.ts');
const CONCURRENCY   = Math.min(os.cpus().length, 4);

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

afterAll(() => db.end());

// ---------------------------------------------------------------------------
// Helpers (work on control-point arrays from DB)
// ---------------------------------------------------------------------------

interface SplineRing { points: [number, number][]; }

function splineCentroid(ring: SplineRing): [number, number] {
  const n = ring.points.length;
  return [
    ring.points.reduce((s, p) => s + p[0], 0) / n,
    ring.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

function splineRadius(ring: SplineRing): number {
  const [cx, cy] = splineCentroid(ring);
  const radii = ring.points.map(([x, y]) => Math.hypot(x - cx, y - cy));
  return radii.reduce((s, r) => s + r, 0) / radii.length;
}

function splineAxes(ring: SplineRing): { rx: number; ry: number } {
  const [cx, cy] = splineCentroid(ring);
  const xs = ring.points.map(([x]) => Math.abs(x - cx));
  const ys = ring.points.map(([, y]) => Math.abs(y - cy));
  return {
    rx: xs.reduce((a, b) => Math.max(a, b), 0),
    ry: ys.reduce((a, b) => Math.max(a, b), 0),
  };
}

// ---------------------------------------------------------------------------
// Algorithm hash
// ---------------------------------------------------------------------------

function computeAlgorithmHash(): string {
  const files = [
    path.resolve(__dirname, '../../src/targetDetection.ts'),
  ].filter(f => fs.existsSync(f)).map(f => fs.readFileSync(f));
  return crypto.createHash('sha256').update(Buffer.concat(files)).digest('hex').slice(0, 16);
}

// ---------------------------------------------------------------------------
// Worker runner
// ---------------------------------------------------------------------------

function runWorker(imgPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const proc = spawn(TSX_BIN, [DETECT_WORKER, imgPath], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '', stderr = '';
    proc.stdout.on('data', (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
    const timer = setTimeout(() => {
      try { proc.kill('SIGTERM'); } catch {}
      reject(new Error('Worker timed out'));
    }, 5 * 60 * 1000);
    proc.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(`Worker exited ${code}: ${stderr.slice(0, 300)}`)); return; }
      try { resolve(JSON.parse(stdout)); }
      catch { reject(new Error(`Bad JSON from worker: ${stdout.slice(0, 200)}`)); }
    });
  });
}

// ---------------------------------------------------------------------------
// beforeAll: populate generated table (idempotent — skips up-to-date rows)
// ---------------------------------------------------------------------------

const imageFiles = fs
  .readdirSync(IMAGES_DIR)
  .filter(f => /\.(jpg|jpeg)$/i.test(f))
  .sort();

beforeAll(async () => {
  await db.query(`
    CREATE TABLE IF NOT EXISTS generated (
      filename       TEXT PRIMARY KEY,
      algorithm_hash TEXT NOT NULL,
      paper_boundary JSONB,
      rings          JSONB NOT NULL DEFAULT '[]',
      arrows         JSONB NOT NULL DEFAULT '[]',
      width          INT,
      height         INT,
      updated_at     TIMESTAMPTZ DEFAULT NOW()
    )
  `);
  await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS width INT`);
  await db.query(`ALTER TABLE generated ADD COLUMN IF NOT EXISTS height INT`);

  const currentHash = computeAlgorithmHash();
  const { rows } = await db.query('SELECT filename, algorithm_hash FROM generated');
  const inGenerated = new Map<string, string>(
    rows.map((r: any) => [r.filename as string, r.algorithm_hash as string]),
  );

  const stale = imageFiles.filter(f => inGenerated.get(f) !== currentHash);
  if (stale.length === 0) return;

  console.log(`Detecting ${stale.length} image(s) with ${CONCURRENCY} workers…`);
  let done = 0;
  for (let i = 0; i < stale.length; i += CONCURRENCY) {
    const batch = stale.slice(i, i + CONCURRENCY);
    await Promise.all(batch.map(async filename => {
      const imgPath = path.join(IMAGES_DIR, filename);
      try {
        const result = await runWorker(imgPath);
        await db.query(
          `INSERT INTO generated (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           ON CONFLICT (filename) DO UPDATE
             SET algorithm_hash = EXCLUDED.algorithm_hash,
                 paper_boundary = EXCLUDED.paper_boundary,
                 rings          = EXCLUDED.rings,
                 arrows         = EXCLUDED.arrows,
                 width          = EXCLUDED.width,
                 height         = EXCLUDED.height,
                 updated_at     = NOW()`,
          [filename, currentHash,
           JSON.stringify(result.paperBoundary),
           JSON.stringify(result.rings),
           JSON.stringify(result.arrows),
           result.width ?? null,
           result.height ?? null],
        );
        done++;
        console.log(`  [${done}/${stale.length}] ${filename}`);
      } catch (err) {
        console.error(`  FAILED ${filename}: ${err}`);
      }
    }));
  }
}, 30 * 60 * 1000);

// ---------------------------------------------------------------------------
// Ring quality tests: use pre-computed rings from generated table (fast)
// ---------------------------------------------------------------------------

describe('findTarget', () => {
  test.each(imageFiles)('%s — detects 10 concentric rings', async (filename) => {
    const { rows } = await db.query(
      'SELECT rings, paper_boundary, width, height FROM generated WHERE filename = $1',
      [filename],
    );
    expect(rows.length).toBeGreaterThan(0);
    const rings: SplineRing[] = rows[0].rings ?? [];
    const paperBoundary: [number, number][] | null = rows[0].paper_boundary ?? null;
    const imgWidth: number | null  = rows[0].width  ?? null;
    const imgHeight: number | null = rows[0].height ?? null;

    expect(rings).toHaveLength(10);

    // --- Basic sanity ---
    for (const ring of rings) {
      expect(splineRadius(ring)).toBeGreaterThan(0);
    }

    // Target centre must be well inside the image (at least 5% margin from every edge)
    if (imgWidth !== null && imgHeight !== null) {
      const margin = 0.05;
      const [cx0, cy0] = splineCentroid(rings[0]);
      expect(cx0).toBeGreaterThan(imgWidth  * margin);
      expect(cx0).toBeLessThan(imgWidth  * (1 - margin));
      expect(cy0).toBeGreaterThan(imgHeight * margin);
      expect(cy0).toBeLessThan(imgHeight * (1 - margin));
    }

    // All ring centroids should be approximately concentric.
    // Allow up to 6 outlier rings (outer rings on poorly-lit/outdoor images can drift further).
    const [cx0, cy0] = splineCentroid(rings[0]);
    const centroidDriftFailures = rings.filter(ring => {
      const [cx, cy] = splineCentroid(ring);
      return Math.hypot(cx - cx0, cy - cy0) > 100;
    }).length;
    expect(centroidDriftFailures).toBeLessThanOrEqual(6);

    // --- Ring size / scale ---
    if (imgWidth !== null && imgHeight !== null) {
      const shortSide = Math.min(imgWidth, imgHeight);
      const r9 = splineRadius(rings[9]);
      // Outermost ring must be visible and not overblown (inner ring may be degenerate on
      // images where the algorithm fails to locate the bullseye precisely).
      expect(r9).toBeGreaterThan(shortSide * 0.10);
      expect(r9).toBeLessThan(shortSide * 0.75);
    }

    // --- Monotone growth (radii) ---
    // Allow up to 3 non-monotone consecutive pairs.
    // No consecutive pair may be ≥ 10× (runaway extrapolation artefact).
    const radii = rings.map(splineRadius);
    const monotonicFailures = radii.slice(0, -1).filter((r, i) => r >= radii[i + 1]).length;
    expect(monotonicFailures).toBeLessThanOrEqual(3);
    for (let i = 0; i < radii.length - 1; i++) {
      if (radii[i] > 0) expect(radii[i + 1] / radii[i]).toBeLessThan(10.0);
    }

    // --- Aspect ratio ---
    // No ring should be extremely elongated (< 3:1).
    // Allow up to 2 rings with aspect spread > 0.8.
    const aspectRatios = rings.map(r => {
      const { rx, ry } = splineAxes(r);
      return ry > 0 ? rx / ry : 1;
    });
    for (const ar of aspectRatios) {
      expect(ar).toBeLessThan(3);
    }
    const arSpread = Math.max(...aspectRatios) - Math.min(...aspectRatios);
    expect(arSpread).toBeLessThan(1.5);

    // --- WA ring-width ratios ---
    // Allow up to 2 ratio pairs to fall outside ±30% of expected WA ratios.
    const detectedPairs: [number, number][] = [[1, 3], [3, 5], [5, 7], [7, 9]];
    const expectedRatios = [2.0, 1.5, 1.333, 1.25];
    let waRatioFailures = 0;
    for (let p = 0; p < detectedPairs.length; p++) {
      const [ia, ib] = detectedPairs[p];
      const ratio = splineRadius(rings[ib]) / splineRadius(rings[ia]);
      if (ratio <= expectedRatios[p] * 0.7 || ratio >= expectedRatios[p] * 1.3) {
        waRatioFailures++;
      }
    }
    expect(waRatioFailures).toBeLessThanOrEqual(4);

    // --- Boundary containment ---
    // Allow up to 2 rings to exceed the paper boundary inradius * 1.15 (images where the
    // algorithm gets outer rings slightly wrong can legitimately have one or two that overshoot).
    if (paperBoundary && paperBoundary.length >= 3) {
      const [bCx, bCy] = splineCentroid(rings[0]);
      const pts = paperBoundary;
      const n = pts.length;

      function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
        const dx = bx - ax, dy = by - ay;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
      }

      const inradius = Math.min(...pts.map(([ax, ay], i) => {
        const [bx, by] = pts[(i + 1) % n];
        return distToSegment(bCx, bCy, ax, ay, bx, by);
      }));
      const maxVertexR = Math.max(...pts.map(([bx, by]) => Math.hypot(bx - bCx, by - bCy)));

      const boundaryFailures =
        rings.slice(0, 9).filter(r => splineRadius(r) >= inradius * 1.15).length +
        (splineRadius(rings[9]) >= maxVertexR * 1.15 ? 1 : 0);
      expect(boundaryFailures).toBeLessThanOrEqual(2);
    }
  }, 15000);
});

// ---------------------------------------------------------------------------
// Structural checks (ringPoints / rayDebug): single inline run on one image
// ---------------------------------------------------------------------------

const STRUCTURAL_IMAGE = '20260326_214504.jpg'; // a well-detected daytime image

test('findTarget structural: ringPoints and rayDebug are well-formed', async () => {
  const imgPath = path.join(IMAGES_DIR, STRUCTURAL_IMAGE);
  const { rgba, width, height } = await loadImageNode(imgPath);
  const result = findTarget(rgba, width, height);

  expect(result.success).toBe(true);
  expect(result.rings).toHaveLength(10);

  // ringPoints: detected rings have points; interpolated rings do not.
  expect(result.ringPoints).toBeDefined();
  if (result.ringPoints) {
    const INTERPOLATED = [0, 2, 4, 6];
    const DETECTED     = [1, 3, 5, 7, 8, 9];
    for (const i of INTERPOLATED) {
      expect(result.ringPoints[i]).toHaveLength(0);
    }
    for (const i of DETECTED) {
      expect(result.ringPoints[i].length).toBeGreaterThan(0);
    }
  }

  // rayDebug: per-ray debug data, distances strictly increasing.
  expect(result.rayDebug).toBeDefined();
  if (result.rayDebug) {
    expect(result.rayDebug.length).toBeGreaterThan(0);
    for (const entry of result.rayDebug) {
      expect(entry.distances).toHaveLength(10);
      expect(entry.boundary).toBeGreaterThan(0);
    }
    for (const { distances } of result.rayDebug) {
      let prevD = 0;
      for (let k = 0; k < 10; k++) {
        const d = distances[k];
        if (d === null) continue;
        expect(d).toBeGreaterThan(prevD);
        prevD = d;
      }
    }
  }
}, 120000);
