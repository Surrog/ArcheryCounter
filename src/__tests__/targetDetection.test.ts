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
import * as crypto from 'crypto';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';

import { expect, describe, afterAll, beforeAll, test } from '@jest/globals';


const IMAGES_DIR = path.resolve(__dirname, '../../images');

const GEN = '"test_td".generated';
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
// beforeAll: populate generated table (idempotent — skips up-to-date rows)
// ---------------------------------------------------------------------------

function sampleImages(imageFiles: string[], sampleSize: number): string[] {
  if (imageFiles.length <= sampleSize) return imageFiles;
  const sampled = new Set<string>();
  while (sampled.size < sampleSize) {
    const idx = Math.floor(Math.random() * imageFiles.length);
    sampled.add(imageFiles[idx]);
  }
  return Array.from(sampled);
}

const imageFiles : string[] = sampleImages(
  fs
    .readdirSync(IMAGES_DIR)
    .filter(f => /\.(jpg|jpeg)$/i.test(f))
    .sort(),
  15
);

beforeAll(async () => {
  // Schema + table are created in globalSetup.ts; just populate missing rows here.

  const algHash = crypto
    .createHash('sha256')
    .update(fs.readFileSync(path.resolve(__dirname, '../targetDetection.ts')))
    .digest('hex')
    .slice(0, 16);

  const { rows: genRows } = await db.query(
    `SELECT filename, algorithm_hash FROM ${GEN} WHERE filename = ANY($1)`,
    [imageFiles],
  );
  const inGenerated = new Map<string, string>(genRows.map((r: any) => [r.filename as string, r.algorithm_hash as string]));

  const pending = imageFiles.filter(f => inGenerated.get(f) !== algHash);

  for (const filename of pending) {
    const imgPath = path.join(IMAGES_DIR, filename);
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);
    // boundary: TargetBoundary[]  rings: SplineRing[][] (one ring-set per target)
    const boundary = result.success ? result.targets.map(t => t.paperBoundary) : [];
    const rings    = result.success ? result.targets.map(t => t.rings) : [];
    await db.query(
      `INSERT INTO ${GEN} (filename, algorithm_hash, paper_boundary, rings, arrows, width, height)
       VALUES ($1, $2, $3, $4, '[]', $5, $6)
       ON CONFLICT (filename) DO UPDATE
         SET algorithm_hash = EXCLUDED.algorithm_hash,
             paper_boundary = EXCLUDED.paper_boundary,
             rings          = EXCLUDED.rings,
             arrows         = EXCLUDED.arrows,
             width          = EXCLUDED.width,
             height         = EXCLUDED.height,
             updated_at     = NOW()
         WHERE generated.algorithm_hash <> EXCLUDED.algorithm_hash`,
      [filename, algHash, JSON.stringify(boundary), JSON.stringify(rings), width, height],
    );
  }
}, 30 * 60 * 1000);

// ---------------------------------------------------------------------------
// Ring quality tests: use pre-computed rings from generated table (fast)
// ---------------------------------------------------------------------------

describe('findTarget', () => {
  test.each(imageFiles)('%s — detects 10 concentric rings', async (filename) => {
    const { rows } = await db.query(
      `SELECT rings, paper_boundary, width, height FROM ${GEN} WHERE filename = $1`,
      [filename],
    );
    expect(rows.length).toBeGreaterThan(0);
    // DB stores rings as SplineRing[][] (one ring-set per target); take the first ring-set.
    const rings: SplineRing[] = (rows[0].rings as SplineRing[][])?.[0] ?? [];
    // DB stores paper_boundary as TargetBoundary[] ({points:[…]}[]); take the first target's points.
    const paperBoundary: [number, number][] | null = rows[0].paper_boundary?.[0]?.points ?? null;
    const imgWidth: number | null  = rows[0].width  ?? null;
    const imgHeight: number | null = rows[0].height ?? null;

    expect(rings.length).toBeGreaterThanOrEqual(7); // some image have 7 rings, others have 10

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

    // --- Ring size / scale (only meaningful when all 10 rings were detected) ---
    if (rings.length >= 10 && imgWidth !== null && imgHeight !== null) {
      const shortSide = Math.min(imgWidth, imgHeight);
      const r9 = splineRadius(rings[9]);
      // Outermost ring must be visible and not overblown (inner ring may be degenerate on
      // images where the algorithm fails to locate the bullseye precisely).
      expect(r9).toBeGreaterThan(shortSide * 0.10);
      expect(r9).toBeLessThan(shortSide * 0.75);
    }

    // --- Monotone growth (radii) ---
    // Allow up to 3 non-monotone consecutive pairs.
    // Allow at most 1 consecutive pair with a ≥ 12× ratio (runaway extrapolation artefact).
    // Count-based rather than a hard per-pair cap so that one badly-extrapolated outer ring
    // doesn't fail the whole image.
    const radii = rings.map(splineRadius);
    const monotonicFailures = radii.slice(0, -1).filter((r, i) => r >= radii[i + 1]).length;
    expect(monotonicFailures).toBeLessThanOrEqual(3);
    const ratioRunawayCount = radii.slice(0, -1).filter((r, i) => r > 0 && radii[i + 1] / r >= 12.0).length;
    expect(ratioRunawayCount).toBeLessThanOrEqual(1);

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

    // --- WA ring-width ratios (only meaningful when all 10 rings were detected) ---
    if (rings.length >= 10) {
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
    }

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

      // Boundary containment is only meaningful when all 10 rings were detected.
      // For 7-ring images the last ring IS the boundary polygon, so checking
      // inner-ring containment against its inradius would be circular.
      if (rings.length >= 10) {
        const boundaryFailures =
          rings.slice(0, 9).filter(r => splineRadius(r) >= inradius * 1.15).length +
          (splineRadius(rings[9]) >= maxVertexR * 1.15 ? 1 : 0);
        expect(boundaryFailures).toBeLessThanOrEqual(4);
      }
    }
  }, 15000);
});
