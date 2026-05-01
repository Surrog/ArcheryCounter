/**
 * Benchmark: compare human annotations vs algorithm detections stored in DB.
 * Reads from `annotations` (human) and `generated` (algorithm) tables.
 * Usage: tsx scripts/benchmark.ts
 */

import { Pool } from 'pg';
import { sampleClosedSpline } from '../src/spline';
import type { SplineRing } from '../src/spline';

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

function ringCentroid(r: SplineRing): [number, number] {
  const n = r.points.length;
  return [
    r.points.reduce((s, p) => s + p[0], 0) / n,
    r.points.reduce((s, p) => s + p[1], 0) / n,
  ];
}

function ringMeanRadius(r: SplineRing): number {
  const [cx, cy] = ringCentroid(r);
  return r.points.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / r.points.length;
}

/** One-sided mean distance: for each point in A, find nearest point in B. */
function meanDistAtoB(a: [number,number][], b: [number,number][]): number {
  let sum = 0;
  for (const pa of a) {
    let minD = Infinity;
    for (const pb of b) {
      const d = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]);
      if (d < minD) minD = d;
    }
    sum += minD;
  }
  return sum / a.length;
}

/** Symmetric mean boundary distance between two spline rings. */
function splineMeanDist(a: SplineRing, b: SplineRing): number {
  const N = 120;
  const aPts = sampleClosedSpline(a.points, N);
  const bPts = sampleClosedSpline(b.points, N);
  return (meanDistAtoB(aPts, bPts) + meanDistAtoB(bPts, aPts)) / 2;
}

// ---------------------------------------------------------------------------
// Arrow matching
// ---------------------------------------------------------------------------

const ARROW_MATCH_PX = 40;

function matchArrows(
  annArrows: { tip: [number,number] }[],
  detArrows: { tip: [number,number] }[],
): { tp: number; fn: number; fp: number; matchedErrors: number[] } {
  const matched = new Set<number>();
  const matchedErrors: number[] = [];
  let tp = 0, fn = 0;

  for (const ann of annArrows) {
    let bestD = Infinity, bestJ = -1;
    detArrows.forEach((det, j) => {
      if (matched.has(j)) return;
      const d = Math.hypot(ann.tip[0] - det.tip[0], ann.tip[1] - det.tip[1]);
      if (d < bestD) { bestD = d; bestJ = j; }
    });
    if (bestJ >= 0 && bestD <= ARROW_MATCH_PX) {
      tp++;
      matched.add(bestJ);
      matchedErrors.push(bestD);
    } else {
      fn++;
    }
  }

  const fp = detArrows.length - matched.size;
  return { tp, fn, fp, matchedErrors };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const { rows } = await db.query(`
    SELECT
      a.filename,
      a.rings          AS ann_rings,
      a.arrows         AS ann_arrows,
      a.paper_boundary AS ann_boundary,
      g.rings          AS det_rings,
      g.arrows         AS det_arrows,
      g.paper_boundary AS det_boundary,
      g.algorithm_hash
    FROM annotations a
    LEFT JOIN generated g ON g.filename = a.filename
    WHERE a.rings IS NOT NULL AND jsonb_array_length(a.rings) > 0
    ORDER BY a.filename
  `);

  console.log(`Loaded ${rows.length} annotated images\n`);

  const results: any[] = [];

  for (const row of rows) {
    const annRings: SplineRing[]  = row.ann_rings  ?? [];
    const detRings: SplineRing[]  = row.det_rings  ?? [];
    const annArrows: any[]        = row.ann_arrows ?? [];
    const detArrows: any[]        = row.det_arrows ?? [];
    const hasDet = row.algorithm_hash != null;

    const entry: any = {
      filename: row.filename,
      has_detection: hasDet,
      detection_success: hasDet && detRings.length > 0,
    };

    if (!hasDet) {
      entry.note = 'no generated row — never processed';
      results.push(entry);
      continue;
    }

    if (detRings.length === 0) {
      entry.note = 'detection failed (no rings)';
      results.push(entry);
      continue;
    }

    // Ring accuracy — only for rings present in both
    const ringErrors: (number | null)[] = [];
    const ringRadiusRelErrors: (number | null)[] = [];

    for (let i = 0; i < 10; i++) {
      const ann = annRings[i];
      const det = detRings[i];
      if (!ann || !det || !ann.points?.length || !det.points?.length) {
        ringErrors.push(null);
        ringRadiusRelErrors.push(null);
        continue;
      }
      const dist = splineMeanDist(ann, det);
      const annR  = ringMeanRadius(ann);
      const detR  = ringMeanRadius(det);
      const relR  = annR > 0 ? Math.abs(detR - annR) / annR : null;
      ringErrors.push(Math.round(dist * 10) / 10);
      ringRadiusRelErrors.push(relR != null ? Math.round(relR * 1000) / 1000 : null);
    }

    entry.ring_mean_dist_px  = ringErrors;
    entry.ring_radius_rel_err = ringRadiusRelErrors;

    // Boundary
    const annBoundary: [number,number][] | null = row.ann_boundary ?? null;
    const detBoundary: [number,number][] | null = row.det_boundary ?? null;
    entry.has_ann_boundary = annBoundary != null;
    entry.has_det_boundary = detBoundary != null;

    // Arrow accuracy
    const arrowMetrics = matchArrows(annArrows, detArrows);
    entry.arrows = {
      ann_count: annArrows.length,
      det_count: detArrows.length,
      tp: arrowMetrics.tp,
      fn: arrowMetrics.fn,
      fp: arrowMetrics.fp,
      matched_tip_errors_px: arrowMetrics.matchedErrors.map(e => Math.round(e)),
    };

    results.push(entry);
  }

  // ---------------------------------------------------------------------------
  // Summary
  // ---------------------------------------------------------------------------

  const processed = results.filter(r => r.detection_success);
  const failed    = results.filter(r => !r.detection_success);

  console.log(`=== Detection success: ${processed.length}/${results.length} ===\n`);

  if (failed.length) {
    console.log('FAILED images:');
    for (const r of failed) console.log(`  ${r.filename}: ${r.note ?? 'detection failed'}`);
    console.log('');
  }

  // Per-ring aggregate errors across all successful images
  console.log('=== Ring boundary errors (mean dist, px) ===');
  const ringNames = ['0 (X)', '1 (10)', '2 (9)', '3 (8)', '4 (7)', '5 (6)', '6 (5)', '7 (4)', '8 (3)', '9 (2)'];
  for (let i = 0; i < 10; i++) {
    const vals = processed
      .map(r => r.ring_mean_dist_px?.[i])
      .filter((v): v is number => v != null);
    if (!vals.length) { console.log(`  ring[${i}] ${ringNames[i]}: no data`); continue; }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const max  = Math.max(...vals);
    const bad  = vals.filter(v => v > 15).length;
    console.log(`  ring[${i}] ${ringNames[i].padEnd(8)}: mean=${mean.toFixed(1)}px  max=${max.toFixed(1)}px  bad(>15px)=${bad}/${vals.length}`);
  }

  // Per-ring relative radius errors
  console.log('\n=== Ring radius relative errors ===');
  for (let i = 0; i < 10; i++) {
    const vals = processed
      .map(r => r.ring_radius_rel_err?.[i])
      .filter((v): v is number => v != null);
    if (!vals.length) { console.log(`  ring[${i}]: no data`); continue; }
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const max  = Math.max(...vals);
    console.log(`  ring[${i}] ${ringNames[i].padEnd(8)}: mean=${(mean*100).toFixed(1)}%  max=${(max*100).toFixed(1)}%`);
  }

  // Arrow summary
  const totalAnn = results.reduce((s, r) => s + (r.arrows?.ann_count ?? 0), 0);
  const totalDet = results.reduce((s, r) => s + (r.arrows?.det_count ?? 0), 0);
  const totalTP  = results.reduce((s, r) => s + (r.arrows?.tp  ?? 0), 0);
  const totalFN  = results.reduce((s, r) => s + (r.arrows?.fn  ?? 0), 0);
  const totalFP  = results.reduce((s, r) => s + (r.arrows?.fp  ?? 0), 0);
  const recall    = totalAnn > 0 ? totalTP / totalAnn : 0;
  const precision = (totalTP + totalFP) > 0 ? totalTP / (totalTP + totalFP) : 0;

  console.log('\n=== Arrow detection ===');
  console.log(`  annotated=${totalAnn}  detected=${totalDet}  TP=${totalTP}  FN=${totalFN}  FP=${totalFP}`);
  console.log(`  recall=${(recall*100).toFixed(1)}%  precision=${(precision*100).toFixed(1)}%`);

  // Per-image breakdown
  console.log('\n=== Per-image detail ===');
  for (const r of results) {
    if (!r.detection_success) {
      console.log(`  FAIL  ${r.filename}`);
      continue;
    }
    const ringWorst = r.ring_mean_dist_px
      ? Math.max(...r.ring_mean_dist_px.filter((v: number|null) => v != null))
      : null;
    const arrows = r.arrows;
    const arrowStr = arrows
      ? `arrows TP=${arrows.tp}/${arrows.ann_count} FP=${arrows.fp}`
      : 'no arrows';
    console.log(`  OK    ${r.filename}  worst_ring=${ringWorst?.toFixed(1) ?? '?'}px  ${arrowStr}`);
  }

  console.log('\n--- raw JSON ---');
  console.log(JSON.stringify(results, null, 2));
}

main().catch(console.error).finally(() => db.end());
