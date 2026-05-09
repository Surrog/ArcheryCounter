#!/usr/bin/env tsx
/**
 * Accuracy report: computes paper boundary IoU, ring IoU, and arrow detection
 * metrics for all annotated images using data already in the DB.
 */
import { Pool } from 'pg';
import { sampleClosedSpline, splineRadius } from '../src/spline';

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

// ---------------------------------------------------------------------------
// Polygon helpers (same as groundTruth.test.ts)
// ---------------------------------------------------------------------------

function signedArea(poly: [number, number][]): number {
  let area = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    area += x1 * y2 - x2 * y1;
  }
  return area / 2;
}

function clipPolygon(subject: [number, number][], clip: [number, number][]): [number, number][] {
  let output = subject.slice();
  for (let i = 0; i < clip.length && output.length > 0; i++) {
    const input = output;
    output = [];
    const [ax, ay] = clip[i];
    const [bx, by] = clip[(i + 1) % clip.length];
    const inside = (px: number, py: number) =>
      (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= 0;
    const intersect = ([p1x, p1y]: [number, number], [p2x, p2y]: [number, number]): [number, number] => {
      const dx1 = p2x - p1x, dy1 = p2y - p1y;
      const dx2 = bx - ax,   dy2 = by - ay;
      const t = ((ax - p1x) * dy2 - (ay - p1y) * dx2) / (dx1 * dy2 - dy1 * dx2);
      return [p1x + t * dx1, p1y + t * dy1];
    };
    for (let j = 0; j < input.length; j++) {
      const curr = input[j];
      const prev = input[(j + input.length - 1) % input.length];
      const currIn = inside(curr[0], curr[1]);
      const prevIn = inside(prev[0], prev[1]);
      if (currIn) {
        if (!prevIn) output.push(intersect(prev, curr));
        output.push(curr);
      } else if (prevIn) {
        output.push(intersect(prev, curr));
      }
    }
  }
  return output;
}

function polyIoU(a: [number, number][], b: [number, number][]): number {
  const aCCW = signedArea(a) < 0 ? [...a].reverse() : a;
  const bCCW = signedArea(b) < 0 ? [...b].reverse() : b;
  const inter = clipPolygon(aCCW, bCCW);
  if (inter.length < 3) return 0;
  const interArea = Math.abs(signedArea(inter));
  const aArea = Math.abs(signedArea(aCCW));
  const bArea = Math.abs(signedArea(bCCW));
  const union = aArea + bArea - interArea;
  return union <= 0 ? 0 : interArea / union;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const { rows: annRows } = await db.query(
    'SELECT a.filename, a.paper_boundary, a.rings, a.arrows, g.paper_boundary as gen_pb, g.rings as gen_rings, g.arrows as gen_arrows ' +
    'FROM annotations a LEFT JOIN generated g USING(filename) ORDER BY a.filename'
  );

  let pbTotal = 0, pbCount = 0;
  let ringTotal = 0, ringCount = 0;
  let arrowTP = 0, arrowFN = 0;

  const pbFailures: string[] = [];
  const ringFailures: string[] = [];

  for (const row of annRows) {
    const filename: string = row.filename;
    const annPB: [number, number][] | null = row.paper_boundary;
    const annRings: { points: [number, number][] }[] = row.rings ?? [];
    const annArrows: { tip: [number, number] }[] = row.arrows ?? [];
    const genPB: [number, number][] | null = row.gen_pb;
    const genRings: { points: [number, number][] }[] = row.gen_rings ?? [];
    const genArrows: { tip: [number, number] }[] = row.gen_arrows ?? [];

    if (!genRings.length) {
      console.log(`  SKIP ${filename} (no generated data)`);
      continue;
    }

    // Paper boundary IoU
    if (annPB && genPB) {
      const iou = polyIoU(annPB, genPB);
      pbTotal += iou;
      pbCount++;
      if (iou < 0.8) pbFailures.push(`  ${filename.padEnd(35)} pb_iou=${iou.toFixed(3)}`);
    }

    // Ring IoU — greedy bipartite
    if (annRings.length > 0 && genRings.length > 0) {
      const annValid = annRings.filter(r => splineRadius(r) >= 1);
      const annPolys = annValid.map(r => sampleClosedSpline(r.points, 60));
      const genPolys = genRings.map(r => sampleClosedSpline(r.points, 60));

      const pairs: { ai: number; gi: number; iou: number }[] = [];
      for (let ai = 0; ai < annPolys.length; ai++) {
        for (let gi = 0; gi < genPolys.length; gi++) {
          pairs.push({ ai, gi, iou: polyIoU(annPolys[ai], genPolys[gi]) });
        }
      }
      pairs.sort((a, b) => b.iou - a.iou);

      const matchedA = new Set<number>(), matchedG = new Set<number>();
      const matched: number[] = [];
      for (const { ai, gi, iou } of pairs) {
        if (matchedA.has(ai) || matchedG.has(gi)) continue;
        matchedA.add(ai); matchedG.add(gi);
        matched.push(iou);
      }
      const avgRingIou = matched.length ? matched.reduce((s, v) => s + v, 0) / matched.length : 0;
      const lowIou = matched.filter(v => v < 0.5).length;
      ringTotal += avgRingIou;
      ringCount++;
      if (avgRingIou < 0.7 || lowIou > 2) {
        ringFailures.push(
          `  ${filename.padEnd(35)} avg_ring_iou=${avgRingIou.toFixed(3)} low(<0.5)=${lowIou}  ` +
          matched.map(v => v.toFixed(2)).join(' ')
        );
      }
    }

    // Arrow TP/FN (tip within 45px)
    const TIP_PX = 45;
    const allPairs: { ai: number; di: number; dist: number }[] = [];
    for (let ai = 0; ai < annArrows.length; ai++) {
      for (let di = 0; di < genArrows.length; di++) {
        allPairs.push({ ai, di, dist: Math.hypot(
          genArrows[di].tip[0] - annArrows[ai].tip[0],
          genArrows[di].tip[1] - annArrows[ai].tip[1],
        )});
      }
    }
    allPairs.sort((a, b) => a.dist - b.dist);
    const matchedA2 = new Set<number>(), matchedD2 = new Set<number>();
    for (const { ai, di, dist } of allPairs) {
      if (matchedA2.has(ai) || matchedD2.has(di)) continue;
      if (dist <= TIP_PX) { matchedA2.add(ai); matchedD2.add(di); }
    }
    arrowTP += matchedA2.size;
    arrowFN += annArrows.length - matchedA2.size;
  }

  console.log('\n=== ACCURACY REPORT ===\n');
  console.log(`Paper boundary IoU — mean: ${(pbTotal / pbCount).toFixed(3)} over ${pbCount} images`);
  if (pbFailures.length) {
    console.log(`  Images with pb_iou < 0.8 (${pbFailures.length}):`);
    pbFailures.forEach(l => console.log(l));
  } else {
    console.log('  All images have pb_iou >= 0.8');
  }

  console.log(`\nRing IoU — mean: ${(ringTotal / ringCount).toFixed(3)} over ${ringCount} images`);
  if (ringFailures.length) {
    console.log(`  Images with avg_ring_iou < 0.7 or >2 low rings (${ringFailures.length}):`);
    ringFailures.forEach(l => console.log(l));
  } else {
    console.log('  All images have good ring IoU');
  }

  const total = arrowTP + arrowFN;
  console.log(`\nArrow tip detection — TP=${arrowTP}, FN=${arrowFN}, recall=${total ? (arrowTP/total).toFixed(3) : 'n/a'}`);

  await db.end();
}

main().catch(e => { console.error(e); process.exit(1); });
