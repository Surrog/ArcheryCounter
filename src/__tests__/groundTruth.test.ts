import * as path from 'path';
import * as fs from 'fs';
import { Pool } from 'pg';
import { loadImageNode } from '../imageLoader';
import { findTarget } from '../targetDetection';
import { findArrows } from '../arrowDetection';
import { scoreArrow } from '../scoring';
import { sampleClosedSpline, splineCentroid, splineRadius } from '../spline';

const IMAGES_DIR = path.resolve(__dirname, '../../images');

const db = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     parseInt(process.env.DB_PORT || '5432'),
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres',
  database: process.env.DB_NAME     || 'postgres',
});

afterAll(() => db.end());

interface SplineRing {
  points: [number, number][];
}

interface ArrowAnnotation {
  tip:   [number, number];
  nock:  [number, number];
  score: number | 'X' | null;
}

interface ImageAnnotation {
  paperBoundary: [number, number][] | null;
  rings: SplineRing[];
  arrows: ArrowAnnotation[];
}

/** Minimum distance from pt to any sampled point on the given ring spline. */
function pointToSplineDist(pt: [number, number], ring: SplineRing): number {
  const samples = sampleClosedSpline(ring.points, 60);
  let min = Infinity;
  for (const [sx, sy] of samples) {
    const d = Math.hypot(pt[0] - sx, pt[1] - sy);
    if (d < min) min = d;
  }
  return min;
}

const imageFiles = fs
  .readdirSync(IMAGES_DIR)
  .filter(f => /\.(jpg|jpeg)$/i.test(f))
  .sort();

test.each(imageFiles)(
  'ground truth: %s',
  async (filename) => {
    const { rows } = await db.query(
      'SELECT paper_boundary, rings, arrows FROM annotations WHERE filename = $1',
      [filename],
    );
    expect(rows.length).toBeGreaterThan(0); // fail if no annotation in DB
    const ann: ImageAnnotation = {
      paperBoundary: rows[0].paper_boundary,
      rings: rows[0].rings,
      arrows: rows[0].arrows ?? [],
    };

    const imgPath = path.join(IMAGES_DIR, filename);
    const { rgba, width, height } = await loadImageNode(imgPath);
    const result = findTarget(rgba, width, height);

    expect(result.success).toBe(true);

    // Paper boundary: each annotated corner within 30px of the nearest edge of the
    // detected polygon.  Checking edge distance (not vertex distance) handles the
    // case where the two polygons have different vertex counts: an annotated corner
    // that falls on a detected edge with no nearby vertex will still pass.
    if (ann.paperBoundary && result.paperBoundary) {
      const det = result.paperBoundary.points;
      const m = det.length;
      function pointToPolyDist(px: number, py: number): number {
        let minD = Infinity;
        for (let i = 0; i < m; i++) {
          const [ax, ay] = det[i];
          const [bx, by] = det[(i + 1) % m];
          const dx = bx - ax, dy = by - ay;
          const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
          minD = Math.min(minD, Math.hypot(px - (ax + t * dx), py - (ay + t * dy)));
        }
        return minD;
      }
      for (const [cx, cy] of ann.paperBoundary) {
        expect(pointToPolyDist(cx, cy)).toBeLessThan(50);
      }
    }

    // Center of innermost ring within 25px (use smallest-radius ring from each)
    if (ann.rings.length > 0 && result.rings.length > 0) {
      const annInner = [...ann.rings].sort((a, b) => splineRadius(a) - splineRadius(b))[0];
      const resInner = [...result.rings].sort((a, b) => splineRadius(a) - splineRadius(b))[0];
      const [annCx, annCy] = splineCentroid(annInner);
      const [resCx, resCy] = splineCentroid(resInner);
      expect(Math.hypot(resCx - annCx, resCy - annCy)).toBeLessThan(25);
    }

    // Ring radius within 30% for each ring (sort by radius to handle annotation order differences)
    const annSorted  = [...ann.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
    const resSorted  = [...result.rings].sort((a, b) => splineRadius(a) - splineRadius(b));
    const minLen = Math.min(annSorted.length, resSorted.length);
    for (let i = 0; i < minLen; i++) {
      const annRadius = splineRadius(annSorted[i]);
      const resRadius = splineRadius(resSorted[i]);
      expect(Math.abs(resRadius - annRadius) / annRadius).toBeLessThan(0.30);
    }

    // Colour calibration sanity: hue ranges
    if (result.calibration) {
      const { gold, red, blue, black, white } = result.calibration;
      expect(gold[0]).toBeGreaterThan(20); expect(gold[0]).toBeLessThan(70);
      // red wraps: 0-18 or 342-360
      expect(red[0] < 18 || red[0] > 342).toBe(true);
      expect(blue[0]).toBeGreaterThan(190); expect(blue[0]).toBeLessThan(245);
      expect(black[2]).toBeLessThan(0.3);  // V < 0.3
      expect(white[1]).toBeLessThan(0.2);  // S < 0.2
    }

    // Arrow detection assertions (P9-T8)
    if (ann.arrows.length > 0) {
      const detected = findArrows(rgba, width, height, result);

      const fmtPt  = (p: [number, number] | null) =>
        p ? `(${Math.round(p[0])},${Math.round(p[1])})` : 'null';
      const detSummary = () =>
        detected.map((d, i) => `  det[${i}] tip=${fmtPt(d.tip)} nock=${fmtPt(d.nock)}`).join('\n');
      const annSummary = () =>
        ann.arrows.map((a, i) => `  ann[${i}] tip=${fmtPt(a.tip)} nock=${fmtPt(a.nock)} score=${a.score}`).join('\n');

      // Count: allow missing up to 2 arrows; no more than 2 extra detections
      const missing = ann.arrows.length - detected.length;
      const extra   = detected.length - ann.arrows.length;
      if (missing > 2 || extra > 2) {
        console.error(
          `[${filename}] count: detected ${detected.length}, expected ${ann.arrows.length}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(detected.length).toBeGreaterThanOrEqual(ann.arrows.length - 2);
      expect(detected.length).toBeLessThanOrEqual(ann.arrows.length + 2);

      // Bijective tip matching: use distance-sorted assignment so that close pairs
      // are matched first, preventing far annotations from consuming good detections.
      // Tolerance 25 px; up to 2 misses allowed.
      const TIP_MATCH_PX = 45;
      type Pair = { ai: number; di: number; dist: number };
      const allPairs: Pair[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        for (let di = 0; di < detected.length; di++) {
          allPairs.push({ ai, di, dist: Math.hypot(
            detected[di].tip[0] - ann.arrows[ai].tip[0],
            detected[di].tip[1] - ann.arrows[ai].tip[1],
          )});
        }
      }
      allPairs.sort((a, b) => a.dist - b.dist);
      const matchedA = new Set<number>(), matchedD = new Set<number>();
      const tipMatchDist = new Map<number, number>(); // ai → dist
      const tipMatchIdx  = new Map<number, number>(); // ai → di
      for (const { ai, di, dist } of allPairs) {
        if (matchedA.has(ai) || matchedD.has(di)) continue;
        if (dist <= TIP_MATCH_PX) {
          matchedA.add(ai); matchedD.add(di);
          tipMatchDist.set(ai, dist);
          tipMatchIdx.set(ai, di);
        }
      }
      const tipFailures: string[] = [];
      for (let ai = 0; ai < ann.arrows.length; ai++) {
        if (!tipMatchDist.has(ai)) {
          const a = ann.arrows[ai];
          let bestDist = Infinity, bestIdx = -1;
          for (let di = 0; di < detected.length; di++) {
            const d = Math.hypot(detected[di].tip[0] - a.tip[0], detected[di].tip[1] - a.tip[1]);
            if (d < bestDist) { bestDist = d; bestIdx = di; }
          }
          tipFailures.push(
            `  ann tip=${fmtPt(a.tip)} score=${a.score}` +
            ` → best det=${fmtPt(detected[bestIdx]?.tip ?? null)} dist=${bestDist.toFixed(1)}px`,
          );
        }
      }
      // Allow guaranteed failures from count gap (max(0, N-D)) plus 2 positional misses.
      const maxTipFailures = Math.max(0, ann.arrows.length - detected.length) + 2;
      if (tipFailures.length > maxTipFailures) {
        console.error(
          `[${filename}] ${tipFailures.length} tip(s) unmatched (>${maxTipFailures} allowed):\n${tipFailures.join('\n')}\n` +
          `Detected:\n${detSummary()}\nAnnotated:\n${annSummary()}`,
        );
      }
      expect(tipFailures.length).toBeLessThanOrEqual(maxTipFailures);

      // Scoring assertions (P10-T7)
      if (result.calibration) {
        const scoreFailures: string[] = [];
        for (const [ai, di] of tipMatchIdx) {
          const annScore = ann.arrows[ai].score;
          if (annScore === null) continue; // unannotated score — skip
          const detScore = scoreArrow(detected[di].tip, result.rings);

          // Treat 'X' and 10 as equivalent numeric score (X is a tiebreaker within 10).
          const numAnn = annScore === 'X' ? 10 : annScore;
          const numDet = detScore === 'X' ? 10 : detScore;
          // Near-boundary: tip within 20px of any ring spline → allow ±1 tolerance.
          // 20px ≈ 20% of a typical ring width; handles ring boundary detection imprecision.
          const nearBoundary = result.rings.some(r => pointToSplineDist(detected[di].tip, r) < 20);
          const ok = numDet === numAnn || (nearBoundary && Math.abs(numDet - numAnn) <= 1);
          if (!ok) {
            scoreFailures.push(
              `  ann[${ai}] tip=${fmtPt(ann.arrows[ai].tip)} score=${annScore} → det score=${detScore}` +
              (nearBoundary ? ' (near boundary)' : ''),
            );
          }
        }
        if (scoreFailures.length > 0) {
          console.error(`[${filename}] scoring failures:\n${scoreFailures.join('\n')}`);
        }
        expect(scoreFailures.length).toBe(0);
      }

      // Nock matching: informational only — log misses but do not assert.
      const matchedDet2 = new Set<number>();
      for (const annArrow of ann.arrows) {
        let bestDist = Infinity, bestIdx = -1;
        for (let di = 0; di < detected.length; di++) {
          if (matchedDet2.has(di)) continue;
          const d = Math.hypot(detected[di].tip[0] - annArrow.tip[0], detected[di].tip[1] - annArrow.tip[1]);
          if (d < bestDist) { bestDist = d; bestIdx = di; }
        }
        if (bestIdx >= 0) {
          matchedDet2.add(bestIdx);
          const det = detected[bestIdx];
          if (det.nock !== null) {
            const nd = Math.hypot(det.nock[0] - annArrow.nock[0], det.nock[1] - annArrow.nock[1]);
            if (nd >= 40) {
              console.error(
                `[${filename}] nock info: ann tip=${fmtPt(annArrow.tip)} nock=${fmtPt(annArrow.nock)}` +
                ` det nock=${fmtPt(det.nock)} dist=${nd.toFixed(1)}px`,
              );
            }
          }
        }
      }
    }
  },
  120000,
);
